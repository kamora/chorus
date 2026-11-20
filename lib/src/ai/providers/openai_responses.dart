// The OpenAI Responses provider, ported from pi's api/openai-responses.ts and api/openai-responses-shared.ts.
//
// Not ported: grammar (custom) tools, which every model here treats as unsupported, so a grammar tool is sent as an
// ordinary function tool, as pi does for models without them; Sign in with ChatGPT tokens; Copilot headers; sampling
// parameters; the WebSocket transport.

import 'dart:async';
import 'dart:convert';
import 'dart:math';

import 'package:http/http.dart' as http;
import 'package:openai_sdk_lite_dart/openai_sdk_lite_dart.dart';

import '../constrained.dart';
import '../event_stream.dart';
import '../hash.dart';
import '../json.dart';
import '../model.dart';
import '../options.dart';
import '../registry.dart' show clampThinkingLevel;
import '../retry.dart';
import '../transcript.dart';
import '../transform.dart';
import '../types.dart';
import 'shared.dart';

/// Options of the OpenAI Responses provider, ported from pi's `OpenAIResponsesOptions`.
class OpenAIResponsesOptions extends StreamOptions {
  const OpenAIResponsesOptions({
    super.signal,
    super.apiKey,
    super.headers,
    super.httpClient,
    super.timeout,
    super.maxRetries,
    super.maxRetryDelay,
    super.temperature,
    super.maxTokens,
    super.cacheRetention,
    super.sessionId,
    super.metadata,
    super.onPayload,
    super.onResponse,
    super.onEvent,
    super.disableContinuation,
    this.reasoningEffort,
    this.reasoningSummary,
    this.serviceTier,
    this.toolChoice,
  });

  /// The provider-neutral [options] as OpenAI Responses options, with none of the provider's own set.
  factory OpenAIResponsesOptions.from(StreamOptions options) {
    return OpenAIResponsesOptions(
      signal: options.signal,
      apiKey: options.apiKey,
      headers: options.headers,
      httpClient: options.httpClient,
      timeout: options.timeout,
      maxRetries: options.maxRetries,
      maxRetryDelay: options.maxRetryDelay,
      temperature: options.temperature,
      maxTokens: options.maxTokens,
      cacheRetention: options.cacheRetention,
      sessionId: options.sessionId,
      metadata: options.metadata,
      onPayload: options.onPayload,
      onResponse: options.onResponse,
      onEvent: options.onEvent,
      disableContinuation: options.disableContinuation,
    );
  }

  /// `minimal`, `low`, `medium`, `high`, `xhigh` or `max`; mapped through the model's thinking level map.
  final String? reasoningEffort;

  /// `auto` (the default with reasoning), `detailed` or `concise`.
  final String? reasoningSummary;

  /// `auto`, `default`, `flex`, `priority`...; flex and priority change the cost.
  final String? serviceTier;

  /// `auto`, `none`, `required`, or a tool to force.
  final Object? toolChoice;
}

/// Providers whose tool call ids carry a Responses item id (`call|item`) worth keeping.
const _toolCallProviders = {'openai', 'openai-codex', 'opencode'};

/// OpenAI Responses rejects `max_output_tokens` below 16.
const _minOutputTokens = 16;

/// The `reasoning.effort` sent while `configuration_update` items carry the thinking level.
const _configurationUpdateRequestEffort = 'medium';

/// OpenAI's limit on `prompt_cache_key`, in characters.
const promptCacheKeyMaxLength = 64;

/// The `compat` settings this provider reads, with their defaults.
class _Compat {
  _Compat(Model model)
    : developerRole = model.compat['supportsDeveloperRole'] as bool? ?? true,
      midSystem = model.compat['supportsMidConvoSystemMessages'] as bool? ?? false,
      affinityFormat = model.compat['sessionAffinityFormat'] as String? ?? _affinityFormat(model),
      cacheKeyMode = model.compat['promptCacheKeyMode'] as String? ?? 'auto',
      longCache = model.compat['supportsLongCacheRetention'] as bool? ?? true,
      strict = model.compat['supportsStrictMode'] as bool? ?? false,
      additionalTools = model.compat['supportsAdditionalTools'] as bool? ?? false,
      toolSearch = model.compat['supportsToolSearch'] as bool? ?? false,
      explicitCache = model.compat['supportsExplicitPromptCacheMode'] as bool? ?? false,
      maxOutputTokens = model.compat['supportsMaxOutputTokens'] as bool? ?? true,
      midEffort = model.compat['supportsMidConvoEffort'] as bool? ?? false;

  final bool developerRole;

  /// Later system messages are sent in place; otherwise they fold into the leading prompt.
  final bool midSystem;

  /// `openai`, `openai-nosession` or `openrouter`: which session headers go with a session.
  final String affinityFormat;

  /// `auto`, `enabled` or `disabled`: whether the session goes as `prompt_cache_key`.
  final String cacheKeyMode;
  final bool longCache;
  final bool strict;
  final bool additionalTools;
  final bool toolSearch;

  /// Prompt caching is set with `prompt_cache_options`, not `prompt_cache_retention`.
  final bool explicitCache;
  final bool maxOutputTokens;

  /// The thinking level goes in `configuration_update` items, so changing it keeps the cached prefix.
  final bool midEffort;

  /// How tool changes are sent: `additional_tools` items need system messages in place to anchor them.
  ToolSupport get toolSupport {
    if (midSystem && additionalTools) {
      return ToolSupport.changes;
    }

    return toolSearch ? ToolSupport.additions : ToolSupport.none;
  }

  static String _affinityFormat(Model model) {
    if (model.provider == 'openrouter' || model.baseUrl.contains('openrouter.ai')) {
      return 'openrouter';
    }

    return 'openai';
  }
}

/// [key] cut to [promptCacheKeyMaxLength] characters.
String? clampPromptCacheKey(String? key) {
  if (key == null || key.runes.length <= promptCacheKeyMaxLength) {
    return key;
  }

  return String.fromCharCodes(key.runes.take(promptCacheKeyMaxLength));
}

/// The effort the `configuration_update` items carry, or null to send it as `reasoning.effort`.
String? _configurationEffort(Model model, OpenAIResponsesOptions options) {
  final effort = options.reasoningEffort;
  if (!_Compat(model).midEffort || effort == null) {
    return null;
  }

  return model.levels?[effort] ?? effort;
}

/// A text block's signature: the Responses message id and its phase.
String _textSignature(String? id, String? phase) => jsonEncode({'v': 1, 'id': ?id, 'phase': ?phase});

({String id, String? phase})? _parseTextSignature(String? signature) {
  if (signature == null || signature.isEmpty) {
    return null;
  }

  if (signature.startsWith('{')) {
    try {
      final parsed = jsonDecode(signature);
      if (parsed case {'v': 1, 'id': final String id}) {
        return (
          id: id,
          phase: switch (parsed['phase']) {
            final String phase when phase == 'commentary' || phase == 'final_answer' => phase,
            _ => null,
          },
        );
      }
    } on FormatException {
      // A plain message id.
    }
  }

  return (id: signature, phase: null);
}

/// A tool result's output: its text, or input items when it has images. transformMessages has already replaced the
/// images of a model without image input, so pi's "(see attached image)" for them is never reached and left out.
Object _toolResultOutput(List<Content> content) {
  final text = content.whereType<TextContent>().map((c) => c.text).join('\n');
  final images = content.whereType<ImageContent>().toList();
  final hasText = text.trim().isNotEmpty;
  if (images.isEmpty) {
    return hasText ? sanitizeSurrogates(text) : '(no tool output)';
  }

  return [
    if (hasText) {'type': 'input_text', 'text': sanitizeSurrogates(text)},
    for (final image in images)
      {'type': 'input_image', 'detail': 'auto', 'image_url': 'data:${image.mimeType};base64,${image.data}'},
  ];
}

/// Precedes every `additional_tools` item after the first: without it, models read the latest item as the whole tool
/// set once a removal was announced, and smaller ones call removed tools again.
String _incrementalToolsNotice(Set<String> unavailable) {
  const notice =
      'This is an incremental tools update. Previously declared tools remain available unless explicitly marked '
      'unavailable. A tool declared again here is available again, and its latest definition replaces the earlier '
      'one.';
  return unavailable.isEmpty ? notice : '$notice\nStill unavailable: ${unavailable.join(', ')}.';
}

String _toolRemovalNotice(List<String> names) {
  return 'The following tools are no longer available. Do not call them:\n${names.map((n) => '- $n').join('\n')}';
}

/// [part] of a tool call id as Responses accepts it: `[a-zA-Z0-9_-]`, at most 64 characters, no trailing `_`.
String _idPart(String part) {
  final safe = part.replaceAll(RegExp('[^a-zA-Z0-9_-]'), '_');
  return (safe.length > 64 ? safe.substring(0, 64) : safe).replaceFirst(RegExp(r'_+$'), '');
}

/// The Responses request input for [messages], ported from pi's `convertResponsesMessages`.
///
/// System messages become `developer` messages (or `system` ones for models without reasoning or the developer role);
/// later ones are sent in place only when the model accepts them, otherwise they fold into the leading prompt. Tool
/// changes follow the model's [ToolSupport]: `additional_tools` items, client tool search items, or none. With
/// [configurationEffort], the thinking level goes in `configuration_update` items: one before every earlier reply of
/// this model with the level it was given, and one with this effort at the end.
List<Json> convertResponsesMessages(Model model, List<Message> messages, {String? configurationEffort}) {
  final compat = _Compat(model);
  final transcript = compat.midSystem ? messages : collapseSystemMessages(messages);
  final input = <Json>[];

  String normalizeId(String id, AssistantMessage source) {
    if (!_toolCallProviders.contains(model.provider) || !id.contains('|')) {
      return _idPart(id);
    }

    final [call, item, ...] = id.split('|');

    // Item ids must start with fc_. A foreign one is hashed into that form; one of another model of this provider is
    // dropped when the call is converted, so pi's fc_ prefix for it is left out.
    var itemId = _idPart(item);
    if (source.provider != model.provider || source.api != model.api) {
      itemId = 'fc_${shortHash(item)}';
    }

    return '${_idPart(call)}|${itemId.length > 64 ? itemId.substring(0, 64) : itemId}';
  }

  final support = compat.toolSupport;
  final folded = resolveTranscriptTools(transcript, support).folded;
  var role = 'system';
  if (model.reasoning && compat.developerRole) {
    role = 'developer';
  }

  var declared = false;

  // Tools announced as removed and not declared again since.
  final unavailable = <String>{};

  void addTools(SystemMessage message, String seed) {
    final tools = message.added ?? const <Tool>[];
    unavailable.removeAll(tools.map((t) => t.name));
    if (tools.isEmpty) {
      return;
    }

    switch (support) {
      case ToolSupport.changes:
        if (declared) {
          input.add({'role': role, 'content': _incrementalToolsNotice(unavailable)});
        }

        declared = true;
        input.add({'type': 'additional_tools', 'role': 'developer', 'tools': convertResponsesTools(tools, model)});
      case ToolSupport.additions:
        final names = tools.map((t) => t.name).toList();
        final call = 'pi_tool_load_${shortHash('$seed:${names.join(',')}')}';
        input
          ..add({
            'type': 'tool_search_call',
            'call_id': call,
            'execution': 'client',
            'status': 'completed',
            'arguments': {'query': names.join(' '), 'limit': names.length},
          })
          ..add({
            'type': 'tool_search_output',
            'call_id': call,
            'execution': 'client',
            'status': 'completed',
            'tools': convertResponsesTools(tools, model, searchResult: true),
          });
      case ToolSupport.none:
        return;
    }
  }

  void configure(String effort) {
    input.add({
      'type': 'configuration_update',
      'reasoning': {'effort': effort},
    });
  }

  var index = 0;
  var systems = 0;
  for (final (position, message) in transformMessages(transcript, model, normalizeId).indexed) {
    final leading = position == 0 && message is SystemMessage;
    switch (message) {
      case SystemMessage():
        // System messages keep their order through transformMessages, so their count matches the transcript's.
        final anchors = systems++ >= folded;
        if (anchors) {
          addTools(message, 'system:$index');
        }

        final parts = [leading ? systemText(message) : renderSystemMessageUpdate(message)];
        if (anchors) {
          // A redefinition is a removal and an addition of the same name; only true removals are announced.
          final added = {for (final t in message.added ?? const <Tool>[]) t.name};
          final removed = [...?message.removed?.where((name) => !added.contains(name))];
          if (removed.isNotEmpty) {
            parts.add(_toolRemovalNotice(removed));
          }

          unavailable.addAll(removed);
        }

        final text = parts.where((p) => p.isNotEmpty).join('\n\n');
        if (text.isNotEmpty) {
          input.add({'role': role, 'content': sanitizeSurrogates(text)});
        }
      case UserMessage():
        if (message.content.isEmpty) {
          continue;
        }

        input.add({
          'role': 'user',
          'content': [
            for (final c in message.content)
              switch (c) {
                ImageContent() => {
                  'type': 'input_image',
                  'detail': 'auto',
                  'image_url': 'data:${c.mimeType};base64,${c.data}',
                },
                _ => {'type': 'input_text', 'text': sanitizeSurrogates((c as TextContent).text)},
              },
          ],
        });
      case AssistantMessage():
        final provider = message.provider == model.provider && message.api == model.api;
        final sameModel = provider && message.model == model.id;
        final output = <Json>[];
        var texts = 0;
        for (final block in message.content) {
          switch (block) {
            case ThinkingContent(:final signature?) when signature.isNotEmpty:
              output.add(jsonDecode(signature) as Json);
            case TextContent():
              final parsed = _parseTextSignature(block.signature);
              var id = parsed?.id ?? '';
              if (id.isEmpty) {
                id = texts == 0 ? 'msg_pi_$index' : 'msg_pi_${index}_$texts';
              } else if (id.length > 64) {
                id = 'msg_${shortHash(id)}';
              }

              texts++;
              output.add({
                'type': 'message',
                'role': 'assistant',
                'content': [
                  {'type': 'output_text', 'text': sanitizeSurrogates(block.text), 'annotations': <Object?>[]},
                ],
                'status': 'completed',
                'id': id,
                'phase': ?parsed?.phase,
              });
            case ToolCall():
              final parts = block.id.split('|');
              String? item = parts.length > 1 ? parts[1] : null;

              // A reply of another model of this provider goes without item ids: OpenAI checks which item ids were
              // paired with its reasoning items. An id of another item type is dropped too.
              if ((provider && !sameModel) || !(item?.startsWith('fc_') ?? false)) {
                item = null;
              }

              output.add({
                'type': 'function_call',
                'id': ?item,
                'call_id': parts.first,
                'name': block.name,
                'arguments': jsonEncode(block.arguments),
                if (sameModel) 'namespace': ?block.namespace,
              });
            default:
              continue;
          }
        }

        if (output.isEmpty) {
          continue;
        }

        final level = message.providerThinkingLevel;
        if (configurationEffort != null && sameModel && level != null) {
          configure(level);
        }

        input.addAll(output);
      case ToolResultMessage():
        input.add({
          'type': 'function_call_output',
          'call_id': message.id.split('|').first,
          'output': _toolResultOutput(message.content),
        });
    }

    if (!leading) {
      index++;
    }
  }

  if (configurationEffort != null) {
    configure(configurationEffort);
  }

  return input;
}

/// [tools] as Responses function tools; strict when the model has strict mode and a tool asks for it.
/// [searchResult] marks them for a client tool search result, which loads them deferred.
List<Json> convertResponsesTools(List<Tool> tools, Model model, {bool searchResult = false}) {
  final supported = _Compat(model).strict;
  return [
    for (final tool in tools)
      () {
        final strict = resolveStrictSampling(tool, supported) ?? false;
        return {
          'type': 'function',
          'name': tool.name,
          'description': tool.description,
          'parameters': strictToolParameters(tool, strict),
          if (searchResult) 'defer_loading': true,
          if (supported) 'strict': strict,
        };
      }(),
  ];
}

/// The request body for [messages], ported from pi's `buildParams`.
Json buildResponsesParams(Model model, List<Message> messages, OpenAIResponsesOptions options) {
  final compat = _Compat(model);
  final transcript = compat.midSystem ? messages : collapseSystemMessages(messages);
  final configurationEffort = _configurationEffort(model, options);
  final retention = options.cacheRetention ?? CacheRetention.short;
  String? retention24h;
  if (retention == CacheRetention.long && compat.longCache && !compat.explicitCache) {
    retention24h = '24h';
  }

  Json? cacheOptions;
  if (compat.explicitCache) {
    if (retention == CacheRetention.none) {
      cacheOptions = {'mode': 'explicit'};
    } else if (retention == CacheRetention.long && compat.longCache) {
      cacheOptions = {'ttl': '30m'};
    }
  }

  final params = <String, Object?>{
    'model': model.id,
    'input': convertResponsesMessages(model, transcript, configurationEffort: configurationEffort),
    'stream': true,
    if (retention != CacheRetention.none && compat.cacheKeyMode != 'disabled')
      'prompt_cache_key': ?clampPromptCacheKey(options.sessionId),
    'prompt_cache_retention': ?retention24h,
    'prompt_cache_options': ?cacheOptions,
    'store': false,
  };
  final maxTokens = options.maxTokens;
  if (maxTokens != null && maxTokens != 0 && compat.maxOutputTokens) {
    params['max_output_tokens'] = max(maxTokens, _minOutputTokens);
  }

  if (options.temperature != null) {
    params['temperature'] = options.temperature;
  }

  if (options.serviceTier != null) {
    params['service_tier'] = options.serviceTier;
  }

  final tools = resolveTranscriptTools(transcript, compat.toolSupport).tools;
  if (tools.isNotEmpty) {
    params['tools'] = convertResponsesTools(tools, model);
  }

  if (options.toolChoice != null) {
    params['tool_choice'] = options.toolChoice;
  }

  if (!model.reasoning) {
    return params;
  }

  final summary = options.reasoningSummary;
  final requested = options.reasoningEffort;
  if (requested != null || (summary?.isNotEmpty ?? false)) {
    var effort = requested == null ? 'medium' : model.levels?[requested] ?? requested;
    if (configurationEffort != null) {
      effort = _configurationUpdateRequestEffort;
    }

    params['reasoning'] = {'effort': effort, 'summary': (summary?.isNotEmpty ?? false) ? summary : 'auto'};
    params['include'] = ['reasoning.encrypted_content'];
  } else if (model.provider != 'github-copilot' && !_offDisabled(model)) {
    params['reasoning'] = {'effort': model.levels?['off'] ?? 'none'};
  }

  if (model.provider == 'xai') {
    params['include'] = ['reasoning.encrypted_content'];
  }

  return params;
}

// The model's level map turns thinking off by leaving it out (`off: null`).
bool _offDisabled(Model model) {
  final levels = model.levels;
  if (levels == null || !levels.containsKey('off')) {
    return false;
  }

  return levels['off'] == null;
}

/// How much [tier] multiplies the cost: flex halves it, priority (OpenAI's "fast") doubles it, 2.5 times for gpt-5.5.
double _tierMultiplier(Model model, String? tier) {
  return switch (tier) {
    'flex' => 0.5,
    'priority' || 'fast' => model.id == 'gpt-5.5' ? 2.5 : 2,
    _ => 1,
  };
}

void _applyTier(Model model, Usage usage, String? tier) {
  final multiplier = _tierMultiplier(model, tier);
  if (multiplier == 1) {
    return;
  }

  final cost = usage.cost
    ..input *= multiplier
    ..output *= multiplier
    ..cacheRead *= multiplier
    ..cacheWrite *= multiplier;
  cost.total = cost.input + cost.output + cost.cacheRead + cost.cacheWrite;
}

/// The stop reason of a terminal response [status], and the error message that goes with it.
({StopReason reason, String? error}) _stopReason(String? status, String? incomplete) {
  return switch (status) {
    null || 'completed' || 'in_progress' || 'queued' => (reason: StopReason.stop, error: null),
    'incomplete' when incomplete == 'max_output_tokens' => (reason: StopReason.length, error: null),
    'incomplete' when incomplete != null => (reason: StopReason.error, error: 'Response incomplete: $incomplete'),
    'incomplete' => (reason: StopReason.error, error: 'Response incomplete without a provider reason'),
    'failed' || 'cancelled' => (reason: StopReason.error, error: null),
    _ => throw ProviderException('Unhandled stop reason: $status'),
  };
}

// JavaScript's string for a missing value, where pi builds a string from one.
String _js(Object? value) => value == null ? 'undefined' : '$value';

// JavaScript's truthiness: false for a missing, empty, zero or false value.
bool _truthy(Object? value) {
  return switch (value) {
    null || '' || 0 || false => false,
    _ => true,
  };
}

// JavaScript's `||`: [value] when truthy, else [fallback].
T _or<T>(Object? value, T fallback) => _truthy(value) ? value as T : fallback;

/// The texts of a reasoning item's [key] parts (`summary` or `content`), joined with blank lines.
String _partsText(Json item, String key) {
  return [for (final p in item[key] as List<Object?>? ?? const []) (p as Json)['text'] ?? ''].join('\n\n');
}

/// An output item being streamed: where its block sits in the output, and a function call's argument text.
typedef _Slot = ({Content block, int index, ToolArgumentsStream? arguments});

/// Reads Responses [events] into [output] and [stream], ported from pi's `processResponsesStream`.
///
/// Throws when the stream fails, ends without a terminal response event, or completes with a tool call whose
/// `output_item.done` never arrived.
Future<void> processResponsesStream(
  Stream<Json> events,
  AssistantMessage output,
  AssistantMessageEventStream stream,
  Model model, {
  FutureOr<void> Function(Json event, Model model)? onEvent,
  String? serviceTier,
}) async {
  var terminal = false;

  // By output_index, as sent: a server that leaves it out puts every item under the same missing key, as pi's does.
  final slots = <Object?, _Slot>{};

  // Tool calls whose output_item.done has not arrived yet.
  final unfinished = Set<ToolCall>.identity();
  final reasoning = <Object?, ThinkingContent>{};

  void phase(Json item) {
    if (item['type'] == 'message' && item['phase'] == 'final_answer') {
      output.stopReason = StopReason.stop;
    }
  }

  _Slot? open(Object? index, Json item) {
    final Content block;
    ToolArgumentsStream? arguments;
    switch (item['type']) {
      case 'reasoning':
        block = ThinkingContent('');
      case 'message':
        phase(item);
        block = TextContent('');
      case 'function_call':
        arguments = ToolArgumentsStream();
        final text = item['arguments'] as String?;
        block = ToolCall(
          '${_js(item['call_id'])}|${_js(item['id'])}',
          item['name'] as String? ?? '',
          (text?.isNotEmpty ?? false) ? arguments.append(text!) : {},
          namespace: item['namespace'] as String?,
        );
        unfinished.add(block as ToolCall);
      default:
        return null;
    }

    output.content.add(block);
    final slot = (block: block, index: output.content.length - 1, arguments: arguments);
    slots[index] = slot;
    stream.push(switch (block) {
      ThinkingContent() => ThinkingStartEvent(slot.index, output),
      TextContent() => TextStartEvent(slot.index, output),
      _ => ToolCallStartEvent(slot.index, output),
    });
    return slot;
  }

  // Azure OpenAI can leave reasoning.encrypted_content out of output_item.done and send it only with the terminal
  // response; it goes into the stored reasoning item so the next turn can replay it without server state.
  void backfill(List<Object?> items) {
    for (final item in items.whereType<Json>()) {
      final encrypted = item['encrypted_content'];
      if (item['type'] != 'reasoning' || !_truthy(encrypted)) {
        continue;
      }

      final block = reasoning[item['id']];
      final signature = block?.signature;
      if (signature == null || signature.isEmpty) {
        continue;
      }

      final stored = jsonDecode(signature) as Json;
      if (_truthy(stored['encrypted_content'])) {
        continue;
      }

      block!.signature = jsonEncode({...stored, 'encrypted_content': encrypted});
    }
  }

  void finish(Json response) {
    terminal = true;
    backfill(response['output'] as List<Object?>? ?? const []);
    final id = response['id'] as String?;
    if (id?.isNotEmpty ?? false) {
      output.responseId = id;
    }

    if (response['usage'] case final Json usage) {
      final details = usage['input_tokens_details'] as Json?;
      final cached = _or<num>(details?['cached_tokens'], 0).toInt();
      final written = _or<num>(details?['cache_write_tokens'], 0).toInt();

      // OpenAI counts cached and cache-write tokens in input_tokens.
      output.usage = Usage(
        input: max(0, _or<num>(usage['input_tokens'], 0).toInt() - cached - written),
        output: _or<num>(usage['output_tokens'], 0).toInt(),
        cacheRead: cached,
        cacheWrite: written,
        reasoning: _or<num>((usage['output_tokens_details'] as Json?)?['reasoning_tokens'], 0).toInt(),
        totalTokens: _or<num>(usage['total_tokens'], 0).toInt(),
      );
    }

    calculateCost(model, output.usage);
    _applyTier(model, output.usage, response['service_tier'] as String? ?? serviceTier);

    // An incomplete response keeps its reason, so truncation and content filtering stay apart.
    final status = response['status'] as String?;
    final incomplete = switch (response['incomplete_details']) {
      {'reason': final String reason} => reason,
      _ => null,
    };
    output.rawStopReason = incomplete == null ? status : '${_js(status)}.$incomplete';
    final mapped = _stopReason(status, incomplete);
    output
      ..stopReason = mapped.reason
      ..errorMessage = mapped.error;
    if (output.stopReason == StopReason.stop && output.content.any((c) => c is ToolCall)) {
      output.stopReason = StopReason.toolUse;
    }
  }

  _Slot? slot<T extends Content>(Json event) {
    final found = slots[event['output_index']];
    return found?.block is T ? found : null;
  }

  await for (final event in events) {
    await onEvent?.call(event, model);
    switch (event['type']) {
      case 'response.created':
        output.responseId = (event['response'] as Json?)?['id'] as String?;
      case 'response.output_item.added':
        open(event['output_index'], event['item'] as Json);
      case 'response.reasoning_summary_text.delta' || 'response.reasoning_text.delta':
        if (slot<ThinkingContent>(event) case final s?) {
          final delta = event['delta'] as String;
          (s.block as ThinkingContent).thinking += delta;
          stream.push(ThinkingDeltaEvent(s.index, delta, output));
        }
      case 'response.reasoning_summary_part.done':
        if (slot<ThinkingContent>(event) case final s?) {
          (s.block as ThinkingContent).thinking += '\n\n';
          stream.push(ThinkingDeltaEvent(s.index, '\n\n', output));
        }
      case 'response.output_text.delta' || 'response.refusal.delta':
        if (slot<TextContent>(event) case final s?) {
          final delta = event['delta'] as String;
          (s.block as TextContent).text += delta;
          stream.push(TextDeltaEvent(s.index, delta, output));
        }
      case 'response.function_call_arguments.delta':
        if (slot<ToolCall>(event) case final s?) {
          final delta = event['delta'] as String;
          (s.block as ToolCall).arguments = s.arguments!.append(delta);
          stream.push(ToolCallDeltaEvent(s.index, delta, output));
        }
      case 'response.function_call_arguments.done':
        if (slot<ToolCall>(event) case final s?) {
          final replaced = s.arguments!.replace(event['arguments'] as String);
          (s.block as ToolCall).arguments = replaced.arguments;
          if (replaced.delta?.isNotEmpty ?? false) {
            stream.push(ToolCallDeltaEvent(s.index, replaced.delta!, output));
          }
        }
      case 'response.output_item.done':
        final item = event['item'] as Json;
        final index = event['output_index'];
        phase(item);
        final s = slots[index] ?? open(index, item);
        switch ((item['type'], s?.block)) {
          case ('reasoning', final ThinkingContent block):
            block
              ..thinking = _or(_partsText(item, 'summary'), _or(_partsText(item, 'content'), block.thinking))
              ..signature = jsonEncode(item);
            reasoning[item['id']] = block;
            stream.push(ThinkingEndEvent(s!.index, block.thinking, output));
            slots.remove(index);
          case ('message', final TextContent block):
            block
              ..text = [
                for (final c in item['content'] as List<Object?>? ?? const [])
                  ((c as Json)['type'] == 'output_text' ? c['text'] : c['refusal']) ?? '',
              ].join()
              ..signature = _textSignature(item['id'] as String?, item['phase'] as String?);
            stream.push(TextEndEvent(s!.index, block.text, output));
            slots.remove(index);
          case ('function_call', final ToolCall block) when s!.arguments != null:
            final text = item['arguments'] as String?;
            if (text?.isNotEmpty ?? false) {
              s.arguments!.replace(text!);
            }

            final finished = s.arguments!.finish();
            block
              ..arguments = finished.arguments
              ..argumentsError = finished.error;
            if (item['namespace'] case final String namespace) {
              block.namespace = namespace;
            }

            unfinished.remove(block);
            stream.push(ToolCallEndEvent(s.index, block, output));
            slots.remove(index);
        }
      case 'response.completed' || 'response.incomplete':
        finish(event['response'] as Json);
      case 'error':
        throw ProviderException('Error Code ${_js(event['code'])}: ${_js(event['message'])}');
      case 'response.failed':
        terminal = true;
        final response = event['response'] as Json?;
        output.rawStopReason = response?['status'] as String?;
        final error = response?['error'];
        final details = response?['incomplete_details'];
        if (error is Map) {
          throw ProviderException('${_or(error['code'], 'unknown')}: ${_or(error['message'], 'no message')}');
        }

        if (details is Map && _truthy(details['reason'])) {
          throw ProviderException('incomplete: ${details['reason']}');
        }

        throw const ProviderException('Unknown error (no error details in response)');
    }
  }

  if (!terminal) {
    throw const ProviderException('OpenAI Responses stream ended before a terminal response event');
  }

  // The agent runs every tool call of the final message; a call whose output_item.done never came may have cut off or
  // mixed-up arguments, as when a server leaves out output_index.
  if (output.stopReason == StopReason.toolUse && unfinished.isNotEmpty) {
    final call = unfinished.first;
    throw ProviderException(
      'OpenAI Responses stream completed with an unfinished tool call: ${call.name} (${call.id})',
    );
  }
}

/// The API key the client sends: [StreamOptions.apiKey], or a placeholder when a header carries the credentials.
/// Throws a [ProviderException] without either.
String _apiKey(Model model, StreamOptions options) {
  final key = options.apiKey;
  if (key != null && key.isNotEmpty) {
    return key;
  }

  for (final MapEntry(:key, :value) in (options.headers ?? const <String, String?>{}).entries) {
    if (const {'authorization', 'cf-aig-authorization'}.contains(key.toLowerCase()) &&
        (value?.trim().isNotEmpty ?? false)) {
      return 'unused';
    }
  }

  throw ProviderException('No API key for provider: ${model.provider}');
}

/// The client's default headers: the model's, the session headers, then [StreamOptions.headers], names lowercased; a
/// `null` value removes a header.
Map<String, String> _headers(Model model, OpenAIResponsesOptions options) {
  final headers = <String, String?>{};
  model.headers?.forEach((name, value) => headers[name.toLowerCase()] = value);
  final session = options.sessionId;
  if (session != null && options.cacheRetention != CacheRetention.none) {
    final format = _Compat(model).affinityFormat;
    if (format == 'openrouter') {
      headers['x-session-id'] = session;
    } else {
      if (format == 'openai') {
        headers['session-id'] = session;
      }

      headers['x-client-request-id'] = session;
    }
  }

  options.headers?.forEach((name, value) => headers[name.toLowerCase()] = value);
  return {for (final MapEntry(:key, :value) in headers.entries) key: ?value};
}

/// The error message for [error], as pi's `formatProviderError` builds it: an API error with a status shows its
/// status and body under [prefix]; a connection failure adds its cause.
String _errorMessage(Object error, String prefix) {
  if (error is! APIError) {
    return switch (error) {
      ProviderException(:final message) || StateError(:final message) || OpenAIError(:final message?) => message,
      FormatException(:final message) => message,
      _ => '$error',
    };
  }

  final message = error.message ?? '';
  final status = error.status;
  if (status == null) {
    final cause = switch (error.cause) {
      http.ClientException(:final message) => message.trim(),
      _ => '',
    };
    if (cause.isEmpty || message.contains(cause)) {
      return message;
    }

    return message.isEmpty ? cause : '$message: $cause';
  }

  final body = error.error;
  if (body is! Map || body.isEmpty) {
    return '$prefix ($status): $message';
  }

  var text = jsonEncode(body);
  if (text.length > 4000) {
    text = '${text.substring(0, 4000)}... [truncated ${text.length - 4000} chars]';
  }

  if (message.contains(text)) {
    return '$prefix ($status): $message';
  }

  return '$prefix ($status): $text';
}

/// Streams one response from the OpenAI Responses API.
///
/// Every failure, including missing credentials, ends the stream with an [ErrorEvent].
AssistantMessageEventStream streamOpenAIResponses(
  Model model,
  List<Message> messages, [
  OpenAIResponsesOptions options = const OpenAIResponsesOptions(),
]) {
  final stream = AssistantMessageEventStream();
  unawaited(_run(model, messages, options, stream));
  return stream;
}

Future<void> _run(
  Model model,
  List<Message> messages,
  OpenAIResponsesOptions options,
  AssistantMessageEventStream stream,
) async {
  final output = AssistantMessage(
    api: model.api,
    provider: model.provider,
    model: model.id,
    providerThinkingLevel: _configurationEffort(model, options),
    timestamp: DateTime.now().millisecondsSinceEpoch,
  );
  final signal = options.signal;
  OpenAI? client;
  try {
    client = OpenAI(
      apiKey: _apiKey(model, options),
      baseURL: model.baseUrl,
      defaultHeaders: _headers(model, options),
      httpClient: options.httpClient,
    );
    if (signal?.aborted ?? false) {
      throw const AbortedException();
    }

    var params = buildResponsesParams(model, messages, options);
    params = await options.onPayload?.call(params, model) ?? params;
    final (:data, :response) = await retryRequest(
      () => client!.responses.create(params, RequestOptions(signal: signal?.future, timeout: options.timeout)),
      maxRetries: options.maxRetries,
      maxDelay: options.maxRetryDelay,
      signal: signal,
    );
    await options.onResponse?.call(ProviderResponse(response.statusCode, response.headers), model);
    stream.push(StartEvent(output));
    await processResponsesStream(
      data,
      output,
      stream,
      model,
      onEvent: options.onEvent,
      serviceTier: options.serviceTier,
    );
    if (signal?.aborted ?? false) {
      throw const AbortedException();
    }

    if (output.stopReason == StopReason.pending) {
      throw const ProviderException('OpenAI Responses stream ended without a stop reason');
    }

    if (output.stopReason == StopReason.aborted || output.stopReason == StopReason.error) {
      throw ProviderException(output.errorMessage ?? 'An unknown error occurred');
    }

    stream.push(DoneEvent(output.stopReason, output));
  } catch (error) {
    output.stopReason = (signal?.aborted ?? false) ? StopReason.aborted : StopReason.error;
    if (output.stopReason == StopReason.aborted) {
      output.errorMessage = 'Request was aborted';
    } else {
      output.errorMessage = _errorMessage(error, '${model.provider == 'openai' ? 'OpenAI' : model.provider} API error');
    }

    stream.push(ErrorEvent(output.stopReason, output));
  } finally {
    client?.close();
  }
}

/// Streams one response with provider-neutral options: [SimpleStreamOptions.reasoning], clamped to the model's
/// levels, becomes the reasoning effort.
///
/// Throws at once when the request has no credentials; any later failure ends the stream with an [ErrorEvent].
AssistantMessageEventStream streamSimpleOpenAIResponses(
  Model model,
  List<Message> messages, [
  SimpleStreamOptions options = const SimpleStreamOptions(),
]) {
  _apiKey(model, options);
  final level = options.reasoning == null ? null : clampThinkingLevel(model, options.reasoning!);
  return streamOpenAIResponses(
    model,
    messages,
    OpenAIResponsesOptions(
      signal: options.signal,
      apiKey: options.apiKey,
      headers: options.headers,
      httpClient: options.httpClient,
      timeout: options.timeout,
      maxRetries: options.maxRetries,
      maxRetryDelay: options.maxRetryDelay,
      temperature: options.temperature,
      maxTokens: clampMaxTokensToContext(model, messages, options.maxTokens ?? model.maxTokens),
      cacheRetention: options.cacheRetention,
      sessionId: options.sessionId,
      metadata: options.metadata,
      onPayload: options.onPayload,
      onResponse: options.onResponse,
      onEvent: options.onEvent,
      disableContinuation: options.disableContinuation,
      reasoningEffort: level == ThinkingLevel.off ? null : level?.name,
      toolChoice: options.toolChoice,
    ),
  );
}
