/// The Anthropic Messages provider, ported from pi-ai's `api/anthropic-messages.ts`.
///
/// Not ported yet: strict tools, message threads, mid-conversation effort, system messages and tool changes, and
/// server-side fallback models. Not planned: pi's OAuth mode, which presents the client as Claude Code, and the
/// GitHub Copilot and OpenRouter specifics.
library;

import 'dart:async';
import 'dart:convert';
import 'dart:math';

import 'package:anthropic_sdk_lite_dart/anthropic_sdk_lite_dart.dart';
import 'package:http/http.dart' as http;

import '../constrained.dart';
import '../event_stream.dart';
import '../json.dart';
import '../model.dart';
import '../options.dart';
import '../retry.dart';
import '../session.dart';
import '../transcript.dart';
import '../transform.dart';
import '../types.dart';

/// Options of the Anthropic provider, ported from pi's `AnthropicOptions`.
class AnthropicOptions extends StreamOptions {
  const AnthropicOptions({
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
    this.thinkingEnabled,
    this.thinkingBudgetTokens,
    this.effort,
    this.thinkingDisplay,
    this.interleavedThinking,
    this.toolChoice,
    this.client,
  });

  /// The provider-neutral [options] as Anthropic options, with none of Anthropic's own set.
  factory AnthropicOptions.from(StreamOptions options) {
    return AnthropicOptions(
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

  /// Extended thinking: adaptive for models with `forceAdaptiveThinking`, budget-based for older ones. Omitted from
  /// the request when `null`.
  final bool? thinkingEnabled;

  /// The thinking budget of budget-based models; 1024 when not set.
  final int? thinkingBudgetTokens;

  /// `low`, `medium`, `high`, `xhigh` or `max`, for adaptive thinking.
  final String? effort;

  /// `summarized` (the default here) or `omitted`.
  final String? thinkingDisplay;

  /// Whether budget-based models get the interleaved thinking beta; true by default.
  final bool? interleavedThinking;

  /// `auto`, `any`, `none`, or `{'type': 'tool', 'name': ...}` to force a tool.
  final Object? toolChoice;

  /// A client to send with instead of one built from [apiKey] and [headers]; for example one with a workload
  /// identity federation `config`.
  final Anthropic? client;
}

/// Thrown by the provider for a failure it detects itself; the message goes to the response's `errorMessage`.
class ProviderException implements Exception {
  const ProviderException(this.message);

  final String message;

  @override
  String toString() => message;
}

const _fineGrainedToolStreaming = 'fine-grained-tool-streaming-2025-05-14';
const _interleavedThinking = 'interleaved-thinking-2025-05-14';
const _midConversationOutputConfig = 'mid-conversation-output-config-2026-07-01';
const _thinkingBindingControls = 'thinking-binding-controls-2026-08-01';
const _inlineTools = 'inline-tools-2026-09-15';
const _serverSideFallback = 'server-side-fallback-2026-07-01';
const _messageThreads = 'message-threads-2026-08-12';

/// A deferred tool declared whenever tool changes are native. Anthropic adds hidden prompt scaffolding for tool changes
/// mid-conversation; declaring this from the first request keeps it in the cached prefix, so the first tool change does
/// not invalidate the cache. It is never activated and the model cannot see it.
const _deferredPlaceholder = {
  'name': '__pi_deferred_placeholder__',
  'description': 'Reserved placeholder. Never available. Never call this.',
  'input_schema': {'type': 'object', 'properties': <String, Object?>{}, 'required': <Object?>[]},
  'defer_loading': true,
};

/// Anthropic's effort levels.
const _efforts = {'low', 'medium', 'high', 'xhigh', 'max'};

/// The `compat` settings this provider reads, with their defaults.
class _Compat {
  _Compat(Model model)
    : eagerToolInput = model.compat['supportsEagerToolInputStreaming'] as bool? ?? true,
      longCache = model.compat['supportsLongCacheRetention'] as bool? ?? true,
      affinity = model.compat['sendSessionAffinityHeaders'] as bool? ?? _openRouter(model),
      affinityFormat = model.compat['sessionAffinityFormat'] as String? ?? (_openRouter(model) ? 'openrouter' : null),
      cachedTools = model.compat['supportsCacheControlOnTools'] as bool? ?? true,
      temperature = model.compat['supportsTemperature'] as bool? ?? true,
      emptySignature = model.compat['allowEmptySignature'] as bool? ?? false,
      adaptive = model.compat['forceAdaptiveThinking'] as bool? ?? false,
      midEffort = model.compat['supportsMidConvoEffort'] as bool? ?? false,
      strictTools = model.compat['supportsStrictTools'] as bool? ?? false,
      midSystem = model.compat['supportsMidConvoSystemMessages'] as bool? ?? false,
      midTools = model.compat['supportsMidConvoToolChanges'] as bool? ?? false,
      fallbacks = (model.compat['allowedFallbackModels'] as List<Object?>? ?? const []).cast<Json>(),
      threads = model.compat['supportsMessageThreads'] as bool? ?? model.baseUrl.contains('api.anthropic.com');

  final bool eagerToolInput;
  final bool longCache;
  final bool affinity;
  final String? affinityFormat;
  final bool cachedTools;
  final bool temperature;
  final bool emptySignature;
  final bool adaptive;

  /// Models with mid-conversation effort reject `temperature`.
  final bool midEffort;
  final bool strictTools;

  /// Later system messages are sent in place, as native system messages.
  final bool midSystem;

  /// Tool changes are sent in place, with native system messages.
  final bool midTools;

  /// The models the API may fall back to: `provider`, `model`, and the `cost` the caller is billed.
  final List<Json> fallbacks;

  /// Message threads: only on Anthropic's own endpoint by default, until proxies generally forward `thread`.
  final bool threads;

  static bool _openRouter(Model model) => model.provider == 'openrouter' || model.baseUrl.contains('openrouter.ai');
}

final _surrogates = RegExp(r'[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]');

/// [text] without unpaired UTF-16 surrogates, which many APIs reject.
String sanitizeSurrogates(String text) => text.replaceAll(_surrogates, '');

/// A tool call id as Anthropic requires it: `[a-zA-Z0-9_-]`, at most 64 characters.
String normalizeToolCallId(String id) {
  final safe = id.replaceAll(RegExp('[^a-zA-Z0-9_-]'), '_');
  return safe.length > 64 ? safe.substring(0, 64) : safe;
}

Json? _cacheControl(Model model, CacheRetention? retention) {
  final value = retention ?? CacheRetention.short;
  if (value == CacheRetention.none) {
    return null;
  }

  return {'type': 'ephemeral', if (value == CacheRetention.long && _Compat(model).longCache) 'ttl': '1h'};
}

Json _image(ImageContent image) {
  return {
    'type': 'image',
    'source': {'type': 'base64', 'media_type': image.mimeType, 'data': image.data},
  };
}

/// Tool result content: one string when it is all text, otherwise blocks, with a placeholder text if there is none.
Object _resultContent(List<Content> content) {
  if (!content.any((c) => c is ImageContent)) {
    return sanitizeSurrogates(content.whereType<TextContent>().map((c) => c.text).join('\n'));
  }

  final blocks = [
    for (final c in content)
      if (c is TextContent) {'type': 'text', 'text': sanitizeSurrogates(c.text)} else _image(c as ImageContent),
  ];
  if (!content.any((c) => c is TextContent)) {
    blocks.insert(0, {'type': 'text', 'text': '(see attached image)'});
  }

  return blocks;
}

Json _toolResult(ToolResultMessage message, List<Content> content) {
  return {
    'type': 'tool_result',
    'tool_use_id': message.id,
    'content': _resultContent(content),
    'is_error': message.isError,
  };
}

/// The conversation (without the leading system message) as Anthropic messages, with [cache] on the last block of
/// the last user message. With [efforts], the effort of each reply of [effortProvider] that recorded one goes into it,
/// by the reply's index in the result.
List<Json> convertMessages(
  List<Message> messages,
  Json? cache, {
  bool emptySignature = false,
  String? effortProvider,
  Map<int, String>? efforts,
  List<Json> Function(List<Tool> tools)? toolDefinitions,
}) {
  final params = <Json>[];

  // Later system messages (only present when the model takes them natively) are held back and sent directly before
  // the next reply that is sent, or at the end: Anthropic rejects one between a tool use and its result, and a reply
  // that converts to nothing is dropped, so holding past it keeps an update from landing before a user message.
  final pending = <Json>[];
  void flush() {
    params.addAll(pending);
    pending.clear();
  }

  for (var i = 0; i < messages.length; i++) {
    final message = messages[i];
    switch (message) {
      case SystemMessage():
        final text = renderSystemMessageUpdate(message);
        final blocks = <Json>[
          if (text.isNotEmpty) {'type': 'text', 'text': sanitizeSurrogates(text)},
          ...?toolDefinitions == null ? null : _toolChanges(message, toolDefinitions),
        ];
        if (blocks.isNotEmpty) {
          pending.add({'role': 'system', 'content': blocks});
        }
      case UserMessage():
        final blocks = <Json>[];
        for (final c in message.content) {
          if (c is ImageContent) {
            blocks.add(_image(c));
          } else if ((c as TextContent).text.trim().isNotEmpty) {
            blocks.add({'type': 'text', 'text': sanitizeSurrogates(c.text)});
          }
        }

        if (blocks.isEmpty) {
          continue;
        }

        params.add({'role': 'user', 'content': blocks});
      case AssistantMessage():
        final blocks = <Json>[];
        for (final block in message.content) {
          switch (block) {
            case TextContent():
              if (block.text.trim().isNotEmpty) {
                blocks.add({'type': 'text', 'text': sanitizeSurrogates(block.text)});
              }
            case ThinkingContent():
              if (block.redacted) {
                blocks.add({'type': 'redacted_thinking', 'data': block.signature});
                continue;
              }

              final signed = block.signature?.trim().isNotEmpty ?? false;
              if (block.thinking.trim().isEmpty && !signed) {
                continue;
              }

              // Thinking without a signature (from an aborted stream) is sent as text, unless the provider
              // accepts empty signatures.
              if (signed) {
                blocks.add({
                  'type': 'thinking',
                  'thinking': sanitizeSurrogates(block.thinking),
                  'signature': block.signature,
                });
              } else if (emptySignature) {
                blocks.add({'type': 'thinking', 'thinking': sanitizeSurrogates(block.thinking), 'signature': ''});
              } else {
                blocks.add({'type': 'text', 'text': sanitizeSurrogates(block.thinking)});
              }
            case ToolCall():
              blocks.add({'type': 'tool_use', 'id': block.id, 'name': block.name, 'input': block.arguments});
            case ImageContent():
              continue;
          }
        }

        if (blocks.isEmpty) {
          continue;
        }

        flush();
        final level = message.providerThinkingLevel;
        if (efforts != null && _replayable(message, effortProvider) && _efforts.contains(level)) {
          efforts[params.length] = level!;
        }

        params.add({'role': 'assistant', 'content': blocks});
      case ToolResultMessage():

        // Consecutive results go in one user message. Anthropic rejects images in error results, so their images
        // follow all the results.
        final results = <Json>[];
        final images = <Json>[];
        for (; i < messages.length && messages[i] is ToolResultMessage; i++) {
          final result = messages[i] as ToolResultMessage;
          final split = splitErrorImages(result);
          results.add(_toolResult(result, split?.content ?? result.content));
          if (split != null) {
            images.add({'type': 'text', 'text': split.caption});
            images.addAll(split.images.map(_image));
          }
        }

        i--;
        params.add({
          'role': 'user',
          'content': [...results, ...images],
        });
    }
  }

  flush();
  if (cache == null || params.isEmpty || !const {'user', 'system'}.contains(params.last['role'])) {
    return params;
  }

  final last = (params.last['content'] as List<Json>).last;
  if (const {'text', 'image', 'tool_result', 'tool_addition', 'tool_removal'}.contains(last['type'])) {
    last['cache_control'] = cache;
  }

  return params;
}

// A system message's tool changes as native blocks: removals, then additions defined by value. A tool added again
// under its name replaces the old definition, so it needs no removal.
List<Json> _toolChanges(SystemMessage message, List<Json> Function(List<Tool> tools) toolDefinitions) {
  final added = message.added ?? const <Tool>[];
  final redefined = {for (final t in added) t.name};
  return [
    for (final name in message.removed ?? const <String>[])
      if (!redefined.contains(name))
        {
          'type': 'tool_removal',
          'tool': {'type': 'tool_reference', 'name': name},
        },
    for (final definition in toolDefinitions(added))
      {
        'type': 'tool_addition',
        'tool': {'type': 'tool_definition', 'definition': definition},
      },
  ];
}

// Whether [message] is a reply of [provider] on this API, whose recorded effort can be replayed.
bool _replayable(AssistantMessage message, String? provider) {
  return provider != null && message.api == 'anthropic-messages' && message.provider == provider;
}

/// [messages] with an effort marker (an empty system message with `output_config`) before each reply in [efforts], and
/// one with [active] at the end.
List<Json> _withEffortMarkers(List<Json> messages, Map<int, String> efforts, String active) {
  Json marker(String effort) {
    return {
      'role': 'system',
      'content': <Object?>[],
      'output_config': {'effort': effort},
    };
  }

  return [
    for (final (i, message) in messages.indexed) ...[if (efforts[i] case final String effort) marker(effort), message],
    marker(active),
  ];
}

/// Whether thinking is disabled for this request: thinking turned off on a model that allows it (`off` not null).
bool _thinkingDisabled(Model model, AnthropicOptions options) {
  if (!model.reasoning || options.thinkingEnabled != false) {
    return false;
  }

  final levels = model.levels ?? const {};
  return !(levels.containsKey('off') && levels['off'] == null);
}

/// Whether the request sets its effort per reply with markers: on models with mid-conversation effort, unless
/// thinking is disabled (the level cannot change between messages then).
bool _effortMarkers(Model model, AnthropicOptions options) {
  return _Compat(model).midEffort && !_thinkingDisabled(model, options);
}

/// Tools as Anthropic declares them, with [cache] on the last one.
// Keywords Anthropic's strict tool use rejects with a 400 for the whole request.
// https://platform.claude.com/docs/en/build-with-claude/structured-outputs#json-schema-limitations
const _strictUnsupported = {
  'minimum',
  'maximum',
  'exclusiveMinimum',
  'exclusiveMaximum',
  'multipleOf',
  'maxItems',
  'uniqueItems',
  'minContains',
  'maxContains',
  'minProperties',
  'maxProperties',
};
const _strictFormats = {'date-time', 'time', 'date', 'duration', 'email', 'hostname', 'uri', 'ipv4', 'ipv6', 'uuid'};

bool _strictRejects(String key, Object? value) {
  if (_strictUnsupported.contains(key)) {
    return true;
  }

  if (key == 'minItems') {
    return value != 0 && value != 1;
  }

  return key == 'format' && !_strictFormats.contains(value);
}

/// Whether [tool] is sent strict: when it asks for strict sampling, the model supports it and its schema converts.
/// [disablePreferred] sends `prefer` tools non-strict, after Anthropic rejected them for exceeding its limits.
bool? _strict(Tool tool, bool supported, bool disablePreferred) {
  if (disablePreferred && prefersStrictSampling(tool)) {
    return null;
  }

  return resolveStrictSampling(tool, supported, _strictRejects);
}

/// The request's tools. A strict tool sends its strict schema; the others only `properties`, `required` and the
/// keywords property references and the model need. [strict] says whether the model supports strict tools.
List<Json> convertTools(
  List<Tool> tools, {
  bool eager = true,
  Json? cache,
  bool strict = false,
  bool disablePreferredStrict = false,
}) {
  return [
    for (final (i, tool) in tools.indexed)
      _tool(tool, eager, i == tools.length - 1 ? cache : null, strict, disablePreferredStrict),
  ];
}

Json _tool(Tool tool, bool eager, Json? cache, bool supported, bool disablePreferred) {
  final strict = _strict(tool, supported, disablePreferred);
  final parameters = strictToolParameters(tool, strict);

  // Anthropic rejects root anyOf, oneOf and allOf, so only these root keywords are kept.
  final schema = {
    'type': 'object',
    'properties': parameters['properties'] ?? <String, Object?>{},
    'required': parameters['required'] ?? <Object?>[],
    if (parameters.containsKey('additionalProperties')) 'additionalProperties': parameters['additionalProperties'],
    if (parameters.containsKey(r'$defs')) r'$defs': parameters[r'$defs'],
    if (parameters.containsKey('definitions')) 'definitions': parameters['definitions'],
  };
  return {
    'name': tool.name,
    'description': tool.description,
    if (eager) 'eager_input_streaming': true,
    if (strict == true) 'strict': true,
    'input_schema': strict == true ? {...parameters, ...schema} : schema,
    'cache_control': ?cache,
  };
}

/// Whether tool changes are sent natively: the model takes both system messages and tool changes mid-conversation, and
/// the transcript starts with tools (Anthropic rejects a tool list whose only tool is the deferred placeholder).
bool _nativeToolChanges(Model model, List<Message> transcript) {
  final compat = _Compat(model);
  if (!compat.midSystem || !compat.midTools) {
    return false;
  }

  return initialSystemMessage(transcript)?.added?.isNotEmpty ?? false;
}

/// The tools a request of [transcript] declares: every addition when tool changes are native, else the current ones.
List<Tool> _sentTools(Model model, List<Message> transcript) {
  if (!_nativeToolChanges(model, transcript)) {
    return currentTools(transcript);
  }

  return [for (final m in transcript.whereType<SystemMessage>()) ...?m.added];
}

/// Whether a request of [tools] sends a `prefer` tool strict, so a rejection for Claude's strict tool limits can be
/// resent without.
bool _hasPreferredStrict(Model model, List<Tool> tools, bool disablePreferred) {
  final supported = _Compat(model).strictTools;
  return tools.any((tool) => prefersStrictSampling(tool) && _strict(tool, supported, disablePreferred) == true);
}

/// Claude's rejections of a request whose strict tools exceed its limits. Observed limits: 20 strict tools, and 16
/// optional or union-typed parameters across all strict tools.
const _strictLimitErrors = [
  'The compiled grammar is too large',
  'Too many strict tools (',
  'Schemas contains too many parameters with union types (',
];

/// Sessions whose strict tools Claude rejected for exceeding its limits: their later requests send `prefer` tools
/// non-strict right away, since the tool set rarely changes and switching back would miss the prompt cache.
final _overStrictLimits = SessionSet(10000);

/// The beta features of a request. An `anthropic-beta` header in the model's or the caller's headers replaces them
/// (a `null` one clears them).
List<String> _betas(
  Model model,
  List<Tool> tools,
  AnthropicOptions options, {
  bool native = false,
  bool threads = false,
}) {
  var configured = false;
  String? value;
  for (final headers in [model.headers, options.headers]) {
    headers?.forEach((name, v) {
      if (name.toLowerCase() == 'anthropic-beta') {
        configured = true;
        value = v;
      }
    });
  }

  if (configured) {
    return {...?value?.split(',').map((f) => f.trim()).where((f) => f.isNotEmpty)}.toList();
  }

  final compat = _Compat(model);
  return {
    if (tools.isNotEmpty && !compat.eagerToolInput) _fineGrainedToolStreaming,
    if (model.reasoning && options.thinkingEnabled == true && (options.interleavedThinking ?? true) && !compat.adaptive)
      _interleavedThinking,
    if (compat.midEffort) ...[_midConversationOutputConfig, _thinkingBindingControls],
    if (compat.fallbacks.isNotEmpty) _serverSideFallback,
    if (native) _inlineTools,
    if (threads) _messageThreads,
  }.toList();
}

/// The request body for [messages], a transcript whose only system message leads.
///
/// [threads] says the session uses message threads, so the threads beta is sent.
Json buildParams(
  Model model,
  List<Message> messages,
  AnthropicOptions options, {
  bool disablePreferredStrict = false,
  bool threads = false,
}) {
  final cache = _cacheControl(model, options.cacheRetention);
  final compat = _Compat(model);
  final system = initialSystemMessage(messages);
  final transformed = transformMessages(messages, model, normalizeToolCallId);
  final tools = currentTools(messages);
  final native = _nativeToolChanges(model, messages);
  final betas = _betas(model, tools, options, native: native, threads: threads);
  final toolCache = compat.cachedTools ? cache : null;
  List<Json> definitions(List<Tool> list, [Json? cache]) {
    return convertTools(
      list,
      eager: compat.eagerToolInput,
      cache: cache,
      strict: compat.strictTools,
      disablePreferredStrict: disablePreferredStrict,
    );
  }

  final text = system == null ? '' : systemText(system);

  final markers = _effortMarkers(model, options);
  final efforts = <int, String>{};
  final converted = convertMessages(
    system == null ? transformed : transformed.sublist(1),
    cache,
    emptySignature: compat.emptySignature,
    effortProvider: markers ? model.provider : null,
    efforts: efforts,
    toolDefinitions: native ? definitions : null,
  );
  final params = <String, Object?>{
    'model': model.id,
    'messages': markers ? _withEffortMarkers(converted, efforts, options.effort ?? 'high') : converted,
    'max_tokens': options.maxTokens ?? model.maxTokens,
    'stream': true,
    if (betas.isNotEmpty) 'betas': betas,
    if (text.isNotEmpty)
      'system': [
        {'type': 'text', 'text': sanitizeSurrogates(text), 'cache_control': ?cache},
      ],
  };

  // Temperature is incompatible with extended thinking and unsupported from Claude Opus 4.7 on.
  final temperature = options.temperature;
  if (temperature != null && options.thinkingEnabled != true && !compat.midEffort && compat.temperature) {
    params['temperature'] = temperature;
  }

  if (native) {
    // The initial tools (with the cache breakpoint) and the placeholder. The list never changes: later tools are
    // defined by value in `tool_addition` blocks and withdrawn with `tool_removal`, so the cached prefix survives every
    // tool change.
    params['tools'] = [...definitions(initialSystemMessage(messages)!.added!, toolCache), _deferredPlaceholder];
  } else if (tools.isNotEmpty) {
    params['tools'] = definitions(tools, toolCache);
  }

  // A `thinkingLevelMap` with `off: null` marks thinking as impossible to turn off.
  if (_thinkingDisabled(model, options)) {
    // No effort is sent, so the model default applies. Anthropic rejects `block_binding` with disabled thinking.
    params['thinking'] = {'type': 'disabled'};
  } else if (markers) {
    // Adaptive thinking, so a prefix mismatch drops the block instead of failing every later request. The markers set
    // each reply's effort; the top-level one is part of the cached prompt, so it stays fixed, and "low" because
    // organizations can cap effort and reject requests with any level above the cap.
    params['thinking'] = {
      'type': 'adaptive',
      'display': options.thinkingDisplay ?? 'summarized',
      'block_binding': {'prefix_mismatch_behavior': 'drop_block'},
    };
    params['output_config'] = {'effort': 'low'};
  } else if (model.reasoning && options.thinkingEnabled == true) {
    // "summarized" keeps Opus 4.7 and later, whose API default is "omitted", like older Claude 4 models.
    final display = options.thinkingDisplay ?? 'summarized';
    if (compat.adaptive) {
      params['thinking'] = {'type': 'adaptive', 'display': display};
      if (options.effort != null) {
        params['output_config'] = {'effort': options.effort};
      }
    } else {
      params['thinking'] = {
        'type': 'enabled',
        'budget_tokens': options.thinkingBudgetTokens ?? 1024,
        'display': display,
      };
    }
  }

  final user = options.metadata?['user_id'];
  if (user is String) {
    params['metadata'] = {'user_id': user};
  }

  final choice = options.toolChoice;
  if (choice != null) {
    params['tool_choice'] = choice is String ? {'type': choice} : choice;
  }

  if (compat.fallbacks.isNotEmpty) {
    params['fallbacks'] = [
      for (final f in compat.fallbacks) {'model': f['model']},
    ];
  }

  return params;
}

bool _hasHeader(Map<String, String?>? headers, String name) {
  if (headers == null) {
    return false;
  }

  return headers.entries.any((h) => h.key.toLowerCase() == name && (h.value?.trim().isNotEmpty ?? false));
}

/// Throws when the request would carry no credentials: no API key and no auth header.
void _assertAuth(Model model, AnthropicOptions options) {
  if (options.client != null) {
    return;
  }

  if (options.apiKey?.isNotEmpty ?? false) {
    return;
  }

  final headers = options.headers;
  if (_hasHeader(headers, 'authorization') ||
      _hasHeader(headers, 'x-api-key') ||
      _hasHeader(headers, 'cf-aig-authorization')) {
    return;
  }

  throw ProviderException('No API key for provider: ${model.provider}');
}

Anthropic _createClient(Model model, AnthropicOptions options) {
  final compat = _Compat(model);
  final headers = <String, String?>{'accept': 'application/json', 'anthropic-dangerous-direct-browser-access': 'true'};
  final session = options.sessionId;
  if (session != null && options.cacheRetention != CacheRetention.none && compat.affinity) {
    headers[compat.affinityFormat == 'openrouter' ? 'x-session-id' : 'x-session-affinity'] = session;
  }

  for (final extra in [model.headers, options.headers]) {
    extra?.forEach((name, value) => headers[name.toLowerCase()] = value);
  }

  return Anthropic(
    apiKey: options.apiKey,
    baseURL: model.baseUrl,
    defaultHeaders: {for (final MapEntry(:key, :value) in headers.entries) key: ?value},
    httpClient: options.httpClient,
  );
}

const _messageEvents = {
  'message_start',
  'message_delta',
  'message_stop',
  'content_block_start',
  'content_block_delta',
  'content_block_stop',
};

/// The server-sent events of [body]: `(event, data)` pairs, split on CR, LF or CRLF.
Stream<({String? event, String data, List<String> raw})> decodeSse(Stream<List<int>> body) async* {
  String? event;
  var data = <String>[];
  var raw = <String>[];
  await for (final line in body.transform(utf8.decoder).transform(const LineSplitter())) {
    if (line.isEmpty) {
      if (event != null || data.isNotEmpty) {
        yield (event: event, data: data.join('\n'), raw: raw);
      }

      event = null;
      data = [];
      raw = [];
      continue;
    }

    raw.add(line);
    if (line.startsWith(':')) {
      continue;
    }

    final colon = line.indexOf(':');
    final name = colon == -1 ? line : line.substring(0, colon);
    var value = colon == -1 ? '' : line.substring(colon + 1);
    if (value.startsWith(' ')) {
      value = value.substring(1);
    }

    if (name == 'event') {
      event = value;
    } else if (name == 'data') {
      data.add(value);
    }
  }

  if (event != null || data.isNotEmpty) {
    yield (event: event, data: data.join('\n'), raw: raw);
  }
}

/// The Messages API events of [body]. An `error` event throws, and so does a stream that ends after
/// `message_start` without `message_stop`.
Stream<Json> anthropicEvents(Stream<List<int>> body) async* {
  var started = false;
  var stopped = false;
  await for (final sse in decodeSse(body)) {
    if (sse.event == 'error') {
      throw ProviderException(sse.data);
    }

    if (!_messageEvents.contains(sse.event)) {
      continue;
    }

    final Json event;
    try {
      event = decodeWithRepair(sse.data) as Json;
    } on FormatException catch (e) {
      throw ProviderException(
        'Could not parse Anthropic SSE event ${sse.event}: ${e.message}; data=${sse.data}; raw=${sse.raw.join(r'\n')}',
      );
    }

    switch (event['type']) {
      case 'message_start':
        started = true;
      case 'message_stop':
        stopped = true;
    }

    yield event;
  }

  if (started && !stopped) {
    throw const ProviderException('Anthropic stream ended before message_stop');
  }
}

/// Whether Anthropic rejected a request because its strict tools exceed Claude's limits.
bool _strictLimitRejection(APIError error) {
  if (error.status != 400 || error.type != 'invalid_request_error') {
    return false;
  }

  return _strictLimitErrors.any(error.message.contains);
}

/// pi's stop reason for Anthropic's, with an error message for refusals.
({StopReason reason, String? error}) mapStopReason(String reason, Json? details) {
  return switch (reason) {
    'end_turn' || 'pause_turn' || 'stop_sequence' => (reason: StopReason.stop, error: null),
    'max_tokens' => (reason: StopReason.length, error: null),
    'tool_use' => (reason: StopReason.toolUse, error: null),
    'refusal' => (
      reason: StopReason.error,
      error: (details?['explanation'] as String?) ?? 'The model refused to complete the request',
    ),
    'sensitive' => (reason: StopReason.error, error: 'Provider stopped with: sensitive'),
    _ => throw ProviderException('Unhandled stop reason: $reason'),
  };
}

/// Streams one response of [model] to [messages], a transcript whose system messages carry the prompt and tools.
///
/// Failures never throw: they end the stream with an [ErrorEvent].
AssistantMessageEventStream streamAnthropic(
  Model model,
  List<Message> messages, [
  AnthropicOptions options = const AnthropicOptions(),
]) {
  final stream = AssistantMessageEventStream();
  // Later system messages stay in place for models that take them natively; otherwise they fold into the leading one.
  final transcript = _Compat(model).midSystem ? messages : collapseSystemMessages(messages);
  unawaited(_run(model, transcript, options, stream));
  return stream;
}

// Message threads: the server keeps a thread's conversation, so a request can continue from the previous reply and send
// only the new messages. The full request is still built; it is compared with the session's previous one, and only the
// new part is sent when nothing earlier changed.

/// Where a session's next request can continue: the length and hash of the last full request plus its reply, and the
/// reply's id.
typedef _Thread = ({int prefixCount, String prefixHash, String replyId});

/// Per session: where the next request can continue, or [_threadsOff] after the endpoint rejected threads. At most
/// 10,000 sessions; a dropped one just starts a new thread next time.
final _threads = SessionMap<Object>(10000);
const _threadsOff = #off;

/// Rejections of a `continue` that a new thread fixes: expired, raced, or a fixed setting changed.
const _newThreadErrors = {
  'thread_not_found',
  'thread_already_continued',
  'thread_continue_in_flight',
  'thread_fingerprint_mismatch',
};

/// Hash of what a `continue` builds on: the settings that must stay the same within a thread (a `continue` omits
/// `system` and `tools`, so only this notices a change to them) and the messages the server already holds, normalized
/// so messages that mean the same hash the same.
String _threadHash(Model model, Json params, List<Object?> messages) {
  final settings = jsonEncode([
    model.provider,
    model.baseUrl,
    params['model'],
    params['system'],
    params['tools'],
    params['thinking'],
    params['output_config'],
    params['tool_choice'],
    params['betas'],
    params['fallbacks'],
  ]);
  return _shortHash(settings + jsonEncode(_withoutCacheMarkers(messages)));
}

// Cache markers move to the newest message from request to request, and a string content means one text block.
Object? _withoutCacheMarkers(Object? value) {
  return switch (value) {
    Map() => {
      for (final MapEntry(:key, :value) in value.entries)
        if (key != 'cache_control') key: _normalized(key, value),
    },
    List() => [for (final v in value) _withoutCacheMarkers(v)],
    _ => value,
  };
}

Object? _normalized(Object? key, Object? value) {
  if (key == 'content' && value is String) {
    return [
      {'type': 'text', 'text': value},
    ];
  }

  return _withoutCacheMarkers(value);
}

/// A fast deterministic hash (pi's `shortHash`): two 32-bit lanes over the UTF-16 units.
String _shortHash(String text) {
  int mul(int a, int b) => (a * b) & 0xffffffff;
  var h1 = 0xdeadbeef;
  var h2 = 0x41c6ce57;
  for (final c in text.codeUnits) {
    h1 = mul(h1 ^ c, 2654435761);
    h2 = mul(h2 ^ c, 1597334677);
  }

  h1 = mul(h1 ^ (h1 >> 16), 2246822507) ^ mul(h2 ^ (h2 >> 13), 3266489909);
  h2 = mul(h2 ^ (h2 >> 16), 2246822507) ^ mul(h1 ^ (h1 >> 13), 3266489909);
  return h2.toRadixString(36) + h1.toRadixString(36);
}

/// A request in a session's thread.
typedef _ThreadRequest = ({String session, Json body});

_ThreadRequest _createThread(String session, Json params) {
  return (
    session: session,
    body: {
      ...params,
      'thread': {'type': 'create'},
    },
  );
}

/// The thread request for [params]: a `continue` with only the new messages when the session's last request plus its
/// reply is an unchanged prefix, else a new thread; null when the session's threads are off.
_ThreadRequest? _prepareThread(Model model, String session, Json params) {
  final state = _threads[session];
  if (state == _threadsOff) {
    return null;
  }

  final messages = params['messages'] as List<Object?>;
  if (state is _Thread &&
      messages.length > state.prefixCount &&
      _threadHash(model, params, messages.sublist(0, state.prefixCount)) == state.prefixHash) {
    // Each reply can be continued once; removing it also keeps a concurrent request from trying.
    _threads.remove(session);
    return (
      session: session,
      body: {
        for (final MapEntry(:key, :value) in params.entries)
          if (key != 'system' && key != 'tools') key: value,
        'messages': messages.sublist(state.prefixCount),
        'thread': {'type': 'continue', 'previous_message_id': state.replyId},
      },
    );
  }

  return _createThread(session, params);
}

/// The `error_code` Anthropic puts in `error.details` of thread rejections.
String? _threadErrorCode(APIError error) {
  final body = error.error;
  if (body is! Map) {
    return null;
  }

  final details = (body['error'] as Map?)?['details'];
  return details is Map ? details['error_code'] as String? : null;
}

/// Sends [thread]. A rejection about the thread (never a started stream) is resent once in full: as a new thread when
/// the old one cannot be continued, or without `thread` when the endpoint cannot use threads, which also turns them
/// off for the session. Other errors are passed on. Returns the response and the thread request it answers, if any.
Future<(http.StreamedResponse, _ThreadRequest?)> _sendThread(
  _ThreadRequest thread,
  Json params,
  Future<http.StreamedResponse> Function(Json body) send,
  AssistantMessage output,
  AbortSignal? signal,
) async {
  try {
    return (await send(thread.body), thread);
  } on APIError catch (error) {
    if (signal?.aborted ?? false) {
      rethrow;
    }

    final code = _threadErrorCode(error);
    final newThread = _newThreadErrors.contains(code);

    // The endpoint cannot use threads: it cannot thread this request, or it does not know the `thread` field and
    // rejects it without a code ("thread: Extra inputs are not permitted"), as without the threads beta.
    final unsupported =
        code == 'thread_unsupported_request' ||
        (code == null && error.status == 400 && error.message.contains('thread'));

    // Any other error (prompt too long, unknown model, strict tool limits...) is not about the thread.
    if (!newThread && !unsupported) {
      rethrow;
    }

    // Recorded before the resend, so a failed resend keeps it.
    output.diagnostics = [
      ...?output.diagnostics,
      Diagnostic(
        type: 'anthropic_thread_fallback',
        timestamp: DateTime.now().millisecondsSinceEpoch,
        error: {'message': error.message},
        details: {'reason': code ?? '${error.status}'},
      ),
    ];
    if (newThread) {
      final created = _createThread(thread.session, params);
      return (await send(created.body), created);
    }

    // With the threads beta, a request without `thread` is a normal one.
    final response = await send(params);
    _threads[thread.session] = _threadsOff;
    return (response, null);
  }
}

/// Records where the session's next request can continue: after [output], the clean reply to [thread], whose full
/// request was [params].
void _recordThread(Model model, _ThreadRequest thread, Json params, AssistantMessage output) {
  // The server does not store a fallback model's reply (`responseModel` is set only then), so it cannot be continued.
  final id = output.responseId;
  if (id == null || output.responseModel != null) {
    return;
  }

  // The reply converted as the next request will convert it, without the placeholder results of trailing tool calls.
  final reply = convertMessages(
    transformMessages([output], model, normalizeToolCallId),
    null,
    emptySignature: _Compat(model).emptySignature,
  ).where((m) => m['role'] == 'assistant');
  final prefix = [...params['messages'] as List<Object?>, ...reply];
  _threads[thread.session] = (prefixCount: prefix.length, prefixHash: _threadHash(model, params, prefix), replyId: id);
}

Future<void> _run(
  Model model,
  List<Message> transcript,
  AnthropicOptions options,
  AssistantMessageEventStream stream,
) async {
  final output = AssistantMessage(
    api: model.api,
    provider: model.provider,
    model: model.id,
    timestamp: DateTime.now().millisecondsSinceEpoch,
  );
  final signal = options.signal;
  Anthropic? owned;
  try {
    _assertAuth(model, options);
    if (options.client == null) {
      owned = _createClient(model, options);
    }

    if (signal?.aborted ?? false) {
      throw const AbortedException();
    }

    // The effort this reply is asked for, recorded so later requests can replay it as its marker.
    if (_effortMarkers(model, options)) {
      output.providerThinkingLevel = options.effort ?? 'high';
    }

    // The session whose requests may use threads: none without prompt caching or on endpoints without threads. A
    // session whose threads are off still sends the beta, which the resend that turned them off was accepted with.
    final session = options.cacheRetention == CacheRetention.none ? null : options.sessionId;
    final threadSession = _Compat(model).threads ? session : null;
    Future<http.StreamedResponse> send(Json body) {
      return retryRequest(
        () => (options.client ?? owned!).beta.messages.create(
          body,
          RequestOptions(signal: signal?.future, timeout: options.timeout),
        ),
        maxRetries: options.maxRetries,
        maxDelay: options.maxRetryDelay,
        signal: signal,
      );
    }

    var disablePreferred = _overStrictLimits.contains(options.sessionId);
    http.StreamedResponse response;
    Json params;

    // The thread request that produced the response; null when no `thread` was sent.
    _ThreadRequest? thread;
    for (;;) {
      if (signal?.aborted ?? false) {
        throw const AbortedException();
      }

      params = buildParams(
        model,
        transcript,
        options,
        disablePreferredStrict: disablePreferred,
        threads: threadSession != null,
      );
      final next = await options.onPayload?.call(params, model);
      if (next != null) {
        params = {...next, 'stream': true};
      }

      _ThreadRequest? prepared;
      if (threadSession != null && options.disableContinuation != true) {
        prepared = _prepareThread(model, threadSession, params);
      }

      try {
        if (prepared == null) {
          response = await send(params);
          thread = null;
        } else {
          (response, thread) = await _sendThread(prepared, params, send, output, signal);
        }

        break;
      } on APIError catch (error) {
        // Only a rejected request is resent, once, with its `prefer` tools non-strict; never a started stream.
        if (!_strictLimitRejection(error) || disablePreferred || (signal?.aborted ?? false)) {
          rethrow;
        }

        if (!_hasPreferredStrict(model, _sentTools(model, transcript), disablePreferred)) {
          rethrow;
        }

        disablePreferred = true;
        _overStrictLimits.add(options.sessionId);
        output.diagnostics = [
          ...?output.diagnostics,
          Diagnostic(
            type: 'anthropic_strict_tools_fallback',
            timestamp: DateTime.now().millisecondsSinceEpoch,
            error: {'message': error.message},
            details: {'reason': 'strict_tool_limits_exceeded'},
          ),
        ];
      }
    }

    await options.onResponse?.call(ProviderResponse(response.statusCode, response.headers), model);
    stream.push(StartEvent(output));
    final transformations = await _read(model, response.stream, options, output, stream);
    if (signal?.aborted ?? false) {
      throw const AbortedException();
    }

    if (output.stopReason == StopReason.pending) {
      throw const ProviderException('Anthropic stream ended without a stop reason');
    }

    if (output.stopReason == StopReason.aborted || output.stopReason == StopReason.error) {
      throw ProviderException(output.errorMessage ?? 'An unknown error occurred');
    }

    if (thread != null) {
      _recordThread(model, thread, params, output);
    }

    if (transformations != null && transformations.isNotEmpty) {
      output.diagnostics = [
        ...?output.diagnostics,
        Diagnostic(
          type: 'anthropic_input_transformations',
          timestamp: DateTime.now().millisecondsSinceEpoch,
          details: {
            'transformations': [
              for (final t in transformations.cast<Json>())
                {'type': ?t['type'], 'path': ?t['path'], 'reason': ?t['reason']},
            ],
          },
        ),
      ];
    }

    stream.push(DoneEvent(output.stopReason, output));
  } catch (error) {
    output.stopReason = (signal?.aborted ?? false) ? StopReason.aborted : StopReason.error;
    output.errorMessage = switch (error) {
      _ when output.stopReason == StopReason.aborted => 'Request was aborted',
      AnthropicError() => error.message,
      _ => error.toString(),
    };
    stream.push(ErrorEvent(output.stopReason, output));
  } finally {
    owned?.close();
  }
}

/// The open content blocks, by Anthropic's block index: where each sits in the output, and a tool call's arguments.
typedef _Open = ({int index, ToolArgumentsStream? arguments});

/// Reads the response into [output] and [stream]; returns the input transformations Anthropic reported last (how it
/// changed the request's input, such as dropping a thinking block), if any.
Future<List<Object?>?> _read(
  Model model,
  Stream<List<int>> body,
  AnthropicOptions options,
  AssistantMessage output,
  AssistantMessageEventStream stream,
) async {
  final open = <int, _Open>{};
  List<Object?>? transformations;

  // The prices billed: an allowed fallback model's when it answered instead.
  ModelCost? fallbackCost;

  void total() {
    final u = output.usage;
    u.totalTokens = u.input + u.output + u.cacheRead + u.cacheWrite;
    calculateCost(model, u, prices: fallbackCost);
  }

  await for (final event in anthropicEvents(body)) {
    await options.onEvent?.call(event, model);
    switch (event['type']) {
      case 'message_start':
        final message = event['message'] as Json;
        output.responseId = message['id'] as String?;
        if (message['input_transformations'] case final List<Object?> list) {
          transformations = list;
        }

        final answered = message['model'] as String?;
        if (answered != null && answered != model.id) {
          output.responseModel = answered;
          final fallback = _Compat(model).fallbacks.where((f) {
            return f['provider'] == model.provider && f['model'] == answered;
          }).firstOrNull;
          if (fallback?['cost'] case final Json cost) {
            fallbackCost = ModelCost.fromJson(cost);
          }
        }

        // Usage from the start, so an aborted stream still counts the input.
        final usage = message['usage'] as Json? ?? const {};
        output.usage
          ..input = usage['input_tokens'] as int? ?? 0
          ..output = usage['output_tokens'] as int? ?? 0
          ..cacheRead = usage['cache_read_input_tokens'] as int? ?? 0
          ..cacheWrite = usage['cache_creation_input_tokens'] as int? ?? 0
          ..cacheWrite1h = (usage['cache_creation'] as Json?)?['ephemeral_1h_input_tokens'] as int? ?? 0;
        total();
      case 'content_block_start':
        final block = event['content_block'] as Json;
        final Content content;
        switch (block['type']) {
          case 'fallback':
            if (output.content.isNotEmpty) {
              throw const ProviderException('Anthropic performed an unsupported mid-output model fallback');
            }

            continue;
          case 'text':
            content = TextContent(block['text'] as String? ?? '');
          case 'thinking':
            content = ThinkingContent(
              block['thinking'] as String? ?? '',
              signature: block['signature'] as String? ?? '',
            );
          case 'redacted_thinking':
            content = ThinkingContent('[Reasoning redacted]', signature: block['data'] as String?, redacted: true);
          case 'tool_use':
            content = ToolCall(block['id'] as String, block['name'] as String, (block['input'] as Json?) ?? {});
          default:
            continue;
        }

        output.content.add(content);
        final at = output.content.length - 1;
        open[event['index'] as int] = (index: at, arguments: content is ToolCall ? ToolArgumentsStream() : null);
        stream.push(switch (content) {
          TextContent() => TextStartEvent(at, output),
          ToolCall() => ToolCallStartEvent(at, output),
          _ => ThinkingStartEvent(at, output),
        });
      case 'content_block_delta':
        final at = open[event['index'] as int];
        if (at == null) {
          continue;
        }

        final block = output.content[at.index];
        final delta = event['delta'] as Json;
        switch (delta['type']) {
          case 'text_delta' when block is TextContent:
            final text = delta['text'] as String;
            block.text += text;
            stream.push(TextDeltaEvent(at.index, text, output));
          case 'thinking_delta' when block is ThinkingContent:
            final text = delta['thinking'] as String;
            block.thinking += text;
            stream.push(ThinkingDeltaEvent(at.index, text, output));
          case 'input_json_delta' when block is ToolCall:
            final text = delta['partial_json'] as String;
            block.arguments = at.arguments!.append(text);
            stream.push(ToolCallDeltaEvent(at.index, text, output));
          case 'signature_delta' when block is ThinkingContent:
            block.signature = (block.signature ?? '') + (delta['signature'] as String);
        }
      case 'content_block_stop':
        final at = open.remove(event['index'] as int);
        if (at == null) {
          continue;
        }

        final block = output.content[at.index];
        switch (block) {
          case TextContent():
            stream.push(TextEndEvent(at.index, block.text, output));
          case ThinkingContent():
            stream.push(ThinkingEndEvent(at.index, block.thinking, output));
          case ToolCall():
            final finished = at.arguments!.finish();
            block.arguments = finished.arguments;
            if (finished.error != null) {
              block.argumentsError = finished.error;
            }

            stream.push(ToolCallEndEvent(at.index, block, output));
          case ImageContent():
            continue;
        }
      case 'message_delta':
        if (event['input_transformations'] case final List<Object?> list) {
          transformations = list;
        }

        final delta = event['delta'] as Json? ?? const {};
        final reason = delta['stop_reason'] as String?;
        if (reason != null) {
          output.rawStopReason = reason;
          final mapped = mapStopReason(reason, delta['stop_details'] as Json?);
          output.stopReason = mapped.reason;
          if (mapped.error != null) {
            output.errorMessage = mapped.error;
          }
        }

        // Only fields that are present: proxies may leave out the input tokens sent at the start.
        final usage = event['usage'] as Json?;
        if (usage != null) {
          final u = output.usage;
          u.input = usage['input_tokens'] as int? ?? u.input;
          u.output = usage['output_tokens'] as int? ?? u.output;
          u.cacheRead = usage['cache_read_input_tokens'] as int? ?? u.cacheRead;
          u.cacheWrite = usage['cache_creation_input_tokens'] as int? ?? u.cacheWrite;
          u.cacheWrite1h = (usage['cache_creation'] as Json?)?['ephemeral_1h_input_tokens'] as int? ?? u.cacheWrite1h;
          u.reasoning = (usage['output_tokens_details'] as Json?)?['thinking_tokens'] as int? ?? u.reasoning;
        }

        total();
    }
  }

  return transformations;
}

/// The adaptive-thinking effort for [level]: the model's own mapping, else pi's default.
String _effort(Model model, ThinkingLevel level) {
  final mapped = model.levels?[level.name];
  if (mapped != null) {
    return mapped;
  }

  return switch (level) {
    ThinkingLevel.minimal || ThinkingLevel.low => 'low',
    ThinkingLevel.medium => 'medium',
    _ => 'high',
  };
}

/// Streams one response with provider-neutral options: [SimpleStreamOptions.reasoning] becomes an effort for
/// adaptive models, or a thinking budget inside the token cap for older ones.
///
/// Throws at once when the request has no credentials; any later failure ends the stream with an [ErrorEvent].
AssistantMessageEventStream streamSimpleAnthropic(
  Model model,
  List<Message> messages, [
  SimpleStreamOptions options = const SimpleStreamOptions(),
]) {
  AnthropicOptions build({required bool thinking, int? maxTokens, String? effort, int? budget}) {
    return AnthropicOptions(
      signal: options.signal,
      apiKey: options.apiKey,
      headers: options.headers,
      httpClient: options.httpClient,
      timeout: options.timeout,
      maxRetries: options.maxRetries,
      maxRetryDelay: options.maxRetryDelay,
      temperature: options.temperature,
      maxTokens: maxTokens,
      cacheRetention: options.cacheRetention,
      sessionId: options.sessionId,
      metadata: options.metadata,
      onPayload: options.onPayload,
      onResponse: options.onResponse,
      onEvent: options.onEvent,
      disableContinuation: options.disableContinuation,
      thinkingEnabled: thinking,
      effort: effort,
      thinkingBudgetTokens: budget,
      toolChoice: options.toolChoice,
    );
  }

  final base = clampMaxTokensToContext(model, messages, options.maxTokens ?? model.maxTokens);
  _assertAuth(model, build(thinking: false));
  final level = options.reasoning;
  if (level == null || level == ThinkingLevel.off) {
    return streamAnthropic(model, messages, build(thinking: false, maxTokens: base));
  }

  if (_Compat(model).adaptive) {
    return streamAnthropic(model, messages, build(thinking: true, maxTokens: base, effort: _effort(model, level)));
  }

  final adjusted = adjustMaxTokensForThinking(base, model.maxTokens, level, options.budgets);
  final ceiling = clampMaxTokensToContext(model, messages, adjusted.maxTokens);
  return streamAnthropic(
    model,
    messages,
    build(thinking: true, maxTokens: ceiling, budget: min(adjusted.budget, max(0, ceiling - 1024))),
  );
}
