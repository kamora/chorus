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

import '../event_stream.dart';
import '../json.dart';
import '../model.dart';
import '../options.dart';
import '../retry.dart';
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
      midEffort = model.compat['supportsMidConvoEffort'] as bool? ?? false;

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
/// the last user message.
List<Json> convertMessages(List<Message> messages, Json? cache, {bool emptySignature = false}) {
  final params = <Json>[];
  for (var i = 0; i < messages.length; i++) {
    final message = messages[i];
    switch (message) {
      case SystemMessage():

        // Later system messages are collapsed into the leading one before conversion.
        continue;
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

  if (cache == null || params.isEmpty || params.last['role'] != 'user') {
    return params;
  }

  final last = (params.last['content'] as List<Json>).last;
  if (const {'text', 'image', 'tool_result'}.contains(last['type'])) {
    last['cache_control'] = cache;
  }

  return params;
}

/// Tools as Anthropic declares them, with [cache] on the last one.
List<Json> convertTools(List<Tool> tools, {bool eager = true, Json? cache}) {
  return [
    for (final (i, tool) in tools.indexed)
      {
        'name': tool.name,
        'description': tool.description,
        if (eager) 'eager_input_streaming': true,

        // Anthropic rejects root anyOf, oneOf and allOf, so only these root keywords are kept.
        'input_schema': {
          'type': 'object',
          'properties': tool.parameters['properties'] ?? <String, Object?>{},
          'required': tool.parameters['required'] ?? <Object?>[],
          if (tool.parameters.containsKey('additionalProperties'))
            'additionalProperties': tool.parameters['additionalProperties'],
          if (tool.parameters.containsKey(r'$defs')) r'$defs': tool.parameters[r'$defs'],
          if (tool.parameters.containsKey('definitions')) 'definitions': tool.parameters['definitions'],
        },
        if (cache != null && i == tools.length - 1) 'cache_control': cache,
      },
  ];
}

/// The beta features of a request. An `anthropic-beta` header in the model's or the caller's headers replaces them
/// (a `null` one clears them).
List<String> _betas(Model model, List<Tool> tools, AnthropicOptions options) {
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
  }.toList();
}

/// The request body for [messages], a transcript whose only system message leads.
Json buildParams(Model model, List<Message> messages, AnthropicOptions options) {
  final cache = _cacheControl(model, options.cacheRetention);
  final compat = _Compat(model);
  final system = initialSystemMessage(messages);
  final transformed = transformMessages(messages, model, normalizeToolCallId);
  final tools = currentTools(messages);
  final betas = _betas(model, tools, options);
  final text = system == null ? '' : systemText(system);

  final params = <String, Object?>{
    'model': model.id,
    'messages': convertMessages(
      system == null ? transformed : transformed.sublist(1),
      cache,
      emptySignature: compat.emptySignature,
    ),
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

  if (tools.isNotEmpty) {
    params['tools'] = convertTools(tools, eager: compat.eagerToolInput, cache: compat.cachedTools ? cache : null);
  }

  // A `thinkingLevelMap` with `off: null` marks thinking as impossible to turn off.
  final levels = model.levels ?? const {};
  if (model.reasoning && options.thinkingEnabled == false && !(levels.containsKey('off') && levels['off'] == null)) {
    params['thinking'] = {'type': 'disabled'};
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
  unawaited(_run(model, collapseSystemMessages(messages), options, stream));
  return stream;
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

    var params = buildParams(model, transcript, options);
    final next = await options.onPayload?.call(params, model);
    if (next != null) {
      params = {...next, 'stream': true};
    }

    final http.StreamedResponse response = await retryRequest(
      () => (options.client ?? owned!).beta.messages.create(
        params,
        RequestOptions(signal: signal?.future, timeout: options.timeout),
      ),
      maxRetries: options.maxRetries,
      maxDelay: options.maxRetryDelay,
      signal: signal,
    );
    await options.onResponse?.call(ProviderResponse(response.statusCode, response.headers), model);
    stream.push(StartEvent(output));
    await _read(model, response.stream, options, output, stream);
    if (signal?.aborted ?? false) {
      throw const AbortedException();
    }

    if (output.stopReason == StopReason.pending) {
      throw const ProviderException('Anthropic stream ended without a stop reason');
    }

    if (output.stopReason == StopReason.aborted || output.stopReason == StopReason.error) {
      throw ProviderException(output.errorMessage ?? 'An unknown error occurred');
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

Future<void> _read(
  Model model,
  Stream<List<int>> body,
  AnthropicOptions options,
  AssistantMessage output,
  AssistantMessageEventStream stream,
) async {
  final open = <int, _Open>{};

  void total() {
    final u = output.usage;
    u.totalTokens = u.input + u.output + u.cacheRead + u.cacheWrite;
    calculateCost(model, u);
  }

  await for (final event in anthropicEvents(body)) {
    await options.onEvent?.call(event, model);
    switch (event['type']) {
      case 'message_start':
        final message = event['message'] as Json;
        output.responseId = message['id'] as String?;
        final answered = message['model'] as String?;
        if (answered != null && answered != model.id) {
          output.responseModel = answered;
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
