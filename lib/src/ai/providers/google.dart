// The Gemini API provider, ported from pi's api/google-generative-ai.ts and api/google-shared.ts.
//
// pi builds the Gen AI SDK's request shape and lets the SDK convert it; chorus builds the REST body the SDK would send
// (Kai's choice), so onPayload sees that body. Not ported: Vertex AI and Cloud Code Assist, with the OpenAPI
// `parameters` form of tool schemas only they use; identity headers.

import 'dart:async';
import 'dart:convert';

import 'package:google_sdk_lite_dart/google_sdk_lite_dart.dart';
import 'package:http/http.dart' as http;

import '../constrained.dart';
import '../event_stream.dart';
import '../model.dart';
import '../options.dart';
import '../registry.dart' show clampThinkingLevel;
import '../retry.dart';
import '../transcript.dart';
import '../transform.dart';
import '../types.dart';
import 'shared.dart';

/// Thinking for a Gemini request: off, a token budget (-1 lets the model choose, 0 turns it off), or one of Gemini's
/// levels (`MINIMAL`, `LOW`, `MEDIUM`, `HIGH`).
class GoogleThinking {
  const GoogleThinking({required this.enabled, this.budgetTokens, this.level});

  final bool enabled;
  final int? budgetTokens;
  final String? level;
}

/// Options of the Google provider, ported from pi's `GoogleOptions`.
class GoogleOptions extends StreamOptions {
  const GoogleOptions({
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
    this.toolChoice,
    this.thinking,
  });

  /// The provider-neutral [options] as Google options, with none of Google's own set.
  factory GoogleOptions.from(StreamOptions options) {
    return GoogleOptions(
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

  /// `auto`, `none` or `any`.
  final String? toolChoice;
  final GoogleThinking? thinking;
}

/// A Google thinking level: the model's own mapping of [level] (lowercased), else [level] itself; only `minimal`,
/// `low`, `medium` and `high` exist. Throws a [ProviderException] for any other.
String resolveGoogleThinkingLevel(Model model, ThinkingLevel level) {
  final levels = model.levels ?? const <String, String?>{};
  final mapped = levels[level.name];
  final resolved = mapped?.toLowerCase() ?? level.name;
  if (const {'minimal', 'low', 'medium', 'high'}.contains(resolved)) {
    return resolved;
  }

  // JavaScript's String() of the mapping: "null" for a level mapped to null, "undefined" for one not mapped.
  var shown = mapped ?? 'undefined';
  if (levels.containsKey(level.name) && mapped == null) {
    shown = 'null';
  }

  throw ProviderException(
    'Unsupported Google thinking level mapping for ${model.provider}/${model.id}: ${level.name} -> $shown',
  );
}

/// Whether [model] takes Gemini's discrete `thinkingLevel` rather than a token `thinkingBudget`: Gemini 3 Pro and
/// Flash, Gemini 4 and later, the `-latest` Flash aliases, and Gemma 4.
bool usesGoogleThinkingLevel(Model model) {
  final id = model.id.toLowerCase();
  return RegExp(r'gemini-3(?:\.\d+)?-(?:pro|flash)').hasMatch(id) ||
      RegExp(r'gemini-(?:[4-9]|[1-9]\d)').hasMatch(id) ||
      id == 'gemini-flash-latest' ||
      id == 'gemini-flash-lite-latest' ||
      RegExp('gemma-?4').hasMatch(id);
}

/// The thinking config that turns thinking off, or as low as [model] goes when it cannot be turned off.
Json disabledGoogleThinkingConfig(Model model) {
  if (!usesGoogleThinkingLevel(model)) {
    return {'thinkingBudget': 0};
  }

  final fallback = clampThinkingLevel(model, ThinkingLevel.off);
  if (fallback == ThinkingLevel.off) {
    return {'thinkingBudget': 0};
  }

  return {'thinkingLevel': resolveGoogleThinkingLevel(model, fallback).toUpperCase()};
}

/// Whether a response part is thinking: only `thought: true` says so; a `thoughtSignature` can be on any part.
bool isThinkingPart(Json part) => part['thought'] == true;

/// [incoming] when it is a signature, else [existing]: some backends send a block's signature only with its first
/// delta.
String? retainThoughtSignature(String? existing, String? incoming) {
  if (incoming != null && incoming.isNotEmpty) {
    return incoming;
  }

  return existing;
}

final _base64 = RegExp(r'^[A-Za-z0-9+/]+={0,2}$');

/// [signature] when it comes from this model and is base64, as Google requires; otherwise none.
String? _signature(bool sameModel, String? signature) {
  if (!sameModel || signature == null || signature.isEmpty || signature.length % 4 != 0) {
    return null;
  }

  return _base64.hasMatch(signature) ? signature : null;
}

int? _geminiMajor(String modelId) {
  final match = RegExp(r'^gemini(?:-live)?-(\d+)').firstMatch(modelId.toLowerCase());
  return match == null ? null : int.parse(match.group(1)!);
}

/// Whether [modelId] needs tool call ids in function calls and responses: Claude and gpt-oss behind Google APIs, and
/// Gemini 3 and later.
bool requiresToolCallId(String modelId) {
  if (modelId.startsWith('claude-') || modelId.startsWith('gpt-oss-')) {
    return true;
  }

  return (_geminiMajor(modelId) ?? 0) >= 3;
}

/// Whether images go inside function responses: Gemini 3 and later, and models that are not Gemini.
bool _multimodalFunctionResponse(String modelId) => (_geminiMajor(modelId) ?? 3) >= 3;

/// Gemini 3 and later enforce required function parameters in validated function calling.
bool supportsGoogleStrictToolSampling(String modelId) => (_geminiMajor(modelId) ?? 0) >= 3;

/// A text-bearing part in the order the SDK's converter writes its fields.
Json _textPart(String text, {bool thought = false, String? signature}) {
  return {'text': sanitizeSurrogates(text), if (thought) 'thought': true, 'thoughtSignature': ?signature};
}

Json _imagePart(ImageContent image) {
  return {
    'inlineData': {'mimeType': image.mimeType, 'data': image.data},
  };
}

bool _blank(String text) => text.trim().isEmpty;

/// The Gemini `contents` for [messages], ported from pi's `convertMessages`: the system messages are left out (the
/// leading prompt goes as `systemInstruction`), replies become `model` turns, and tool results `user` turns of
/// function responses.
List<Json> convertGoogleMessages(Model model, List<Message> messages) {
  final contents = <Json>[];
  final ids = requiresToolCallId(model.id);

  String normalizeId(String id, AssistantMessage source) {
    if (!ids) {
      return id;
    }

    final safe = id.replaceAll(RegExp('[^a-zA-Z0-9_-]'), '_');
    return safe.length > 64 ? safe.substring(0, 64) : safe;
  }

  for (final message in transformMessages(
    collapseSystemMessages(messages).where((m) => m is! SystemMessage).toList(),
    model,
    normalizeId,
  )) {
    switch (message) {
      case UserMessage(:final content):
        if (content.isEmpty) {
          continue;
        }

        contents.add({
          'role': 'user',
          'parts': [
            for (final c in content)
              if (c is ImageContent) _imagePart(c) else _textPart((c as TextContent).text),
          ],
        });
      case AssistantMessage():
        // Thinking and signatures are kept only for replies of this provider and model.
        final same = message.provider == model.provider && message.model == model.id;
        final parts = <Json>[];
        for (final block in message.content) {
          switch (block) {
            case TextContent():
              final signature = _signature(same, block.signature);

              // An empty part can carry a signature Gemini needs back; only an unsigned one is dropped.
              if (_blank(block.text) && signature == null) {
                continue;
              }

              parts.add(_textPart(block.text, signature: signature));

            // transformMessages keeps thinking only from this model, so pi's plain-text form of another model's
            // thinking is never reached and left out.
            case ThinkingContent():
              final signature = _signature(same, block.signature);
              if (_blank(block.thinking) && signature == null) {
                continue;
              }

              parts.add(_textPart(block.thinking, thought: true, signature: signature));
            case ToolCall():
              parts.add({
                'functionCall': {'args': block.arguments, if (ids) 'id': block.id, 'name': block.name},
                'thoughtSignature': ?_signature(same, block.thoughtSignature),
              });
            case ImageContent() || ServerToolContent():
              continue;
          }
        }

        if (parts.isNotEmpty) {
          contents.add({'role': 'model', 'parts': parts});
        }
      case ToolResultMessage():
        final text = message.content.whereType<TextContent>().map((c) => c.text).join('\n');
        // transformMessages has replaced the images of a model without image input, so pi's check for it is left out.
        final images = [for (final image in message.content.whereType<ImageContent>()) _imagePart(image)];
        final nested = _multimodalFunctionResponse(model.id);
        var value = '';
        if (!_blank(text)) {
          value = sanitizeSurrogates(text);
        } else if (images.isNotEmpty) {
          value = '(see attached image)';
        }

        final response = {
          'functionResponse': {
            'name': message.name,
            'response': message.isError ? {'error': value} : {'output': value},
            if (images.isNotEmpty && nested) 'parts': images,
            if (ids) 'id': message.id,
          },
        };

        // All function responses of a turn go in one user turn, as Cloud Code Assist requires.
        final last = contents.lastOrNull;
        final parts = last?['parts'] as List<Json>?;
        if (last?['role'] == 'user' && (parts?.any((p) => p.containsKey('functionResponse')) ?? false)) {
          parts!.add(response);
        } else {
          contents.add({
            'role': 'user',
            'parts': <Json>[response],
          });
        }

        // Before Gemini 3, images follow in a user turn of their own.
        if (images.isNotEmpty && !nested) {
          contents.add({
            'role': 'user',
            'parts': [_textPart('Tool result image:'), ...images],
          });
        }
      case SystemMessage():
        continue;
    }
  }

  return contents;
}

/// Keywords whose values map names to subschemas; their keys are names, not keywords.
const _nameMaps = {'properties', 'patternProperties', r'$defs', 'definitions', 'dependentSchemas', 'dependencies'};

/// Keywords whose values are a subschema or a list of them, not literal data.
const _subschemas = {
  'additionalProperties',
  'unevaluatedProperties',
  'propertyNames',
  'items',
  'prefixItems',
  'additionalItems',
  'unevaluatedItems',
  'contains',
  'allOf',
  'anyOf',
  'oneOf',
  'not',
  'if',
  'then',
  'else',
  'contentSchema',
};

/// [schema] with string and number `const` values as single-value `enum`s: Gemini ignores `const` in
/// `parametersJsonSchema` but honors `enum`. Literal data (`default`, `examples`...) is left alone.
Object? _constAsEnum(Object? schema, [bool names = false]) {
  if (schema is List) {
    return [for (final item in schema) _constAsEnum(item)];
  }

  if (schema is! Map) {
    return schema;
  }

  final result = <String, Object?>{
    for (final MapEntry(:key, :value) in schema.entries)
      key as String: switch (key) {
        _ when names || _subschemas.contains(key) => _constAsEnum(value),
        _ when _nameMaps.contains(key) => _constAsEnum(value, true),
        _ => value,
      },
  };
  if (!names && result.containsKey('const') && result['enum'] == null) {
    final value = result['const'];
    if (value is String || value is num) {
      result
        ..remove('const')
        ..['enum'] = [value];
    }
  }

  return result;
}

/// [tools] as Gemini function declarations, with JSON Schema parameters; strict where asked for and [strict] allows.
List<Json>? convertGoogleTools(List<Tool> tools, {bool strict = true}) {
  if (tools.isEmpty) {
    return null;
  }

  return [
    {
      'functionDeclarations': [
        for (final tool in tools)
          {
            'name': tool.name,
            'description': tool.description,
            'parametersJsonSchema': _constAsEnum(strictToolParameters(tool, resolveStrictSampling(tool, strict))),
          },
      ],
    },
  ];
}

/// The function calling mode: `NONE` or `ANY` when asked for, `VALIDATED` when a tool is strict, else the tool choice
/// (`AUTO` for anything else), or none.
String? googleFunctionCallingMode(List<Tool> tools, String? toolChoice, bool strict) {
  // First, as in pi: a tool that requires strict sampling the model lacks throws whatever the tool choice.
  final validated = tools.any((t) => resolveStrictSampling(t, strict) == true);
  if (toolChoice == 'none' || toolChoice == 'any') {
    return toolChoice!.toUpperCase();
  }

  if (validated) {
    return 'VALIDATED';
  }

  return toolChoice == null ? null : 'AUTO';
}

/// The stop reason of a Gemini finish reason. Throws a [ProviderException] for one pi does not know.
StopReason mapGoogleStopReason(String reason) {
  return switch (reason) {
    'STOP' => StopReason.stop,

    // CONTINUATION: the request's token limit came first; the continuation token is not used.
    'MAX_TOKENS' || 'CONTINUATION' => StopReason.length,
    'BLOCKLIST' ||
    'PROHIBITED_CONTENT' ||
    'SPII' ||
    'SAFETY' ||
    'IMAGE_SAFETY' ||
    'IMAGE_PROHIBITED_CONTENT' ||
    'IMAGE_RECITATION' ||
    'IMAGE_OTHER' ||
    'RECITATION' ||
    'FINISH_REASON_UNSPECIFIED' ||
    'OTHER' ||
    'LANGUAGE' ||
    'MALFORMED_FUNCTION_CALL' ||
    'UNEXPECTED_TOOL_CALL' ||
    'TOO_MANY_TOOL_CALLS' ||
    'NO_IMAGE' => StopReason.error,
    _ => throw ProviderException('Unhandled stop reason: $reason'),
  };
}

/// The REST body for [messages], as the Gen AI SDK would send the request pi builds: `contents`, the leading prompt as
/// `systemInstruction`, `tools`, `toolConfig`, and `generationConfig`.
Json buildGoogleParams(Model model, List<Message> messages, GoogleOptions options) {
  final collapsed = collapseSystemMessages(messages);
  final tools = currentTools(collapsed);
  final strict = supportsGoogleStrictToolSampling(model.id);
  final mode = tools.isEmpty ? null : googleFunctionCallingMode(tools, options.toolChoice, strict);
  final system = initialSystemMessage(collapsed);
  final instruction = system == null ? '' : systemText(system);
  final thinking = options.thinking;
  Json? thinkingConfig;
  if (thinking != null && model.reasoning) {
    if (!thinking.enabled) {
      thinkingConfig = disabledGoogleThinkingConfig(model);
    } else if (thinking.level != null) {
      thinkingConfig = {'includeThoughts': true, 'thinkingLevel': thinking.level};
    } else {
      thinkingConfig = {'includeThoughts': true, 'thinkingBudget': ?thinking.budgetTokens};
    }
  }

  return {
    'contents': convertGoogleMessages(model, messages),
    if (instruction.isNotEmpty)
      'systemInstruction': {
        'role': 'user',
        'parts': [
          {'text': sanitizeSurrogates(instruction)},
        ],
      },
    'tools': ?convertGoogleTools(tools, strict: strict),
    if (mode != null)
      'toolConfig': {
        'functionCallingConfig': {'mode': mode},
      },
    'generationConfig': {
      'temperature': ?options.temperature,
      'maxOutputTokens': ?options.maxTokens,
      'thinkingConfig': ?thinkingConfig,
    },
  };
}

var _toolCalls = 0;

/// The error message for [error], as pi's `formatProviderError` gives it for Google: the SDK error's message.
String _errorMessage(Object error) {
  return switch (error) {
    ApiError(:final message) ||
    ResponseError(:final message) ||
    ProviderException(:final message) ||
    StateError(:final message) ||
    FormatException(:final message) ||
    http.ClientException(:final message) => message,
    _ => '$error',
  };
}

/// Streams one response from the Gemini API.
///
/// Every failure, including missing credentials, ends the stream with an [ErrorEvent].
AssistantMessageEventStream streamGoogle(
  Model model,
  List<Message> messages, [
  GoogleOptions options = const GoogleOptions(),
]) {
  final stream = AssistantMessageEventStream();
  unawaited(_run(model, messages, options, stream));
  return stream;
}

Future<void> _run(
  Model model,
  List<Message> messages,
  GoogleOptions options,
  AssistantMessageEventStream stream,
) async {
  final output = AssistantMessage(
    api: 'google-generative-ai',
    provider: model.provider,
    model: model.id,
    timestamp: DateTime.now().millisecondsSinceEpoch,
  );
  final signal = options.signal;
  GoogleGenAI? client;
  try {
    final key = options.apiKey;
    if (key == null || key.isEmpty) {
      throw ProviderException('No API key for provider: ${model.provider}');
    }

    final headers = <String, String?>{};
    for (final extra in [model.headers, options.headers]) {
      extra?.forEach((name, value) => headers[name.toLowerCase()] = value);
    }

    client = GoogleGenAI(
      apiKey: key,
      baseUrl: model.baseUrl.isEmpty ? 'https://generativelanguage.googleapis.com/' : model.baseUrl,

      // A model's base URL already holds the API version.
      apiVersion: model.baseUrl.isEmpty ? 'v1beta' : '',
      headers: {for (final MapEntry(:key, :value) in headers.entries) key: ?value},
      httpClient: options.httpClient,
    );
    var params = buildGoogleParams(model, messages, options);
    if (signal?.aborted ?? false) {
      throw const AbortedException();
    }

    params = await options.onPayload?.call(params, model) ?? params;
    // The request goes out, and may fail, before the stream starts.
    final chunks = await retryRequest(
      () => client!.models.generateContentStream(model.id, params, signal: signal?.future),
      maxRetries: options.maxRetries,
      maxDelay: options.maxRetryDelay,
      signal: signal,
    );
    stream.push(StartEvent(output));
    Content? open;

    void close() {
      switch (open) {
        case TextContent(:final text):
          stream.push(TextEndEvent(output.content.length - 1, text, output));
        case ThinkingContent(:final thinking):
          stream.push(ThinkingEndEvent(output.content.length - 1, thinking, output));
        default:
          return;
      }
    }

    await for (final chunk in chunks) {
      await options.onEvent?.call(chunk, model);

      // The first non-empty response id of the stream.
      final id = chunk['responseId'] as String?;
      if ((output.responseId?.isEmpty ?? true) && (id?.isNotEmpty ?? false)) {
        output.responseId = id;
      }

      final candidate = (chunk['candidates'] as List<Object?>?)?.firstOrNull as Json?;
      for (final part in ((candidate?['content'] as Json?)?['parts'] as List<Object?>? ?? const []).cast<Json>()) {
        if (part['text'] case final String text) {
          final thinking = isThinkingPart(part);
          if (open == null || (thinking && open is! ThinkingContent) || (!thinking && open is! TextContent)) {
            close();
            open = thinking ? ThinkingContent('') : TextContent('');
            output.content.add(open);
            stream.push(
              thinking
                  ? ThinkingStartEvent(output.content.length - 1, output)
                  : TextStartEvent(output.content.length - 1, output),
            );
          }

          final signature = part['thoughtSignature'] as String?;
          switch (open) {
            case final ThinkingContent block:
              block
                ..thinking += text
                ..signature = retainThoughtSignature(block.signature, signature);
              stream.push(ThinkingDeltaEvent(output.content.length - 1, text, output));
            case final TextContent block:
              block
                ..text += text
                ..signature = retainThoughtSignature(block.signature, signature);
              stream.push(TextDeltaEvent(output.content.length - 1, text, output));
            default:
              break;
          }
        }

        if (part['functionCall'] case final Json call) {
          close();
          open = null;

          // A missing or repeated id gets a new one.
          final given = call['id'] as String?;
          var callId = given;
          if (given == null || given.isEmpty || output.content.any((b) => b is ToolCall && b.id == given)) {
            callId = '${call['name']}_${DateTime.now().millisecondsSinceEpoch}_${++_toolCalls}';
          }

          final signature = part['thoughtSignature'] as String?;
          final tool = ToolCall(
            callId!,
            call['name'] as String? ?? '',
            (call['args'] as Json?) ?? {},
            thoughtSignature: (signature?.isEmpty ?? true) ? null : signature,
          );
          output.content.add(tool);
          final index = output.content.length - 1;
          stream
            ..push(ToolCallStartEvent(index, output))
            ..push(ToolCallDeltaEvent(index, jsonEncode(tool.arguments), output))
            ..push(ToolCallEndEvent(index, tool, output));
        }
      }

      if (candidate?['finishReason'] case final String reason when reason.isNotEmpty) {
        output
          ..rawStopReason = reason
          ..stopReason = mapGoogleStopReason(reason);
        if (output.stopReason == StopReason.stop && output.content.any((b) => b is ToolCall)) {
          output.stopReason = StopReason.toolUse;
        }
      }

      if (chunk['usageMetadata'] case final Json usage) {
        int count(String key) => (usage[key] as num?)?.toInt() ?? 0;
        output.usage = Usage(
          input: count('promptTokenCount') - count('cachedContentTokenCount'),
          output: count('candidatesTokenCount') + count('thoughtsTokenCount'),
          cacheRead: count('cachedContentTokenCount'),
          reasoning: count('thoughtsTokenCount'),
          totalTokens: count('totalTokenCount'),
        );
        calculateCost(model, output.usage);
      }
    }

    close();
    if (signal?.aborted ?? false) {
      throw const AbortedException();
    }

    if (output.stopReason == StopReason.pending) {
      throw const ProviderException('Google stream ended without a finish reason');
    }

    if (output.stopReason == StopReason.aborted || output.stopReason == StopReason.error) {
      final raw = output.rawStopReason;
      throw ProviderException(raw == null ? 'An unknown error occurred' : 'Provider stopped with: $raw');
    }

    stream.push(DoneEvent(output.stopReason, output));
  } catch (error) {
    output.stopReason = (signal?.aborted ?? false) ? StopReason.aborted : StopReason.error;
    if (output.stopReason == StopReason.aborted) {
      output.errorMessage = 'Request was aborted';
    } else {
      output.errorMessage = _errorMessage(error);
    }

    stream.push(ErrorEvent(output.stopReason, output));
  } finally {
    client?.close();
  }
}

/// The thinking budget of [level] on Gemini 2.5 models: [custom]'s, else pi's per model family; -1 (the model
/// chooses) for others.
int _googleBudget(Model model, String level, ThinkingBudgets? custom) {
  final given = switch (level) {
    'minimal' => custom?.minimal,
    'low' => custom?.low,
    'medium' => custom?.medium,
    _ => custom?.high,
  };
  if (given != null) {
    return given;
  }

  final List<int> budgets;
  if (model.id.contains('2.5-pro')) {
    budgets = [128, 2048, 8192, 32768];
  } else if (model.id.contains('2.5-flash-lite')) {
    budgets = [512, 2048, 8192, 24576];
  } else if (model.id.contains('2.5-flash')) {
    budgets = [128, 2048, 8192, 24576];
  } else {
    return -1;
  }

  return budgets[const ['minimal', 'low', 'medium', 'high'].indexOf(level)];
}

/// Streams one response with provider-neutral options: [SimpleStreamOptions.reasoning], clamped to the model's
/// levels, becomes a Gemini thinking level or a token budget; without it, thinking is turned off.
///
/// Throws at once when the request has no API key; any later failure ends the stream with an [ErrorEvent].
AssistantMessageEventStream streamSimpleGoogle(
  Model model,
  List<Message> messages, [
  SimpleStreamOptions options = const SimpleStreamOptions(),
]) {
  if (options.apiKey?.isEmpty ?? true) {
    throw ProviderException('No API key for provider: ${model.provider}');
  }

  GoogleOptions build(GoogleThinking thinking) {
    return GoogleOptions(
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
      toolChoice: options.toolChoice,
      thinking: thinking,
    );
  }

  final reasoning = options.reasoning;
  final level = reasoning == null ? ThinkingLevel.off : clampThinkingLevel(model, reasoning);
  if (level == ThinkingLevel.off) {
    return streamGoogle(model, messages, build(const GoogleThinking(enabled: false)));
  }

  final resolved = resolveGoogleThinkingLevel(model, level);
  if (usesGoogleThinkingLevel(model)) {
    return streamGoogle(model, messages, build(GoogleThinking(enabled: true, level: resolved.toUpperCase())));
  }

  return streamGoogle(
    model,
    messages,
    build(GoogleThinking(enabled: true, budgetTokens: _googleBudget(model, resolved, options.budgets))),
  );
}
