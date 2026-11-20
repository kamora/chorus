// A scripted provider for tests: it answers each request with the next queued response, streams it in token-sized
// deltas like a real provider, and estimates usage from the request, prompt caching per session included. Ported from
// pi's providers/faux.ts, without deferred responses (no chorus API has them).

import 'dart:async';
import 'dart:convert';
import 'dart:math';

import '../event_stream.dart';
import '../json.dart';
import '../model.dart';
import '../options.dart';
import '../registry.dart';
import '../transcript.dart' show systemText;
import '../types.dart';

TextContent fauxText(String text) => TextContent(text);

ThinkingContent fauxThinking(String thinking) => ThinkingContent(thinking);

var _ids = 0;

/// A tool call; its id is generated unless given.
ToolCall fauxToolCall(String name, Json arguments, {String? id}) {
  return ToolCall(id ?? 'tool:${DateTime.now().millisecondsSinceEpoch}:${_ids++}', name, arguments);
}

/// A response to queue: [content] is a string (one text block), a content block, or a list of them.
AssistantMessage fauxAssistantMessage(
  Object content, {
  StopReason stopReason = StopReason.stop,
  String? errorMessage,
  String? responseId,
  int? timestamp,
}) {
  return AssistantMessage(
    content: switch (content) {
      final String text => [TextContent(text)],
      final Content block => [block],
      final List<Content> blocks => blocks,
      _ => throw ArgumentError.value(content, 'content', 'not a string, a content block or a list of them'),
    },
    api: 'faux',
    provider: 'faux',
    model: 'faux-1',
    stopReason: stopReason,
    errorMessage: errorMessage,
    responseId: responseId,
    timestamp: timestamp ?? DateTime.now().millisecondsSinceEpoch,
  );
}

/// A model the faux provider serves.
class FauxModel {
  const FauxModel(
    this.id, {
    this.name,
    this.reasoning = false,
    this.input = const ['text', 'image'],
    this.inputLimits,
    this.cost = const ModelCost(input: 0, output: 0, cacheRead: 0, cacheWrite: 0),
    this.contextWindow = 128000,
    this.maxTokens = 16384,
  });

  final String id;
  final String? name;
  final bool reasoning;
  final List<String> input;
  final InputLimits? inputLimits;
  final ModelCost cost;
  final int contextWindow;
  final int maxTokens;
}

/// What the faux provider has done.
class FauxState {
  /// Requests received, answered or not.
  var callCount = 0;
}

/// Builds a response from the request: the messages, the options, the provider's state and the requested model.
typedef FauxResponse = FutureOr<AssistantMessage> Function(
  List<Message> messages,
  SimpleStreamOptions options,
  FauxState state,
  Model model,
);

/// A scripted provider: register [provider] in a [Models] collection, queue responses, and stream as with any
/// provider.
///
/// ```dart
/// final faux = FauxProvider();
/// final models = Models()..setProvider(faux.provider);
/// faux.setResponses([fauxAssistantMessage('hi')]);
/// ```
class FauxProvider {
  /// [tokensPerSecond] paces the stream (none: as fast as possible); deltas are [minTokenSize] to [maxTokenSize]
  /// tokens of four characters, sized by [random].
  FauxProvider({
    this.api = 'faux',
    this.id = 'faux',
    List<FauxModel> models = const [FauxModel('faux-1', name: 'Faux Model')],
    this.tokensPerSecond,
    int minTokenSize = 3,
    int maxTokenSize = 5,
    Random? random,
  }) : _minTokenSize = max(1, min(minTokenSize, maxTokenSize)),
       _maxTokenSize = max(max(1, min(minTokenSize, maxTokenSize)), maxTokenSize),
       _random = random ?? Random(),
       models = [
         for (final m in models)
           Model(
             id: m.id,
             name: m.name ?? m.id,
             api: api,
             provider: id,
             baseUrl: 'http://localhost:0',
             reasoning: m.reasoning,
             input: m.input,
             inputLimits: m.inputLimits,
             cost: m.cost,
             contextWindow: m.contextWindow,
             maxTokens: m.maxTokens,
           ),
       ] {
    if (this.models.isEmpty) {
      throw ArgumentError.value(models, 'models', 'must not be empty');
    }
  }

  final String api;

  /// The provider id.
  final String id;
  final List<Model> models;
  final double? tokensPerSecond;
  final int _minTokenSize;
  final int _maxTokenSize;
  final Random _random;
  final state = FauxState();
  var _pending = <Object>[];

  /// The prompt each session sent last, message by message, for the cache simulation.
  final _prompts = <String, List<String>>{};

  /// The provider to register.
  late final provider = Provider(
    id: id,
    models: models,
    apis: {
      api: ProviderStreams(
        stream: (m, messages, o) => streamSimple(m, messages, _simple(o)),
        streamSimple: streamSimple,
      ),
    },
  );

  /// The first model, or the model [modelId], if served.
  Model? getModel([String? modelId]) =>
      modelId == null ? models.first : models.where((m) => m.id == modelId).firstOrNull;

  /// Replaces the queued responses: [AssistantMessage]s and [FauxResponse]s, answered in order.
  void setResponses(List<Object> responses) => _pending = [for (final r in responses) _step(r)];

  void appendResponses(List<Object> responses) => _pending.addAll([for (final r in responses) _step(r)]);

  int get pendingResponseCount => _pending.length;

  /// Streams the next queued response; a request with none queued ends with an error.
  AssistantMessageEventStream streamSimple(
    Model model,
    List<Message> messages, [
    SimpleStreamOptions options = const SimpleStreamOptions(),
  ]) {
    final stream = AssistantMessageEventStream();
    state.callCount++;
    unawaited(_run(stream, _pending.isEmpty ? null : _pending.removeAt(0), model, messages, options));
    return stream;
  }

  Future<void> _run(
    AssistantMessageEventStream stream,
    Object? step,
    Model model,
    List<Message> messages,
    SimpleStreamOptions options,
  ) async {
    await null;
    try {
      await options.onResponse?.call(const ProviderResponse(200, {}), model);
      if (step == null) {
        stream.push(
          ErrorEvent(StopReason.error, _withUsage(_error('No more faux responses queued', model), messages, options)),
        );
        return;
      }

      await _stream(
        stream,
        _withUsage(
          _clone(
            step is AssistantMessage ? step : await (step as FauxResponse)(messages, options, state, model),
            model,
          ),
          messages,
          options,
        ),
        options.signal,
      );
    } catch (error) {
      stream.push(ErrorEvent(StopReason.error, _error(_message(error), model)));
    }
  }

  // [response] as this provider's answer from [model].
  AssistantMessage _clone(AssistantMessage response, Model model) {
    return AssistantMessage.fromJson({...response.toJson(), 'api': api, 'provider': id, 'model': model.id});
  }

  AssistantMessage _error(String message, Model model) {
    return AssistantMessage(
      api: api,
      provider: id,
      model: model.id,
      stopReason: StopReason.error,
      errorMessage: message,
      timestamp: DateTime.now().millisecondsSinceEpoch,
    );
  }

  /// [message] with usage estimated from [messages]: four characters a token. With a session (and caching on), every
  /// prompt token is a cache read up to the first difference from the session's previous prompt, or a cache write.
  AssistantMessage _withUsage(AssistantMessage message, List<Message> messages, SimpleStreamOptions options) {
    final prompt = [for (final m in messages) '${_role(m)}:${_messageText(m)}'];
    final promptTokens = (_joinedLength(prompt) / 4).ceil();
    final output = (_assistantText(message.content).length / 4).ceil();
    var input = promptTokens;
    var cacheRead = 0;
    var cacheWrite = 0;
    final session = options.sessionId;
    if (session != null && options.cacheRetention != CacheRetention.none) {
      final previous = _prompts[session];
      if (previous == null) {
        cacheWrite = promptTokens;
      } else {
        cacheRead = (_commonPrefixLength(previous, prompt) / 4).ceil();
        cacheWrite = max(0, promptTokens - cacheRead);
      }

      input = 0;
      _prompts[session] = prompt;
    }

    message.usage = Usage(
      input: input,
      output: output,
      cacheRead: cacheRead,
      cacheWrite: cacheWrite,
      totalTokens: input + output + cacheRead + cacheWrite,
    );
    return message;
  }

  /// Streams [message] block by block in token-sized deltas.
  Future<void> _stream(AssistantMessageEventStream stream, AssistantMessage message, AbortSignal? signal) async {
    final partial = AssistantMessage(
      api: message.api,
      provider: message.provider,
      model: message.model,
      responseId: message.responseId,
      usage: message.usage,
      timestamp: message.timestamp,
    );

    // Ends the stream as aborted when the signal fired; returns whether it did.
    bool aborted() {
      if (!(signal?.aborted ?? false)) {
        return false;
      }

      partial
        ..stopReason = StopReason.aborted
        ..errorMessage = 'Request was aborted';
      stream.push(ErrorEvent(StopReason.aborted, partial));
      return true;
    }

    if (aborted()) {
      return;
    }

    stream.push(StartEvent(partial));
    for (final (index, block) in message.content.indexed) {
      if (aborted()) {
        return;
      }

      switch (block) {
        case ThinkingContent():
          final open = ThinkingContent('');
          partial.content.add(open);
          stream.push(ThinkingStartEvent(index, partial));
          for (final chunk in _chunks(block.thinking)) {
            await _pace(chunk);
            if (aborted()) {
              return;
            }

            open.thinking += chunk;
            stream.push(ThinkingDeltaEvent(index, chunk, partial));
          }

          stream.push(ThinkingEndEvent(index, block.thinking, partial));
        case TextContent():
          final open = TextContent('');
          partial.content.add(open);
          stream.push(TextStartEvent(index, partial));
          for (final chunk in _chunks(block.text)) {
            await _pace(chunk);
            if (aborted()) {
              return;
            }

            open.text += chunk;
            stream.push(TextDeltaEvent(index, chunk, partial));
          }

          stream.push(TextEndEvent(index, block.text, partial));
        case ToolCall():
          final open = ToolCall(block.id, block.name, {});
          partial.content.add(open);
          stream.push(ToolCallStartEvent(index, partial));

          // Partial arguments come from the parser real providers use, so consumers see them grow.
          final arguments = ToolArgumentsStream();
          for (final chunk in _chunks(jsonEncode(block.arguments))) {
            await _pace(chunk);
            if (aborted()) {
              return;
            }

            open.arguments = arguments.append(chunk);
            stream.push(ToolCallDeltaEvent(index, chunk, partial));
          }

          open.arguments = block.arguments;
          stream.push(ToolCallEndEvent(index, block, partial));
        case ImageContent() || ServerToolContent():
          continue;
      }
    }

    switch (message.stopReason) {
      case StopReason.pending:
        throw StateError('Faux response ended without a stop reason');
      case StopReason.error || StopReason.aborted:
        stream.push(ErrorEvent(message.stopReason, message));
      default:
        stream.push(DoneEvent(message.stopReason, message));
    }
  }

  /// [text] in pieces of [_minTokenSize] to [_maxTokenSize] tokens of four characters; one empty piece for none.
  List<String> _chunks(String text) {
    final chunks = <String>[];
    for (var i = 0; i < text.length;) {
      final size = (_minTokenSize + _random.nextInt(_maxTokenSize - _minTokenSize + 1)) * 4;
      chunks.add(text.substring(i, min(i + size, text.length)));
      i += size;
    }

    return chunks.isEmpty ? [''] : chunks;
  }

  /// Waits as long as [chunk] takes at [tokensPerSecond]; only for the event loop to turn without pacing.
  Future<void> _pace(String chunk) {
    final rate = tokensPerSecond;
    if (rate == null || rate <= 0) {
      return Future<void>.value();
    }

    return Future<void>.delayed(Duration(microseconds: ((chunk.length / 4).ceil() / rate * 1e6).round()));
  }
}

// An error's message, as JavaScript's `error.message` gives it.
String _message(Object error) {
  if (error is StateError) {
    return error.message;
  }

  final text = '$error';
  return text.startsWith('Exception: ') ? text.substring('Exception: '.length) : text;
}

Object _step(Object response) {
  if (response is AssistantMessage || response is FauxResponse) {
    return response;
  }

  throw ArgumentError.value(response, 'response', 'not an AssistantMessage or a FauxResponse');
}

SimpleStreamOptions _simple(StreamOptions o) {
  if (o is SimpleStreamOptions) {
    return o;
  }

  return SimpleStreamOptions(
    signal: o.signal,
    apiKey: o.apiKey,
    headers: o.headers,
    httpClient: o.httpClient,
    timeout: o.timeout,
    maxRetries: o.maxRetries,
    maxRetryDelay: o.maxRetryDelay,
    temperature: o.temperature,
    maxTokens: o.maxTokens,
    cacheRetention: o.cacheRetention,
    sessionId: o.sessionId,
    metadata: o.metadata,
    onPayload: o.onPayload,
    onResponse: o.onResponse,
    onEvent: o.onEvent,
    disableContinuation: o.disableContinuation,
  );
}

String _role(Message message) {
  return switch (message) {
    SystemMessage() => 'system',
    UserMessage() => 'user',
    AssistantMessage() => 'assistant',
    ToolResultMessage() => 'toolResult',
  };
}

String _contentText(List<Content> content) {
  return [
    for (final c in content)
      switch (c) {
        TextContent(:final text) => text,
        ImageContent(:final mimeType, :final data) => '[image:$mimeType:${data.length}]',
        _ => '',
      },
  ].join('\n');
}

String _assistantText(List<Content> content) {
  return [
    for (final c in content)
      switch (c) {
        TextContent(:final text) => text,
        ThinkingContent(:final thinking) => thinking,
        ToolCall(:final name, :final arguments) => '$name:${jsonEncode(arguments)}',
        ImageContent() || ServerToolContent() => '',
      },
  ].join('\n');
}

String _messageText(Message message) {
  return switch (message) {
    SystemMessage() => [
      systemText(message),
      for (final name in message.removed ?? const <String>[]) 'tool-:${jsonEncode({'name': name})}',
      for (final tool in message.added ?? const <Tool>[]) 'tool+:${jsonEncode(tool.toJson())}',
    ].where((part) => part.isNotEmpty).join('\n'),
    UserMessage(:final content) => _contentText(content),
    AssistantMessage(:final content) => _assistantText(content),
    ToolResultMessage(:final name, :final content) => [
      name,
      for (final c in content) _contentText([c]),
    ].join('\n'),
  };
}

/// The length of the prompt that joins [messages] (the first [count]) with blank lines.
int _joinedLength(List<String> messages, [int? count]) {
  final n = count ?? messages.length;
  var length = n > 0 ? (n - 1) * 2 : 0;
  for (var i = 0; i < n; i++) {
    length += messages[i].length;
  }

  return length;
}

/// The length of the common prefix of two joined prompts: equal messages compare whole, characters only from the first
/// message that differs.
int _commonPrefixLength(List<String> previous, List<String> current) {
  var index = 0;
  while (index < previous.length && index < current.length && previous[index] == current[index]) {
    index++;
  }

  String rest(List<String> messages) {
    if (index == messages.length) {
      return '';
    }

    return '${index > 0 ? '\n\n' : ''}${messages.sublist(index).join('\n\n')}';
  }

  final a = rest(previous);
  final b = rest(current);
  var common = 0;
  while (common < a.length && common < b.length && a[common] == b[common]) {
    common++;
  }

  return _joinedLength(previous, index) + common;
}
