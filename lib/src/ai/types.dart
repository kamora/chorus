/// The message, content and event types of chorus, ported from pi-ai's `types.ts`.
///
/// The JSON shapes match pi's, so a transcript written by one can be read by the other. A string `content` in pi's
/// user and system messages is read here as a single [TextContent].
library;

typedef Json = Map<String, Object?>;

/// Thinking levels, `off` included (pi's `ModelThinkingLevel`).
enum ThinkingLevel { off, minimal, low, medium, high, xhigh, max }

/// Prompt cache retention preference; providers map it to what they support.
enum CacheRetention { none, short, long }

enum StopReason { pending, stop, length, toolUse, error, aborted, deferred }

sealed class Content {
  const Content();

  factory Content.fromJson(Json json) {
    return switch (json['type']) {
      'text' => TextContent.fromJson(json),
      'thinking' => ThinkingContent.fromJson(json),
      'image' => ImageContent.fromJson(json),
      'toolCall' => ToolCall.fromJson(json),
      final type => throw FormatException('Unknown content type: $type'),
    };
  }

  Json toJson();
}

class TextContent extends Content {
  TextContent(this.text, {this.signature});

  factory TextContent.fromJson(Json json) {
    return TextContent(json['text'] as String, signature: json['textSignature'] as String?);
  }

  /// Grows through `text_delta` events while the response streams.
  String text;

  /// Provider metadata replayed with the text, e.g. an OpenAI Responses message id.
  String? signature;

  @override
  Json toJson() => {'type': 'text', 'text': text, 'textSignature': ?signature};
}

class ThinkingContent extends Content {
  ThinkingContent(this.thinking, {this.signature, this.redacted = false});

  factory ThinkingContent.fromJson(Json json) {
    return ThinkingContent(
      json['thinking'] as String,
      signature: json['thinkingSignature'] as String?,
      redacted: json['redacted'] as bool? ?? false,
    );
  }

  /// Grows through `thinking_delta` events while the response streams.
  String thinking;

  /// Provider-specific opaque reasoning data, passed back to the API on later turns.
  String? signature;

  /// The thinking was redacted by safety filters; its encrypted payload is in [signature].
  bool redacted;

  @override
  Json toJson() {
    return {'type': 'thinking', 'thinking': thinking, 'thinkingSignature': ?signature, if (redacted) 'redacted': true};
  }
}

class ImageContent extends Content {
  const ImageContent(this.data, this.mimeType);

  factory ImageContent.fromJson(Json json) => ImageContent(json['data'] as String, json['mimeType'] as String);

  /// Base64-encoded image data.
  final String data;

  /// E.g. `image/png`.
  final String mimeType;

  @override
  Json toJson() => {'type': 'image', 'data': data, 'mimeType': mimeType};
}

class ToolCall extends Content {
  ToolCall(this.id, this.name, this.arguments, {this.thoughtSignature, this.namespace, this.argumentsError});

  factory ToolCall.fromJson(Json json) {
    return ToolCall(
      json['id'] as String,
      json['name'] as String,
      Map.of(json['arguments'] as Json),
      thoughtSignature: json['thoughtSignature'] as String?,
      namespace: json['namespace'] as String?,
      argumentsError: json['argumentsError'] as String?,
    );
  }

  final String id;
  final String name;

  /// Updated as the arguments stream in.
  Json arguments;

  /// Google: an opaque signature for reusing thought context.
  String? thoughtSignature;

  /// OpenAI Responses: the namespace of a dynamically loaded or namespaced tool.
  String? namespace;

  /// Set when the complete arguments text was not a JSON object; [arguments] is then empty.
  String? argumentsError;

  @override
  Json toJson() {
    return {
      'type': 'toolCall',
      'id': id,
      'name': name,
      'arguments': arguments,
      'thoughtSignature': ?thoughtSignature,
      'namespace': ?namespace,
      'argumentsError': ?argumentsError,
    };
  }
}

/// Cost in dollars, split like [Usage]'s tokens.
class Cost {
  Cost({this.input = 0, this.output = 0, this.cacheRead = 0, this.cacheWrite = 0, this.total = 0, this.billed});

  factory Cost.fromJson(Json json) {
    return Cost(
      input: (json['input'] as num).toDouble(),
      output: (json['output'] as num).toDouble(),
      cacheRead: (json['cacheRead'] as num).toDouble(),
      cacheWrite: (json['cacheWrite'] as num).toDouble(),
      total: (json['total'] as num).toDouble(),
      billed: json['totalFromProvider'] as bool?,
    );
  }

  double input;
  double output;
  double cacheRead;
  double cacheWrite;

  /// The provider's billed cost when [billed] is true, otherwise the catalog estimate.
  double total;
  bool? billed;

  Json toJson() {
    return {
      'input': input,
      'output': output,
      'cacheRead': cacheRead,
      'cacheWrite': cacheWrite,
      'total': total,
      'totalFromProvider': ?billed,
    };
  }
}

class Usage {
  Usage({
    this.input = 0,
    this.output = 0,
    this.cacheRead = 0,
    this.cacheWrite = 0,
    this.cacheWrite1h,
    this.reasoning,
    this.totalTokens = 0,
    Cost? cost,
  }) : cost = cost ?? Cost();

  factory Usage.fromJson(Json json) {
    return Usage(
      input: json['input'] as int,
      output: json['output'] as int,
      cacheRead: json['cacheRead'] as int,
      cacheWrite: json['cacheWrite'] as int,
      cacheWrite1h: json['cacheWrite1h'] as int?,
      reasoning: json['reasoning'] as int?,
      totalTokens: json['totalTokens'] as int,
      cost: Cost.fromJson(json['cost'] as Json),
    );
  }

  int input;
  int output;
  int cacheRead;
  int cacheWrite;

  /// The part of [cacheWrite] written with 1h retention; only Anthropic reports it.
  int? cacheWrite1h;

  /// Reasoning tokens, a part of [output], when the provider reports them.
  int? reasoning;
  int totalTokens;
  Cost cost;

  Json toJson() {
    return {
      'input': input,
      'output': output,
      'cacheRead': cacheRead,
      'cacheWrite': cacheWrite,
      'cacheWrite1h': ?cacheWrite1h,
      'reasoning': ?reasoning,
      'totalTokens': totalTokens,
      'cost': cost.toJson(),
    };
  }
}

/// A tool the model may call; [parameters] is the JSON Schema of its arguments.
class Tool {
  const Tool(this.name, this.description, this.parameters);

  factory Tool.fromJson(Json json) {
    return Tool(json['name'] as String, json['description'] as String, json['parameters'] as Json);
  }

  final String name;
  final String description;
  final Json parameters;

  Json toJson() => {'name': name, 'description': description, 'parameters': parameters};
}

sealed class Message {
  const Message(this.timestamp);

  factory Message.fromJson(Json json) {
    return switch (json['role']) {
      'system' => SystemMessage.fromJson(json),
      'user' => UserMessage.fromJson(json),
      'assistant' => AssistantMessage.fromJson(json),
      'toolResult' => ToolResultMessage.fromJson(json),
      final role => throw FormatException('Unknown message role: $role'),
    };
  }

  /// Unix time in milliseconds.
  final int timestamp;

  Json toJson();
}

/// System instructions and tool declarations at one point in the transcript.
///
/// The leading system message is the system prompt. Later ones change it: [content] adds instructions from that
/// point on, [sections] replace or remove (`null`) named prompt sections, and [added] and [removed] change the tool
/// set. Replaying every system message in order gives the current prompt and tools.
class SystemMessage extends Message {
  const SystemMessage(this.content, {this.sections, this.added, this.removed, required int timestamp})
    : super(timestamp);

  factory SystemMessage.fromJson(Json json) {
    return SystemMessage(
      _texts(json['content']),
      sections: (json['sections'] as Json?)?.cast<String, String?>(),
      added: (json['toolsAdded'] as List<Object?>?)?.map((t) => Tool.fromJson(t as Json)).toList(),
      removed: (json['toolsRemoved'] as List<Object?>?)?.map((t) => (t as Json)['name'] as String).toList(),
      timestamp: json['timestamp'] as int,
    );
  }

  final List<TextContent> content;
  final Map<String, String?>? sections;
  final List<Tool>? added;

  /// Names of the tools that stop being available here.
  final List<String>? removed;

  @override
  Json toJson() {
    return {
      'role': 'system',
      'content': [for (final c in content) c.toJson()],
      'sections': ?sections,
      'toolsAdded': ?added?.map((t) => t.toJson()).toList(),
      'toolsRemoved': ?removed?.map((n) => {'name': n}).toList(),
      'timestamp': timestamp,
    };
  }
}

class UserMessage extends Message {
  const UserMessage(this.content, {required int timestamp}) : super(timestamp);

  factory UserMessage.fromJson(Json json) {
    final content = json['content'];
    if (content is String) {
      return UserMessage([TextContent(content)], timestamp: json['timestamp'] as int);
    }

    return UserMessage([
      for (final c in content as List<Object?>) Content.fromJson(c as Json),
    ], timestamp: json['timestamp'] as int);
  }

  /// [TextContent] and [ImageContent] only.
  final List<Content> content;

  @override
  Json toJson() {
    return {
      'role': 'user',
      'content': [for (final c in content) c.toJson()],
      'timestamp': timestamp,
    };
  }
}

class AssistantMessage extends Message {
  AssistantMessage({
    List<Content>? content,
    required this.api,
    required this.provider,
    required this.model,
    this.responseModel,
    this.responseId,
    this.providerThinkingLevel,
    this.thinkingLevel,
    Usage? usage,
    this.stopReason = StopReason.pending,
    this.errorMessage,
    this.rawStopReason,
    this.endTurn,
    required int timestamp,
    this.durationMs,
  }) : content = content ?? [],
       usage = usage ?? Usage(),
       super(timestamp);

  factory AssistantMessage.fromJson(Json json) {
    final level = json['thinkingLevel'] as String?;
    return AssistantMessage(
      content: [for (final c in json['content'] as List<Object?>) Content.fromJson(c as Json)],
      api: json['api'] as String,
      provider: json['provider'] as String,
      model: json['model'] as String,
      responseModel: json['responseModel'] as String?,
      responseId: json['responseId'] as String?,
      providerThinkingLevel: json['providerThinkingLevel'] as String?,
      thinkingLevel: level == null ? null : ThinkingLevel.values.byName(level),
      usage: Usage.fromJson(json['usage'] as Json),
      stopReason: StopReason.values.byName(json['stopReason'] as String),
      errorMessage: json['errorMessage'] as String?,
      rawStopReason: json['rawStopReason'] as String?,
      endTurn: json['endTurn'] as bool?,
      timestamp: json['timestamp'] as int,
      durationMs: json['durationMs'] as int?,
    );
  }

  /// [TextContent], [ThinkingContent] and [ToolCall]; grows while the response streams.
  final List<Content> content;
  final String api;
  final String provider;
  final String model;

  /// The concrete model the provider reports, when it does.
  String? responseModel;
  String? responseId;

  /// The provider-native effort level used for this response.
  String? providerThinkingLevel;

  /// The thinking level the agent loop asked for.
  ThinkingLevel? thinkingLevel;
  Usage usage;
  StopReason stopReason;
  String? errorMessage;
  String? rawStopReason;

  /// Whether the provider says the model ended its turn; kept for debugging.
  bool? endTurn;

  /// Milliseconds from [timestamp] until the response ended, set by [AssistantMessageEventStream].
  int? durationMs;

  @override
  Json toJson() {
    return {
      'role': 'assistant',
      'content': [for (final c in content) c.toJson()],
      'api': api,
      'provider': provider,
      'model': model,
      'responseModel': ?responseModel,
      'responseId': ?responseId,
      'providerThinkingLevel': ?providerThinkingLevel,
      'thinkingLevel': ?thinkingLevel?.name,
      'usage': usage.toJson(),
      'stopReason': stopReason.name,
      'errorMessage': ?errorMessage,
      'rawStopReason': ?rawStopReason,
      'endTurn': ?endTurn,
      'timestamp': timestamp,
      'durationMs': ?durationMs,
    };
  }
}

class ToolResultMessage extends Message {
  const ToolResultMessage({
    required this.id,
    required this.name,
    required this.content,
    this.details,
    this.usage,
    required this.isError,
    required int timestamp,
    this.durationMs,
  }) : super(timestamp);

  factory ToolResultMessage.fromJson(Json json) {
    final usage = json['usage'] as Json?;
    return ToolResultMessage(
      id: json['toolCallId'] as String,
      name: json['toolName'] as String,
      content: [for (final c in json['content'] as List<Object?>) Content.fromJson(c as Json)],
      details: json['details'],
      usage: usage == null ? null : Usage.fromJson(usage),
      isError: json['isError'] as bool,
      timestamp: json['timestamp'] as int,
      durationMs: json['durationMs'] as int?,
    );
  }

  /// The id of the [ToolCall] this answers.
  final String id;

  /// The name of the tool that ran.
  final String name;

  /// [TextContent] and [ImageContent] only.
  final List<Content> content;

  /// Any JSON value the tool attaches for the app; not sent to the model.
  final Object? details;

  /// Usage of the tool's own execution, outside the main context accounting.
  final Usage? usage;
  final bool isError;
  final int? durationMs;

  @override
  Json toJson() {
    return {
      'role': 'toolResult',
      'toolCallId': id,
      'toolName': name,
      'content': [for (final c in content) c.toJson()],
      'details': ?details,
      'usage': ?usage?.toJson(),
      'isError': isError,
      'timestamp': timestamp,
      'durationMs': ?durationMs,
    };
  }
}

List<TextContent> _texts(Object? content) {
  if (content is String) {
    return [TextContent(content)];
  }

  return [for (final c in content as List<Object?>) TextContent.fromJson(c as Json)];
}

/// One step of a streamed assistant response.
///
/// A successful stream emits [StartEvent] first and ends with [DoneEvent]. It may end with [ErrorEvent] instead,
/// even before [StartEvent] when the request fails to start. [partial] is the live response so far, not a snapshot:
/// text and thinking are empty at their start event and grow through their deltas until the end event.
sealed class AssistantMessageEvent {
  const AssistantMessageEvent();
}

/// An event inside the response, carrying the response so far.
sealed class PartialEvent extends AssistantMessageEvent {
  const PartialEvent(this.partial);

  final AssistantMessage partial;
}

/// An event about one content block, at [index] in the response's content.
sealed class BlockEvent extends PartialEvent {
  const BlockEvent(this.index, super.partial);

  final int index;
}

class StartEvent extends PartialEvent {
  const StartEvent(super.partial);
}

class TextStartEvent extends BlockEvent {
  const TextStartEvent(super.index, super.partial);
}

class TextDeltaEvent extends BlockEvent {
  const TextDeltaEvent(super.index, this.delta, super.partial);

  final String delta;
}

class TextEndEvent extends BlockEvent {
  const TextEndEvent(super.index, this.content, super.partial);

  final String content;
}

class ThinkingStartEvent extends BlockEvent {
  const ThinkingStartEvent(super.index, super.partial);
}

class ThinkingDeltaEvent extends BlockEvent {
  const ThinkingDeltaEvent(super.index, this.delta, super.partial);

  final String delta;
}

class ThinkingEndEvent extends BlockEvent {
  const ThinkingEndEvent(super.index, this.content, super.partial);

  final String content;
}

class ToolCallStartEvent extends BlockEvent {
  const ToolCallStartEvent(super.index, super.partial);
}

class ToolCallDeltaEvent extends BlockEvent {
  const ToolCallDeltaEvent(super.index, this.delta, super.partial);

  /// The next piece of the arguments' JSON text.
  final String delta;
}

class ToolCallEndEvent extends BlockEvent {
  const ToolCallEndEvent(super.index, this.call, super.partial);

  final ToolCall call;
}

/// The response finished; [reason] is [StopReason.stop], [StopReason.length], [StopReason.toolUse] or
/// [StopReason.deferred].
class DoneEvent extends AssistantMessageEvent {
  const DoneEvent(this.reason, this.message);

  final StopReason reason;
  final AssistantMessage message;
}

/// The response failed; [reason] is [StopReason.error] or [StopReason.aborted], and [error] carries the message.
class ErrorEvent extends AssistantMessageEvent {
  const ErrorEvent(this.reason, this.error);

  final StopReason reason;
  final AssistantMessage error;
}
