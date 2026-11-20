/// The agent loop's types, ported from pi-agent-core's `types.ts`.
library;

import 'dart:async';

import '../ai/event_stream.dart';
import '../ai/model.dart';
import '../ai/options.dart';
import '../ai/types.dart';

/// One message of the agent's transcript: a [Message], or a message of the app's own that `convertToLlm` turns into
/// messages for the model or leaves out.
typedef AgentMessage = Object;

/// Streams one model response. Failures must end the stream with an [ErrorEvent], never throw.
typedef StreamFn = FutureOr<AssistantMessageEventStream> Function(
  Model model,
  List<Message> messages,
  SimpleStreamOptions options,
);

/// How the tool calls of one assistant message run: one after another, or all at once once each is prepared.
enum ToolExecution { sequential, parallel }

/// A tool's result, final or partial.
class AgentToolResult {
  AgentToolResult(this.content, {this.details, this.usage, this.isError = false, this.terminate = false});

  /// An error result with [message] as its text.
  factory AgentToolResult.error(String message) => AgentToolResult([TextContent(message)], details: const {});

  /// Text and images for the model.
  final List<Content> content;

  /// Any JSON value for logs and the app; not sent to the model.
  final Object? details;

  /// Usage of the tool's own execution, outside the main context accounting.
  final Usage? usage;

  /// A failure reported without throwing: the model sees [content] as an error.
  final bool isError;

  /// Asks the agent to stop after this batch of tool calls; it stops only when every result in the batch asks.
  bool terminate;
}

/// A tool the agent can run.
class AgentTool extends Tool {
  const AgentTool({
    required String name,
    required String description,
    required Json parameters,
    required this.execute,
    String? label,
    this.prepareArguments,
    this.executionMode,
  }) : label = label ?? name,
       super(name, description, parameters);

  /// A name for display.
  final String label;

  /// Fixes up the model's raw arguments before validation, for tools whose arguments changed shape.
  final Json Function(Json arguments)? prepareArguments;

  /// Overrides the loop's [ToolExecution] for this tool; one sequential tool makes its whole batch sequential.
  final ToolExecution? executionMode;

  /// Runs the call. Throw on failure, or return a result with `isError`. [onUpdate] reports partial results.
  final Future<AgentToolResult> Function(
    String id,
    Object? arguments,
    AbortSignal? signal,
    void Function(AgentToolResult partial) onUpdate,
  )
  execute;
}

/// The transcript and tools of one run.
class AgentContext {
  AgentContext(this.messages, {this.tools = const []});

  final List<AgentMessage> messages;

  /// The tools the agent can run; differences from the tools the transcript declares are announced to the model.
  final List<AgentTool> tools;
}

/// Returned by [AgentLoopConfig.beforeToolCall]: [block] answers the call with an error saying [reason] instead of
/// running it.
class BeforeToolCall {
  const BeforeToolCall({this.block = false, this.reason, this.terminate = false});

  final bool block;
  final String? reason;

  /// For a blocked call: asks the agent to stop after this batch.
  final bool terminate;
}

/// Returned by [AgentLoopConfig.afterToolCall]: each field set replaces the result's.
class AfterToolCall {
  const AfterToolCall({this.content, this.details, this.usage, this.isError, this.terminate});

  final List<Content>? content;
  final Object? details;
  final Usage? usage;
  final bool? isError;
  final bool? terminate;
}

/// One tool call, as the hooks see it.
class ToolCallContext {
  const ToolCallContext(this.message, this.call, this.arguments, this.context);

  /// The assistant message that made the call.
  final AssistantMessage message;
  final ToolCall call;

  /// The arguments after preparation and validation.
  final Object? arguments;
  final AgentContext context;
}

/// A finished assistant turn: its message, the results of its tool calls, and the run so far.
class AgentTurn {
  const AgentTurn(this.message, this.results, this.context, this.added);

  final AssistantMessage message;
  final List<ToolResultMessage> results;
  final AgentContext context;

  /// The messages this run has added; what the run returns if it ends here.
  final List<AgentMessage> added;
}

/// What [AgentLoopConfig.finishTurn] decides: [proceed] makes sure one more request follows, [end] stops the run.
enum TurnDecision { proceed, end }

/// State to use from the next request on; fields left `null` keep theirs.
class TurnUpdate {
  const TurnUpdate({this.context, this.messages, this.model, this.level});

  final AgentContext? context;

  /// Messages to add before the next request (only from [AgentLoopConfig.prepareNextTurn]).
  final List<AgentMessage>? messages;
  final Model? model;
  final ThinkingLevel? level;
}

/// The hooks a tool call runs through.
class ToolHooks {
  const ToolHooks({this.beforeToolCall, this.afterToolCall});

  /// Called after a call's arguments are validated, before it runs.
  final Future<BeforeToolCall?> Function(ToolCallContext call, AbortSignal? signal)? beforeToolCall;

  /// Called after a call ran, before its end event; may replace parts of its result.
  final Future<AfterToolCall?> Function(
    ToolCallContext call,
    AgentToolResult result,
    bool isError,
    AbortSignal? signal,
  )?
  afterToolCall;
}

/// The settings of one agent run, ported from pi's `AgentLoopConfig`.
class AgentLoopConfig extends ToolHooks {
  const AgentLoopConfig({
    required this.model,
    required this.convertToLlm,
    this.level = ThinkingLevel.off,
    this.options = const SimpleStreamOptions(),
    this.transformContext,
    this.getApiKey,
    this.finishTurn,
    this.prepareRequest,
    this.prepareNextTurn,
    this.getSteeringMessages,
    this.getFollowUpMessages,
    this.toolExecution = ToolExecution.parallel,
    super.beforeToolCall,
    super.afterToolCall,
  });

  final Model model;

  /// The thinking level of requests.
  final ThinkingLevel level;

  /// Request options; the loop sets the thinking level, API key and abort signal itself.
  final SimpleStreamOptions options;

  /// Turns the transcript into messages for the model before each request, leaving out the app's own messages that
  /// the model should not see. Must not throw.
  final FutureOr<List<Message>> Function(List<AgentMessage> messages) convertToLlm;

  /// Changes the transcript before [convertToLlm], e.g. to prune it to the context window. Must not throw.
  final Future<List<AgentMessage>> Function(List<AgentMessage> messages, AbortSignal? signal)? transformContext;

  /// The API key for a provider, asked before each request, for keys that expire. Must not throw.
  final FutureOr<String?> Function(String provider)? getApiKey;

  /// Called after a turn's tool results, before `TurnEnd`. An error or abort ends the run whatever it returns.
  final FutureOr<TurnDecision?> Function(AgentTurn turn, AbortSignal? signal)? finishTurn;

  /// Called before every request, the first included, after pending messages are added. Its update holds for the
  /// rest of the run; its `messages` are ignored.
  final FutureOr<TurnUpdate?> Function(AgentContext context, Model model, ThinkingLevel level, AbortSignal? signal)?
  prepareRequest;

  /// Called after `TurnEnd` when the run goes on, before the next turn starts.
  final FutureOr<TurnUpdate?> Function(AgentTurn turn)? prepareNextTurn;

  /// Messages the user sent while the agent worked, added before the next request. Must not throw.
  final Future<List<AgentMessage>> Function()? getSteeringMessages;

  /// Messages to go on with when the agent would stop. Must not throw.
  final Future<List<AgentMessage>> Function()? getFollowUpMessages;

  final ToolExecution toolExecution;

  AgentLoopConfig copyWith({Model? model, ThinkingLevel? level}) {
    return AgentLoopConfig(
      model: model ?? this.model,
      convertToLlm: convertToLlm,
      level: level ?? this.level,
      options: options,
      transformContext: transformContext,
      getApiKey: getApiKey,
      finishTurn: finishTurn,
      prepareRequest: prepareRequest,
      prepareNextTurn: prepareNextTurn,
      getSteeringMessages: getSteeringMessages,
      getFollowUpMessages: getFollowUpMessages,
      toolExecution: toolExecution,
      beforeToolCall: beforeToolCall,
      afterToolCall: afterToolCall,
    );
  }
}

/// What happens in an agent run, for UIs and logs.
sealed class AgentEvent {
  const AgentEvent();
}

class AgentStart extends AgentEvent {
  const AgentStart();
}

/// The last event of a run, with the messages it added.
class AgentEnd extends AgentEvent {
  const AgentEnd(this.messages);

  final List<AgentMessage> messages;
}

/// A turn is one assistant response and the results of its tool calls.
class TurnStart extends AgentEvent {
  const TurnStart();
}

class TurnEnd extends AgentEvent {
  const TurnEnd(this.message, this.results);

  final AssistantMessage message;
  final List<ToolResultMessage> results;
}

class MessageStart extends AgentEvent {
  const MessageStart(this.message);

  final AgentMessage message;
}

/// A step of a streaming assistant message.
class MessageUpdate extends AgentEvent {
  const MessageUpdate(this.message, this.event);

  final AssistantMessage message;
  final AssistantMessageEvent event;
}

class MessageEnd extends AgentEvent {
  const MessageEnd(this.message);

  final AgentMessage message;
}

class ToolExecutionStart extends AgentEvent {
  const ToolExecutionStart(this.call);

  final ToolCall call;
}

class ToolExecutionUpdate extends AgentEvent {
  const ToolExecutionUpdate(this.call, this.partial);

  final ToolCall call;
  final AgentToolResult partial;
}

class ToolExecutionEnd extends AgentEvent {
  const ToolExecutionEnd(this.call, this.result, this.isError, [this.durationMs]);

  final ToolCall call;
  final AgentToolResult result;
  final bool isError;

  /// How long the tool ran; `null` when it did not run.
  final int? durationMs;
}
