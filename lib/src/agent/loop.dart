/// The agent loop, ported from pi-agent-core's `agent-loop.ts`: it works on the agent's own messages throughout and
/// turns them into model messages only for each request.
library;

import 'dart:async';
import 'dart:convert';

import '../ai/event_stream.dart';
import '../ai/options.dart';
import '../ai/transcript.dart';
import '../ai/types.dart';
import 'types.dart';

/// Receives the events of a run; the loop waits for each.
typedef AgentEventSink = FutureOr<void> Function(AgentEvent event);

/// The events of one run; its result is the messages the run added.
class AgentEventStream extends EventStream<AgentEvent, List<AgentMessage>> {
  AgentEventStream() : super((e) => e is AgentEnd, (e) => (e as AgentEnd).messages);
}

/// Starts a run with [prompts] added to [context], and streams its events.
AgentEventStream agentLoop(
  List<AgentMessage> prompts,
  AgentContext context,
  AgentLoopConfig config,
  StreamFn stream, [
  AbortSignal? signal,
]) {
  final events = AgentEventStream();
  unawaited(runAgentLoop(prompts, context, config, events.push, stream, signal).then(events.end));
  return events;
}

/// Continues a run from [context] as it is, for a retry: its last message must turn into a user message or a tool
/// result for the model.
AgentEventStream agentLoopContinue(
  AgentContext context,
  AgentLoopConfig config,
  StreamFn stream, [
  AbortSignal? signal,
]) {
  _checkContinue(context);
  final events = AgentEventStream();
  unawaited(runAgentLoopContinue(context, config, events.push, stream, signal).then(events.end));
  return events;
}

void _checkContinue(AgentContext context) {
  if (context.messages.isEmpty) {
    throw StateError('Cannot continue: no messages in context');
  }

  if (context.messages.last is AssistantMessage) {
    throw StateError('Cannot continue from message role: assistant');
  }
}

/// Runs the loop with [prompts], sending its events to [emit]; returns the messages it added.
Future<List<AgentMessage>> runAgentLoop(
  List<AgentMessage> prompts,
  AgentContext context,
  AgentLoopConfig config,
  AgentEventSink emit,
  StreamFn stream, [
  AbortSignal? signal,
]) async {
  final initial = _declareToolChanges(context, prompts);
  final added = [...initial];
  await emit(const AgentStart());
  await emit(const TurnStart());
  for (final message in initial) {
    await emit(MessageStart(message));
    await emit(MessageEnd(message));
  }

  await _run(
    AgentContext([...context.messages, ...initial], tools: context.tools),
    added,
    config,
    signal,
    emit,
    stream,
  );
  return added;
}

/// Runs the loop from [context] as it is, sending its events to [emit]; returns the messages it added.
Future<List<AgentMessage>> runAgentLoopContinue(
  AgentContext context,
  AgentLoopConfig config,
  AgentEventSink emit,
  StreamFn stream, [
  AbortSignal? signal,
]) async {
  _checkContinue(context);
  final added = <AgentMessage>[];
  await emit(const AgentStart());
  await emit(const TurnStart());
  await _run(AgentContext([...context.messages], tools: context.tools), added, config, signal, emit, stream);
  return added;
}

Future<void> _run(
  AgentContext initial,
  List<AgentMessage> added,
  AgentLoopConfig initialConfig,
  AbortSignal? signal,
  AgentEventSink emit,
  StreamFn stream,
) async {
  var context = initial;
  var config = initialConfig;
  AgentTurn? last;
  var proceed = false;

  // Steering the user sent while waiting for the run to start.
  var pending = await config.getSteeringMessages?.call() ?? <AgentMessage>[];
  while (true) {
    var calls = true;
    while (calls || pending.isNotEmpty) {
      var prepared = <AgentMessage>[];
      if (last != null) {
        final update = await config.prepareNextTurn?.call(last);
        if (update != null) {
          context = update.context ?? context;
          prepared = update.messages ?? [];
          config = config.copyWith(model: update.model, level: update.level);
        }

        // Preparation can take long (a compaction, say), so steering sent meanwhile is picked up. Only when the
        // earlier poll found nothing: otherwise one-at-a-time queues would deliver two messages in this turn.
        if (pending.isEmpty) {
          pending = await config.getSteeringMessages?.call() ?? [];
        }

        await emit(const TurnStart());
      }

      for (final message in _declareToolChanges(context, [...prepared, ...pending])) {
        await emit(MessageStart(message));
        await emit(MessageEnd(message));
        context.messages.add(message);
        added.add(message);
      }

      pending = [];
      final request = await config.prepareRequest?.call(context, config.model, config.level, signal);
      if (request != null) {
        context = request.context ?? context;
        config = config.copyWith(model: request.model, level: request.level);
      }

      final message = await _respond(context, config, signal, emit, stream);
      added.add(message);
      if (message.stopReason == StopReason.error || message.stopReason == StopReason.aborted) {
        last = AgentTurn(message, const [], context, added);
        await config.finishTurn?.call(last, signal);
        await emit(TurnEnd(message, const []));
        await emit(AgentEnd(added));
        return;
      }

      final toolCalls = message.content.whereType<ToolCall>().toList();
      final results = <ToolResultMessage>[];
      calls = false;
      if (toolCalls.isNotEmpty) {
        // A `length` stop cut the output at the token limit, so any call may carry cut-off arguments: none runs.
        final batch = message.stopReason == StopReason.length
            ? await _failTruncated(toolCalls, emit)
            : await _execute(context, message, config, signal, emit);
        results.addAll(batch.messages);
        calls = !batch.terminate;
        context.messages.addAll(results);
        added.addAll(results);
      }

      last = AgentTurn(message, results, context, added);
      final decision = await config.finishTurn?.call(last, signal);
      await emit(TurnEnd(message, results));
      if (decision == TurnDecision.end) {
        await emit(AgentEnd(added));
        return;
      }

      proceed = decision == TurnDecision.proceed;
      pending = await config.getSteeringMessages?.call() ?? [];
      if (calls || pending.isNotEmpty) {
        proceed = false;
      }
    }

    // The agent would stop here, unless follow-up messages wait.
    final followUp = await config.getFollowUpMessages?.call() ?? [];
    if (followUp.isNotEmpty) {
      proceed = false;
      pending = followUp;
      continue;
    }

    // `finishTurn` asked to go on and nothing else made a request: one more, with the context as it is.
    if (proceed) {
      proceed = false;
      continue;
    }

    break;
  }

  await emit(AgentEnd(added));
}

/// [pending] with the tool changes declared to the model.
///
/// The context's tools are what the agent can run; the transcript's system messages declare what the model may
/// call. Before each request the difference becomes `toolsAdded` and `toolsRemoved` of a system message: the last
/// pending system message's tool changes are replaced by it, or a new one goes before the first pending message that
/// is not a system message.
List<AgentMessage> _declareToolChanges(AgentContext context, List<AgentMessage> pending) {
  final at = pending.lastIndexWhere((m) => m is SystemMessage);
  final system = at == -1 ? null : pending[at] as SystemMessage;
  final baseline = [for (final (i, m) in pending.indexed) i == at ? _withChanges(system!, const [], const []) : m];
  final changes = toolStateChanges(
    currentTools([...context.messages, ...baseline].whereType<Message>().toList()),
    context.tools,
  );
  final unchanged = changes.added.isEmpty && changes.removed.isEmpty;
  if (system != null) {
    // The caller's message is kept when it already declares no tool changes.
    if (unchanged && (system.added?.isEmpty ?? true) && (system.removed?.isEmpty ?? true)) {
      return pending;
    }

    return [for (final (i, m) in baseline.indexed) i == at ? _withChanges(system, changes.added, changes.removed) : m];
  }

  if (unchanged) {
    return pending;
  }

  final first = pending.indexWhere((m) => m is! SystemMessage);
  final index = first == -1 ? pending.length : first;
  return [
    ...pending.sublist(0, index),
    _withChanges(
      SystemMessage(const [], timestamp: DateTime.now().millisecondsSinceEpoch),
      changes.added,
      changes.removed,
    ),
    ...pending.sublist(index),
  ];
}

SystemMessage _withChanges(SystemMessage message, List<Tool> added, List<String> removed) {
  return SystemMessage(
    message.content,
    sections: message.sections,
    added: added.isEmpty ? null : added,
    removed: removed.isEmpty ? null : removed,
    timestamp: message.timestamp,
  );
}

/// Streams one assistant response, keeping its partial message last in the context while it streams.
Future<AssistantMessage> _respond(
  AgentContext context,
  AgentLoopConfig config,
  AbortSignal? signal,
  AgentEventSink emit,
  StreamFn stream,
) async {
  final messages = await config.convertToLlm(
    config.transformContext == null ? context.messages : await config.transformContext!(context.messages, signal),
  );
  final o = config.options;
  final response = await stream(
    config.model,
    messages,
    SimpleStreamOptions(
      signal: signal,
      apiKey: await config.getApiKey?.call(config.model.provider) ?? o.apiKey,
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
      reasoning: config.level == ThinkingLevel.off ? null : config.level,
      budgets: o.budgets,
      toolChoice: o.toolChoice,
    ),
  );

  var partial = false;
  await for (final event in response) {
    switch (event) {
      case StartEvent():
        context.messages.add(event.partial);
        partial = true;
        await emit(MessageStart(event.partial));
      case BlockEvent() when partial:
        context.messages.last = event.partial;
        await emit(MessageUpdate(event.partial, event));
      case DoneEvent() || ErrorEvent():
        break;
      default:
        continue;
    }
  }

  // The level asked for is recorded, whichever stream function answered.
  final last = (await response.result)..thinkingLevel = config.level;
  if (partial) {
    context.messages.last = last;
  } else {
    context.messages.add(last);
    await emit(MessageStart(last));
  }

  await emit(MessageEnd(last));
  return last;
}

typedef _Batch = ({List<ToolResultMessage> messages, bool terminate});

/// A tool call's final outcome, after the hooks.
typedef _Outcome = ({ToolCall call, AgentToolResult result, bool isError, int? durationMs});

/// Answers every call of a message cut off at the token limit with an error, without running it.
Future<_Batch> _failTruncated(List<ToolCall> calls, AgentEventSink emit) async {
  final messages = <ToolResultMessage>[];
  for (final call in calls) {
    await emit(ToolExecutionStart(call));
    final _Outcome outcome = (
      call: call,
      result: AgentToolResult.error(
        'Tool call "${call.name}" was not executed: the response hit the output token limit, so its arguments may be '
        'truncated. Re-issue the tool call with complete arguments.',
      ),
      isError: true,
      durationMs: null,
    );
    await _emitEnd(outcome, emit);
    messages.add(await _emitResult(outcome, emit));
  }

  return (messages: messages, terminate: false);
}

Future<_Batch> _execute(
  AgentContext context,
  AssistantMessage message,
  AgentLoopConfig config,
  AbortSignal? signal,
  AgentEventSink emit,
) async {
  final calls = message.content.whereType<ToolCall>().toList();
  final sequential = calls.any((c) {
    return context.tools.where((t) => t.name == c.name).firstOrNull?.executionMode == ToolExecution.sequential;
  });
  if (config.toolExecution == ToolExecution.sequential || sequential) {
    return _sequential(context, message, calls, config, signal, emit);
  }

  return _parallel(context, message, calls, config, signal, emit);
}

Future<_Batch> _sequential(
  AgentContext context,
  AssistantMessage message,
  List<ToolCall> calls,
  AgentLoopConfig config,
  AbortSignal? signal,
  AgentEventSink emit,
) async {
  final outcomes = <_Outcome>[];
  final messages = <ToolResultMessage>[];
  for (final call in calls) {
    await emit(ToolExecutionStart(call));
    final _Outcome outcome;
    switch (await _prepare(context, message, call, config, signal, context.tools)) {
      case _Immediate(:final result):
        outcome = (call: call, result: result, isError: true, durationMs: null);
      case final _Prepared prepared:
        outcome = await _finalize(
          context,
          message,
          prepared,
          await _executePrepared(prepared, signal, _updates(call, emit)),
          config,
          signal,
        );
    }

    await _emitEnd(outcome, emit);
    messages.add(await _emitResult(outcome, emit));
    outcomes.add(outcome);
  }

  return (messages: messages, terminate: _terminates(outcomes));
}

Future<_Batch> _parallel(
  AgentContext context,
  AssistantMessage message,
  List<ToolCall> calls,
  AgentLoopConfig config,
  AbortSignal? signal,
  AgentEventSink emit,
) async {
  // Prepared one by one; then the prepared calls run at once, each ending (with its end event) as it finishes. The
  // results go into the transcript in the order of the calls.
  final entries = <Future<_Outcome> Function()>[];
  for (final call in calls) {
    await emit(ToolExecutionStart(call));
    switch (await _prepare(context, message, call, config, signal, context.tools)) {
      case _Immediate(:final result):
        final _Outcome outcome = (call: call, result: result, isError: true, durationMs: null);
        await _emitEnd(outcome, emit);
        entries.add(() async => outcome);
      case final _Prepared prepared:
        entries.add(() async {
          final _Outcome outcome;
          if (signal?.aborted ?? false) {
            outcome = (call: call, result: AgentToolResult.error('Operation aborted'), isError: true, durationMs: null);
          } else {
            outcome = await _finalize(
              context,
              message,
              prepared,
              await _executePrepared(prepared, signal, _updates(call, emit)),
              config,
              signal,
            );
          }

          await _emitEnd(outcome, emit);
          return outcome;
        });
    }
  }

  final outcomes = await Future.wait([for (final entry in entries) entry()]);
  final messages = <ToolResultMessage>[];
  for (final outcome in outcomes) {
    messages.add(await _emitResult(outcome, emit));
  }

  return (messages: messages, terminate: _terminates(outcomes));
}

bool _terminates(List<_Outcome> outcomes) => outcomes.isNotEmpty && outcomes.every((o) => o.result.terminate);

/// The outcome of preparing a call: an answer without running it, or a call ready to run.
sealed class _Preparation {
  const _Preparation();
}

/// A call answered with an error without running: aborted, unknown, invalid or blocked.
class _Immediate extends _Preparation {
  const _Immediate(this.result);

  final AgentToolResult result;
}

class _Prepared extends _Preparation {
  const _Prepared(this.call, this.tool, this.arguments);

  final ToolCall call;
  final AgentTool tool;
  final Object? arguments;
}

/// The text of a thrown error, as the model sees it.
String _errorText(Object error) {
  final text = '$error';
  return text.startsWith('Exception: ') ? text.substring('Exception: '.length) : text;
}

/// Finds the call's tool, prepares and validates its arguments, and runs `beforeToolCall`.
Future<_Preparation> _prepare(
  AgentContext context,
  AssistantMessage message,
  ToolCall call,
  ToolHooks hooks,
  AbortSignal? signal,
  List<AgentTool> tools,
) async {
  // Once aborted, the call is answered without preparation or hooks, so no call is left without a result.
  if (signal?.aborted ?? false) {
    return _Immediate(AgentToolResult.error('Operation aborted'));
  }

  final tool = tools.where((t) => t.name == call.name).firstOrNull;
  if (tool == null) {
    return _Immediate(AgentToolResult.error('Tool ${call.name} not found'));
  }

  try {
    if (call.argumentsError != null) {
      throw 'Invalid arguments for tool "${call.name}": ${call.argumentsError}';
    }

    // A copy, so a tool that changes its arguments never changes the call in the transcript.
    final raw = jsonDecode(jsonEncode(tool.prepareArguments?.call(call.arguments) ?? call.arguments)) as Json;
    final arguments = tool.validate == null ? raw : tool.validate!(raw);
    final before = await hooks.beforeToolCall?.call(ToolCallContext(message, call, arguments, context), signal);
    if (signal?.aborted ?? false) {
      return _Immediate(AgentToolResult.error('Operation aborted'));
    }

    if (before != null && before.block) {
      return _Immediate(
        AgentToolResult.error(before.reason ?? 'Tool execution was blocked')..terminate = before.terminate,
      );
    }

    return _Prepared(call, tool, arguments);
  } catch (error) {
    return _Immediate(AgentToolResult.error(_errorText(error)));
  }
}

typedef _Executed = ({AgentToolResult result, bool isError, int durationMs});

/// Runs a prepared call; a thrown error becomes an error result. Updates sent after it finished are dropped.
Future<_Executed> _executePrepared(
  _Prepared prepared,
  AbortSignal? signal,
  FutureOr<void> Function(AgentToolResult) onUpdate,
) async {
  final updates = <Future<void>>[];
  var accepting = true;
  final clock = Stopwatch()..start();
  try {
    final result = await prepared.tool.execute(prepared.call.id, prepared.arguments, signal, (partial) {
      if (accepting) {
        updates.add(Future.sync(() => onUpdate(partial)));
      }
    });
    accepting = false;
    await Future.wait(updates);
    return (result: result, isError: result.isError, durationMs: clock.elapsedMilliseconds);
  } catch (error) {
    final durationMs = clock.elapsedMilliseconds;
    accepting = false;
    await Future.wait(updates);
    return (result: AgentToolResult.error(_errorText(error)), isError: true, durationMs: durationMs);
  } finally {
    accepting = false;
  }
}

/// Runs `afterToolCall` on an executed call; each field it sets replaces the result's. A thrown error becomes an
/// error result.
Future<_Outcome> _finalize(
  AgentContext context,
  AssistantMessage message,
  _Prepared prepared,
  _Executed executed,
  ToolHooks hooks,
  AbortSignal? signal,
) async {
  var result = executed.result;
  var isError = executed.isError;
  final after = hooks.afterToolCall;
  if (after != null) {
    try {
      final change = await after(
        ToolCallContext(message, prepared.call, prepared.arguments, context),
        result,
        isError,
        signal,
      );
      if (change != null) {
        result = AgentToolResult(
          change.content ?? result.content,
          details: change.details ?? result.details,
          usage: change.usage ?? result.usage,
          isError: result.isError,
          terminate: change.terminate ?? result.terminate,
        );
        isError = change.isError ?? isError;
      }
    } catch (error) {
      result = AgentToolResult.error(_errorText(error));
      isError = true;
    }
  }

  return (call: prepared.call, result: result, isError: isError, durationMs: executed.durationMs);
}

FutureOr<void> Function(AgentToolResult) _updates(ToolCall call, AgentEventSink emit) {
  return (partial) => emit(ToolExecutionUpdate(call, partial));
}

Future<void> _emitEnd(_Outcome outcome, AgentEventSink emit) async {
  await emit(ToolExecutionEnd(outcome.call, outcome.result, outcome.isError, outcome.durationMs));
}

Future<ToolResultMessage> _emitResult(_Outcome outcome, AgentEventSink emit) async {
  final message = ToolResultMessage(
    id: outcome.call.id,
    name: outcome.call.name,
    content: outcome.result.content,
    details: outcome.result.details,
    usage: outcome.result.usage,
    isError: outcome.isError,
    timestamp: DateTime.now().millisecondsSinceEpoch,
    durationMs: outcome.durationMs,
  );
  await emit(MessageStart(message));
  await emit(MessageEnd(message));
  return message;
}

/// The outcome of one tool call run outside the loop.
class ToolCallOutcome {
  const ToolCallOutcome(this.call, this.result, this.isError, [this.durationMs]);

  final ToolCall call;
  final AgentToolResult result;
  final bool isError;
  final int? durationMs;
}

/// Runs one tool call through the steps a model's call takes: argument preparation, validation, `beforeToolCall`,
/// execution and `afterToolCall`, with no events and no messages. Tools that call other tools use it, so the hooks
/// (permission checks, say) apply to those calls too. Failures come back as error outcomes, never thrown.
Future<ToolCallOutcome> runToolCall(
  ToolCall call, {
  required List<AgentTool> tools,
  required AssistantMessage message,
  required AgentContext context,
  required ToolHooks hooks,
  AbortSignal? signal,
  FutureOr<void> Function(AgentToolResult partial)? onUpdate,
}) async {
  switch (await _prepare(context, message, call, hooks, signal, tools)) {
    case _Immediate(:final result):
      return ToolCallOutcome(call, result, true);
    case final _Prepared prepared:
      final outcome = await _finalize(
        context,
        message,
        prepared,
        await _executePrepared(prepared, signal, onUpdate ?? (_) {}),
        hooks,
        signal,
      );
      return ToolCallOutcome(call, outcome.result, outcome.isError, outcome.durationMs);
  }
}
