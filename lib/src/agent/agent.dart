/// The stateful agent, ported from pi-agent-core's `agent.ts`: it owns the transcript, runs the loop, emits its
/// events to subscribers, and queues steering and follow-up messages.
library;

import 'dart:async';

import '../ai/model.dart';
import '../ai/options.dart';
import '../ai/transcript.dart';
import '../ai/types.dart';
import 'loop.dart';
import 'types.dart';

/// How many queued messages one drain takes: all of them, or the oldest only.
enum QueueMode { all, oneAtATime }

/// Receives the agent's events with the run's abort signal; the agent waits for each listener in turn.
typedef AgentListener = FutureOr<void> Function(AgentEvent event, AbortSignal signal);

class _Queue {
  _Queue(this.mode);

  QueueMode mode;
  var _messages = <AgentMessage>[];

  bool get isNotEmpty => _messages.isNotEmpty;

  void add(AgentMessage message) => _messages.add(message);

  /// The messages the next drain takes.
  List<AgentMessage> peek() {
    if (mode == QueueMode.all) {
      return [..._messages];
    }

    return _messages.take(1).toList();
  }

  List<AgentMessage> drain() {
    final taken = peek();
    _messages = _messages.sublist(taken.length);
    return taken;
  }

  void clear() => _messages = [];
}

const _unknown = Model(
  id: 'unknown',
  name: 'unknown',
  api: 'unknown',
  provider: 'unknown',
  baseUrl: '',
  input: [],
  cost: ModelCost(input: 0, output: 0, cacheRead: 0, cacheWrite: 0),
  contextWindow: 0,
  maxTokens: 0,
);

List<Message> _llmMessages(List<AgentMessage> messages) => messages.whereType<Message>().toList();

/// A conversation with a model: its transcript, tools and settings, and the run in progress.
class Agent {
  /// [systemPrompt] and [tools] become the leading system message, unless [messages] already starts with one.
  Agent({
    required this.stream,
    this.model = _unknown,
    this.level = ThinkingLevel.off,
    String? systemPrompt,
    List<AgentTool> tools = const [],
    List<AgentMessage> messages = const [],
    this.convertToLlm = _llmMessages,
    this.transformContext,
    this.getApiKey,
    this.options = const SimpleStreamOptions(),
    this.beforeToolCall,
    this.afterToolCall,
    this.finishTurn,
    this.prepareRequest,
    this.prepareNextTurn,
    QueueMode steeringMode = QueueMode.oneAtATime,
    QueueMode followUpMode = QueueMode.oneAtATime,
    this.toolExecution = ToolExecution.parallel,
  }) : _tools = [...tools],
       _messages = [...messages],
       _steering = _Queue(steeringMode),
       _followUp = _Queue(followUpMode) {
    if (_messages.firstOrNull is SystemMessage) {
      return;
    }

    if ((systemPrompt?.isNotEmpty ?? false) || tools.isNotEmpty) {
      _messages.insert(
        0,
        SystemMessage(
          [if (systemPrompt?.isNotEmpty ?? false) TextContent(systemPrompt!)],
          added: tools.isEmpty ? null : [for (final t in tools) Tool(t.name, t.description, t.parameters)],
          timestamp: 0,
        ),
      );
    }
  }

  /// Streams one model response.
  StreamFn stream;

  /// The model of future turns.
  Model model;

  /// The thinking level of future turns.
  ThinkingLevel level;
  List<AgentTool> _tools;
  List<AgentMessage> _messages;
  FutureOr<List<Message>> Function(List<AgentMessage> messages) convertToLlm;
  Future<List<AgentMessage>> Function(List<AgentMessage> messages, AbortSignal? signal)? transformContext;
  FutureOr<String?> Function(String provider)? getApiKey;

  /// Request options: session id, thinking budgets, retry limits, payload and response hooks...
  SimpleStreamOptions options;
  Future<BeforeToolCall?> Function(ToolCallContext call, AbortSignal? signal)? beforeToolCall;
  Future<AfterToolCall?> Function(ToolCallContext call, AgentToolResult result, bool isError, AbortSignal? signal)?
  afterToolCall;
  FutureOr<TurnDecision?> Function(AgentTurn turn, AbortSignal? signal)? finishTurn;
  FutureOr<TurnUpdate?> Function(AgentContext context, Model model, ThinkingLevel level, AbortSignal? signal)?
  prepareRequest;
  FutureOr<TurnUpdate?> Function(AgentTurn turn, AbortSignal? signal)? prepareNextTurn;
  ToolExecution toolExecution;

  final _listeners = <AgentListener>[];
  final _Queue _steering;
  final _Queue _followUp;
  AbortSignal? _signal;
  Completer<void>? _run;
  AgentMessage? _streaming;
  var _pending = <String>{};
  String? _error;

  /// The executable tools; differences from the tools the transcript declares are announced before the next request.
  List<AgentTool> get tools => _tools;

  /// Replaced by a copy of [tools].
  set tools(List<AgentTool> tools) => _tools = [...tools];

  /// The transcript; its system messages carry the prompt and the tool declarations.
  List<AgentMessage> get messages => _messages;

  /// Replaced by a copy of [messages].
  set messages(List<AgentMessage> messages) => _messages = [...messages];

  /// The current system prompt, replayed from the transcript. To change it, add a system message.
  String get systemPrompt {
    final system = currentSystemMessage(_messages.whereType<Message>().toList());
    return system == null ? '' : systemText(system);
  }

  /// Whether a run is in progress; it stays true until the listeners of its `AgentEnd` finish.
  bool get isStreaming => _run != null;

  /// The message being streamed, if any.
  AgentMessage? get streamingMessage => _streaming;

  /// The ids of the tool calls running now.
  Set<String> get pendingToolCalls => _pending;

  /// The error of the latest failed or aborted turn.
  String? get errorMessage => _error;

  /// The abort signal of the run in progress.
  AbortSignal? get signal => _signal;

  QueueMode get steeringMode => _steering.mode;

  set steeringMode(QueueMode mode) => _steering.mode = mode;

  QueueMode get followUpMode => _followUp.mode;

  set followUpMode(QueueMode mode) => _followUp.mode = mode;

  /// Adds [listener]; returns a function that removes it. Listeners run in the order added, and a run ends only
  /// after they finish.
  void Function() subscribe(AgentListener listener) {
    _listeners.add(listener);
    return () => _listeners.remove(listener);
  }

  /// Queues [message] for after the current assistant turn.
  void steer(AgentMessage message) => _steering.add(message);

  /// Queues [message] for when the agent would otherwise stop.
  void followUp(AgentMessage message) => _followUp.add(message);

  void clearSteeringQueue() => _steering.clear();

  void clearFollowUpQueue() => _followUp.clear();

  void clearAllQueues() {
    _steering.clear();
    _followUp.clear();
  }

  bool get hasQueuedMessages => _steering.isNotEmpty || _followUp.isNotEmpty;

  /// The queued messages the next turn would take, without taking them.
  List<AgentMessage> peekQueuedMessages() {
    final steering = _steering.peek();
    return steering.isNotEmpty ? steering : _followUp.peek();
  }

  /// Aborts the run in progress, if any.
  void abort() => _signal?.abort();

  /// Completes when the run in progress and its listeners have finished.
  Future<void> waitForIdle() => _run?.future ?? Future.value();

  /// Clears the conversation and the queues, keeping the replayed system prompt and tools.
  void reset() {
    if (_run != null) {
      throw StateError('Agent is already processing. Wait for completion before resetting.');
    }

    _messages = [?currentSystemMessage(_messages.whereType<Message>().toList())];
    _streaming = null;
    _pending = {};
    _error = null;
    clearAllQueues();
  }

  /// Starts a run with [messages].
  Future<void> prompt(List<AgentMessage> messages) async {
    if (_run != null) {
      throw StateError(
        'Agent is already processing a prompt. Use steer() or followUp() to queue messages, or wait for completion.',
      );
    }

    await _runPrompt(messages);
  }

  /// Starts a run with a user message of [text] and [images].
  Future<void> promptText(String text, [List<ImageContent> images = const []]) {
    return prompt([
      UserMessage([TextContent(text), ...images], timestamp: DateTime.now().millisecondsSinceEpoch),
    ]);
  }

  /// Continues from the transcript (pi's `continue()`, a reserved word in Dart). After an assistant message it takes
  /// queued steering, else queued follow-ups; otherwise the last message must be a user message or a tool result.
  Future<void> proceed() async {
    if (_run != null) {
      throw StateError('Agent is already processing. Wait for completion before continuing.');
    }

    if (_messages.every((m) => m is SystemMessage)) {
      throw StateError('No messages to continue from');
    }

    if (_messages.last is! AssistantMessage) {
      await _lifecycle((signal) => runAgentLoopContinue(_snapshot(), _config(), _emit, stream, signal));
      return;
    }

    final steering = _steering.drain();
    if (steering.isNotEmpty) {
      await _runPrompt(steering, skipSteering: true);
      return;
    }

    final followUps = _followUp.drain();
    if (followUps.isNotEmpty) {
      await _runPrompt(followUps);
      return;
    }

    throw StateError('Cannot continue from message role: assistant');
  }

  Future<void> _runPrompt(List<AgentMessage> messages, {bool skipSteering = false}) {
    return _lifecycle((signal) {
      return runAgentLoop(messages, _snapshot(), _config(skipSteering: skipSteering), _emit, stream, signal);
    });
  }

  AgentContext _snapshot() => AgentContext([..._messages], tools: [..._tools]);

  AgentLoopConfig _config({bool skipSteering = false}) {
    var skip = skipSteering;
    final next = prepareNextTurn;
    return AgentLoopConfig(
      model: model,
      level: level,
      options: options,
      convertToLlm: convertToLlm,
      transformContext: transformContext,
      getApiKey: getApiKey,
      toolExecution: toolExecution,
      beforeToolCall: beforeToolCall,
      afterToolCall: afterToolCall,
      finishTurn: finishTurn,
      prepareRequest: prepareRequest,
      prepareNextTurn: next == null ? null : (turn) => next(turn, _signal),
      getSteeringMessages: () async {
        if (skip) {
          skip = false;
          return [];
        }

        return _steering.drain();
      },
      getFollowUpMessages: () async => _followUp.drain(),
    );
  }

  Future<void> _lifecycle(Future<void> Function(AbortSignal signal) run) async {
    if (_run != null) {
      throw StateError('Agent is already processing.');
    }

    final signal = _signal = AbortSignal();
    _run = Completer<void>();
    _streaming = null;
    _error = null;
    try {
      await run(signal);
    } catch (error) {
      await _fail(error, signal.aborted);
    } finally {
      _streaming = null;
      _pending = {};
      _run!.complete();
      _run = null;
      _signal = null;
    }
  }

  /// A failure the loop threw: reported as an empty assistant message with the error, with its run's last events.
  Future<void> _fail(Object error, bool aborted) async {
    final message = AssistantMessage(
      content: [TextContent('')],
      api: model.api,
      provider: model.provider,
      model: model.id,
      stopReason: aborted ? StopReason.aborted : StopReason.error,
      errorMessage: errorText(error),
      timestamp: DateTime.now().millisecondsSinceEpoch,
    );
    await _emit(MessageStart(message));
    await _emit(MessageEnd(message));
    await _emit(TurnEnd(message, const []));
    await _emit(AgentEnd([message]));
  }

  /// Updates the state for [event], then waits for each listener.
  Future<void> _emit(AgentEvent event) async {
    switch (event) {
      case MessageStart(:final message):
        _streaming = message;
      case MessageUpdate(:final message):
        _streaming = message;
      case MessageEnd(:final message):
        _streaming = null;
        _messages.add(message);
      case ToolExecutionStart(:final call):
        _pending = {..._pending, call.id};
      case ToolExecutionEnd(:final call):
        _pending = {..._pending}..remove(call.id);
      case TurnEnd(:final message) when message.errorMessage != null:
        _error = message.errorMessage;
      case AgentEnd():
        _streaming = null;
      default:
        break;
    }

    final signal = _signal;
    if (signal == null) {
      throw StateError('Agent listener invoked outside active run');
    }

    for (final listener in [..._listeners]) {
      await listener(event, signal);
    }
  }
}
