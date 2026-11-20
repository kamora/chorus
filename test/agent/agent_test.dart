import 'dart:async';

import 'package:chorus/chorus.dart';
import 'package:test/test.dart';

// Ported from pi's packages/agent/test/agent.test.ts. Left out: the default stream function and request identity
// (chorus has neither). `continue()` is `proceed()` here; pi's state object is the agent itself.

AssistantMessage assistant(String text) {
  return AssistantMessage(
    content: [TextContent(text)],
    api: 'openai-responses',
    provider: 'openai',
    model: 'mock',
    stopReason: StopReason.stop,
    timestamp: DateTime.now().millisecondsSinceEpoch,
  );
}

AssistantMessage toolUse(List<ToolCall> calls) {
  return AssistantMessage(
    content: calls,
    api: 'openai-responses',
    provider: 'openai',
    model: 'mock',
    stopReason: StopReason.toolUse,
    timestamp: DateTime.now().millisecondsSinceEpoch,
  );
}

UserMessage user(String text) => UserMessage([TextContent(text)], timestamp: DateTime.now().millisecondsSinceEpoch);

AgentTool tool(String name, [Future<AgentToolResult> Function(void Function(AgentToolResult) onUpdate)? run]) {
  return AgentTool(
    name: name,
    description: '$name tool',
    parameters: const {'type': 'object'},
    execute: (id, args, signal, onUpdate) => run?.call(onUpdate) ?? Future.value(AgentToolResult([TextContent(name)])),
  );
}

AssistantMessageEventStream unused(Model model, List<Message> messages, SimpleStreamOptions options) {
  throw StateError('Unexpected stream call');
}

/// Answers each request with `reply(n, messages)`, a microtask later.
StreamFn replies(AssistantMessage Function(int n, List<Message> messages) reply) {
  var n = 0;
  return (model, messages, options) {
    final stream = AssistantMessageEventStream();
    final message = reply(n++, messages);
    scheduleMicrotask(() {
      if (message.stopReason == StopReason.error || message.stopReason == StopReason.aborted) {
        stream.push(ErrorEvent(message.stopReason, message));
      } else {
        stream.push(DoneEvent(message.stopReason, message));
      }
    });
    return stream;
  };
}

/// A response that starts and then waits until the request is aborted.
AssistantMessageEventStream untilAborted(Model model, List<Message> messages, SimpleStreamOptions options) {
  final stream = AssistantMessageEventStream();
  scheduleMicrotask(() async {
    stream.push(StartEvent(assistant('')));
    await options.signal!.future;
    stream.push(ErrorEvent(StopReason.aborted, assistant('Aborted')..stopReason = StopReason.aborted));
  });
  return stream;
}

/// The plain user texts of [messages].
List<String> users(List<Message> messages) {
  return [
    for (final m in messages)
      if (m is UserMessage)
        for (final c in m.content.whereType<TextContent>()) c.text,
  ];
}

String role(Object? message) {
  return switch (message) {
    SystemMessage() => 'system',
    UserMessage() => 'user',
    AssistantMessage() => 'assistant',
    ToolResultMessage() => 'toolResult',
    _ => 'custom',
  };
}

Future<void> tick([int ms = 10]) => Future<void>.delayed(Duration(milliseconds: ms));

void main() {
  group('Agent', () {
    test('should create an agent instance with default state', () {
      final agent = Agent(stream: unused);
      expect(agent.model, isNotNull);
      expect(agent.level, ThinkingLevel.off);
      expect(agent.tools, isEmpty);
      expect(agent.messages, isEmpty);
      expect(agent.isStreaming, isFalse);
      expect(agent.streamingMessage, isNull);
      expect(agent.pendingToolCalls, isEmpty);
      expect(agent.errorMessage, isNull);
    });

    test('should create an agent instance with custom initial state', () {
      const model = Model(
        id: 'gpt-4o-mini',
        name: 'GPT-4o mini',
        api: 'openai-responses',
        provider: 'openai',
        baseUrl: 'https://api.openai.com/v1',
        cost: ModelCost(input: 0.15, output: 0.6, cacheRead: 0.075, cacheWrite: 0),
        contextWindow: 128000,
        maxTokens: 16384,
      );
      final agent = Agent(
        stream: unused,
        systemPrompt: 'You are a helpful assistant.',
        model: model,
        level: ThinkingLevel.low,
      );

      expect(agent.messages.map((m) => (m as Message).toJson()), [
        {
          'role': 'system',
          'content': [
            {'type': 'text', 'text': 'You are a helpful assistant.'},
          ],
          'timestamp': 0,
        },
      ]);
      expect(agent.model, same(model));
      expect(agent.level, ThinkingLevel.low);
    });

    test('converts initial prompt and tools into transcript state', () {
      final initial =
          Agent(stream: unused, systemPrompt: 'You are helpful.', tools: [tool('echo')]).messages.first
              as SystemMessage;
      expect(initial.content.single.text, 'You are helpful.');
      expect(initial.added!.map((t) => t.name), ['echo']);
      expect(initial.added!.single, isNot(isA<AgentTool>()));
    });

    test('keeps a leading system message of the initial messages instead of adding one', () {
      final system = SystemMessage([TextContent('Given prompt.')], timestamp: 1);
      final agent = Agent(
        stream: unused,
        systemPrompt: 'Ignored.',
        tools: [tool('echo')],
        messages: [system, user('hi')],
      );
      expect(agent.messages, hasLength(2));
      expect(agent.messages.first, same(system));
    });

    test('declares tool loadout changes to the model before the next request', () async {
      final requests = <List<String>>[];
      final agent = Agent(
        systemPrompt: 'You are helpful.',
        tools: [tool('first')],
        stream: (model, messages, options) {
          requests.add([
            for (final m in messages.whereType<SystemMessage>()) ...[
              '+${(m.added ?? []).map((t) => t.name).join(',')}',
              '-${(m.removed ?? []).join(',')}',
            ],
          ]);
          return replies((_, _) => assistant('done'))(model, messages, options);
        },
      );

      await agent.promptText('one');
      agent.tools = [tool('second')];
      await agent.promptText('two');
      await agent.promptText('three');

      expect(requests, [
        ['+first', '-'],
        ['+first', '-', '+second', '-first'],
        ['+first', '-', '+second', '-first'],
      ]);
      final update = agent.messages.whereType<SystemMessage>().firstWhere((m) => m.removed != null);
      expect(update.toJson(), {
        'role': 'system',
        'content': <Object?>[],
        'toolsAdded': [
          {
            'name': 'second',
            'description': 'second tool',
            'parameters': {'type': 'object'},
          },
        ],
        'toolsRemoved': [
          {'name': 'first'},
        ],
        'timestamp': update.timestamp,
      });
    });

    test('merges tool changes into a pending system message', () async {
      final agent = Agent(
        systemPrompt: 'You are helpful.',
        stream: (model, messages, options) {
          expect(messages.whereType<SystemMessage>(), hasLength(2));
          return replies((_, _) => assistant('done'))(model, messages, options);
        },
      );

      agent.tools = [tool('echo')];
      await agent.prompt([
        SystemMessage(const [], sections: const {'skills': '<skills>x</skills>'}, timestamp: 1),
        UserMessage([TextContent('hi')], timestamp: 2),
      ]);

      expect((agent.messages[1] as SystemMessage).toJson(), {
        'role': 'system',
        'content': <Object?>[],
        'sections': {'skills': '<skills>x</skills>'},
        'toolsAdded': [
          {
            'name': 'echo',
            'description': 'echo tool',
            'parameters': {'type': 'object'},
          },
        ],
        'timestamp': 1,
      });
    });

    test('rewrites pending tool declarations to match the executable set', () async {
      final agent = Agent(
        systemPrompt: 'You are helpful.',
        tools: [tool('first')],
        stream: replies((_, _) => assistant('done')),
      );

      // The pending message claims to add `second` and remove `first`, but the executable set still has `first`
      // and lacks `second`: the executable set wins.
      await agent.prompt([
        SystemMessage(
          const [],
          sections: const {'note': '<note>x</note>'},
          added: [tool('second')],
          removed: const ['first'],
          timestamp: 1,
        ),
        UserMessage([TextContent('hi')], timestamp: 2),
      ]);

      expect((agent.messages[1] as SystemMessage).toJson(), {
        'role': 'system',
        'content': <Object?>[],
        'sections': {'note': '<note>x</note>'},
        'timestamp': 1,
      });
      expect(agent.messages.whereType<SystemMessage>().expand((m) => m.added ?? <Tool>[]).map((t) => t.name), [
        'first',
      ]);
    });

    test('restores the transcript baseline when reset', () {
      final agent = Agent(
        stream: unused,
        systemPrompt: 'You are helpful.',
        tools: [tool('echo')],
        messages: [
          UserMessage([TextContent('old')], timestamp: 1),
        ],
      );

      agent.reset();

      final initial = agent.messages.single as SystemMessage;
      expect(initial.content.single.text, 'You are helpful.');
      expect(initial.added!.map((t) => t.name), ['echo']);
    });

    test('should subscribe to events', () {
      final agent = Agent(stream: unused);
      var count = 0;
      final unsubscribe = agent.subscribe((_, _) => count++);
      agent.level = ThinkingLevel.low;
      expect(count, 0);
      unsubscribe();
      agent.level = ThinkingLevel.high;
      expect(count, 0);
    });

    test('emits full lifecycle events for thrown run failures', () async {
      final agent = Agent(stream: (_, _, _) => throw Exception('provider exploded'));
      final events = <Type>[];
      agent.subscribe((event, _) => events.add(event.runtimeType));

      await agent.promptText('hello');

      expect(events, [AgentStart, TurnStart, MessageStart, MessageEnd, MessageStart, MessageEnd, TurnEnd, AgentEnd]);
      final last = agent.messages.last as AssistantMessage;
      expect(last.stopReason, StopReason.error);
      expect(last.errorMessage, 'provider exploded');
      expect(agent.errorMessage, 'provider exploded');
    });

    test('should await async subscribers before prompt resolves', () async {
      final barrier = Completer<void>();
      final agent = Agent(stream: replies((_, _) => assistant('ok')));
      var finished = false;
      agent.subscribe((event, _) async {
        if (event is AgentEnd) {
          await barrier.future;
          finished = true;
        }
      });

      var resolved = false;
      final prompt = agent.promptText('hello').then((_) => resolved = true);
      await tick();
      expect((resolved, finished, agent.isStreaming), (false, false, true));

      barrier.complete();
      await prompt;
      expect((resolved, finished, agent.isStreaming), (true, true, false));
    });

    test('waitForIdle should wait for async subscribers', () async {
      final barrier = Completer<void>();
      final agent = Agent(stream: replies((_, _) => assistant('ok')));
      agent.subscribe((event, _) async {
        if (event is MessageEnd && event.message is AssistantMessage) {
          await barrier.future;
        }
      });

      final prompt = agent.promptText('hello');
      var idle = false;
      final waiting = agent.waitForIdle().then((_) => idle = true);
      await tick();
      expect((idle, agent.isStreaming), (false, true));

      barrier.complete();
      await Future.wait([prompt, waiting]);
      expect((idle, agent.isStreaming), (true, false));
    });

    test('should pass the active abort signal to subscribers', () async {
      AbortSignal? received;
      final agent = Agent(stream: untilAborted);
      agent.subscribe((event, signal) {
        if (event is AgentStart) {
          received = signal;
        }
      });

      final prompt = agent.promptText('hello');
      await tick();
      expect(received?.aborted, isFalse);

      agent.abort();
      await prompt;
      expect(received?.aborted, isTrue);
    });

    test('should ignore tool updates after the tool execution settles', () async {
      void Function(AgentToolResult)? late;
      final events = <AgentEvent>[];
      final agent = Agent(
        tools: [
          tool('delayed_tool', (onUpdate) async {
            late = onUpdate;
            onUpdate(AgentToolResult([TextContent('running')]));
            return AgentToolResult([TextContent('ok')], terminate: true);
          }),
        ],
        stream: replies((_, _) => toolUse([ToolCall('call-1', 'delayed_tool', {})])),
      );
      agent.subscribe((event, _) => events.add(event));

      await agent.promptText('run tool');
      final count = events.length;
      late!(AgentToolResult([TextContent('late')]));
      await tick(0);

      expect(events.whereType<ToolExecutionUpdate>(), hasLength(1));
      expect(events, hasLength(count));
    });

    test('should ignore a settled parallel tool update while another tool is still running', () async {
      final slowStarted = Completer<void>();
      final settledEnded = Completer<void>();
      final release = Completer<void>();
      void Function(AgentToolResult)? late;
      final events = <AgentEvent>[];
      final agent = Agent(
        tools: [
          tool('settled_tool', (onUpdate) async {
            late = onUpdate;
            return AgentToolResult([TextContent('done')], terminate: true);
          }),
          tool('slow_tool', (_) async {
            slowStarted.complete();
            await release.future;
            return AgentToolResult([TextContent('done')], terminate: true);
          }),
        ],
        stream: replies((_, _) {
          return toolUse([ToolCall('call-1', 'settled_tool', {}), ToolCall('call-2', 'slow_tool', {})]);
        }),
      );
      agent.subscribe((event, _) {
        events.add(event);
        if (event is ToolExecutionEnd && event.call.id == 'call-1') {
          settledEnded.complete();
        }
      });

      final prompt = agent.promptText('run tools');
      await Future.wait([slowStarted.future, settledEnded.future]);
      final count = events.length;
      late!(AgentToolResult([TextContent('late')]));
      await tick(0);
      expect(events, hasLength(count));

      release.complete();
      await prompt;
      expect(events.whereType<ToolExecutionUpdate>(), isEmpty);
    });

    test('should update state with mutators', () {
      final agent = Agent(stream: unused);
      agent.level = ThinkingLevel.high;
      expect(agent.level, ThinkingLevel.high);

      final tools = [tool('test')];
      agent.tools = tools;
      expect(agent.tools, tools);
      expect(agent.tools, isNot(same(tools)));

      final messages = <AgentMessage>[user('Hello')];
      agent.messages = messages;
      expect(agent.messages, messages);
      expect(agent.messages, isNot(same(messages)));

      final added = assistant('Hi');
      agent.messages.add(added);
      expect(agent.messages, hasLength(2));
      expect(agent.messages[1], same(added));

      agent.messages = [];
      expect(agent.messages, isEmpty);
    });

    test('should support steering and follow-up message queues', () {
      final agent = Agent(stream: unused);
      final steering = user('Steering message');
      final followUp = user('Follow-up message');
      agent.steer(steering);
      agent.followUp(followUp);
      expect(agent.messages, isNot(contains(steering)));
      expect(agent.messages, isNot(contains(followUp)));
      expect(agent.hasQueuedMessages, isTrue);
    });

    test('should handle abort controller', () {
      expect(() => Agent(stream: unused).abort(), returnsNormally);
    });

    test('should reject reset while processing without corrupting the transcript', () async {
      final started = Completer<void>();
      final release = Completer<void>();
      final agent = Agent(
        stream: (model, messages, options) {
          final stream = AssistantMessageEventStream();
          scheduleMicrotask(() async {
            stream.push(StartEvent(assistant('')));
            started.complete();
            await release.future;
            stream.push(DoneEvent(StopReason.stop, assistant('Done')));
          });
          return stream;
        },
      );

      final prompt = agent.promptText('Hello');
      await started.future;
      expect(agent.isStreaming, isTrue);
      expect(agent.messages.map(role), ['user']);
      expect(
        agent.reset,
        throwsA(
          isA<StateError>().having(
            (e) => e.message,
            'message',
            'Agent is already processing. Wait for completion before resetting.',
          ),
        ),
      );
      expect(agent.isStreaming, isTrue);
      expect(agent.messages.map(role), ['user']);

      release.complete();
      await prompt;
      expect(agent.isStreaming, isFalse);
      expect(agent.messages.map(role), ['user', 'assistant']);
    });

    test('should throw when prompt() called while streaming', () async {
      final agent = Agent(stream: untilAborted);
      final first = agent.promptText('First message');
      await tick();
      expect(agent.isStreaming, isTrue);
      await expectLater(
        agent.promptText('Second message'),
        throwsA(
          isA<StateError>().having(
            (e) => e.message,
            'message',
            'Agent is already processing a prompt. Use steer() or followUp() to queue messages, or wait for '
                'completion.',
          ),
        ),
      );

      agent.abort();
      await first;
    });

    test('should throw when proceed() called while streaming', () async {
      final agent = Agent(stream: untilAborted);
      final first = agent.promptText('First message');
      await tick();
      await expectLater(
        agent.proceed(),
        throwsA(
          isA<StateError>().having(
            (e) => e.message,
            'message',
            'Agent is already processing. Wait for completion before continuing.',
          ),
        ),
      );

      agent.abort();
      await first;
    });

    test('proceed() should process queued follow-up messages after an assistant turn', () async {
      final agent = Agent(stream: replies((_, _) => assistant('Processed')));
      agent.messages = [user('Initial'), assistant('Initial response')];
      agent.followUp(user('Queued follow-up'));

      await agent.proceed();

      expect(users(agent.messages.whereType<Message>().toList()), contains('Queued follow-up'));
      expect(agent.messages.last, isA<AssistantMessage>());
    });

    for (final (mode, expected) in [(QueueMode.oneAtATime, 2), (QueueMode.all, 1)]) {
      test('proceed() keeps ${mode.name} steering semantics for assistant-tail fallback', () async {
        final requests = <List<String>>[];
        final agent = Agent(
          steeringMode: mode,
          stream: replies((_, messages) {
            requests.add(users(messages));
            return assistant('Processed');
          }),
        );
        agent.messages = [user('Initial'), assistant('Initial response')];
        agent.steer(user('Steering 1'));
        agent.steer(user('Steering 2'));

        await agent.proceed();

        expect(requests, hasLength(expected));
        expect(requests[0], contains('Steering 1'));
        if (mode == QueueMode.oneAtATime) {
          expect(requests[0], isNot(contains('Steering 2')));
          expect(requests[1], contains('Steering 2'));
        } else {
          expect(requests[0], contains('Steering 2'));
        }
      });
    }

    test('passes the active abort signal to prepareNextTurn', () async {
      var requests = 0;
      AbortSignal? seen;

      await Agent(
        tools: [tool('noop')],
        prepareNextTurn: (turn, signal) async {
          seen = signal;
          return null;
        },
        stream: replies((n, _) {
          requests++;
          return n == 0 ? toolUse([ToolCall('tool-1', 'noop', {})]) : assistant('done');
        }),
      ).promptText('start');
      expect(requests, 2);
      expect(seen, isNotNull);
    });

    test('forwards finishTurn with the active abort signal', () async {
      var requests = 0;
      AbortSignal? seen;
      List<String>? roles;

      await Agent(
        tools: [tool('noop')],
        finishTurn: (turn, signal) {
          seen = signal;
          roles = turn.context.messages.map(role).toList();
          return TurnDecision.end;
        },
        stream: replies((n, _) {
          requests++;
          return n == 0 ? toolUse([ToolCall('tool-1', 'noop', {})]) : assistant('should not run');
        }),
      ).promptText('start');
      expect(requests, 1);
      expect(seen, isNotNull);
      expect(roles, ['system', 'user', 'assistant', 'toolResult']);
    });

    for (final (name, messages) in [
      ('empty', <AgentMessage>[]),
      (
        'system-only',
        <AgentMessage>[
          SystemMessage([TextContent('system only')], timestamp: 1),
        ],
      ),
    ]) {
      test('rejects a queued continuation from $name context without draining queues', () async {
        final agent = Agent(stream: unused, messages: messages);
        final steering = user('steering');
        final followUp = user('follow-up');
        agent.steer(steering);
        agent.followUp(followUp);

        await expectLater(
          agent.proceed(),
          throwsA(isA<StateError>().having((e) => e.message, 'message', 'No messages to continue from')),
        );
        expect(agent.peekQueuedMessages(), [steering]);
        agent.clearSteeringQueue();
        expect(agent.peekQueuedMessages(), [followUp]);
      });
    }

    for (final (name, messages) in [
      ('user', <AgentMessage>[user('existing user')]),
      (
        'toolResult',
        <AgentMessage>[
          user('existing user'),
          toolUse([ToolCall('call-1', 'noop', {})]),
          ToolResultMessage(id: 'call-1', name: 'noop', content: [TextContent('done')], isError: false, timestamp: 1),
        ],
      ),
    ]) {
      test('defers follow-up input on the first continuation request from a $name tail', () async {
        final requests = <List<String>>[];
        final agent = Agent(
          messages: messages,
          stream: replies((_, messages) {
            requests.add(users(messages));
            return assistant('done');
          }),
        );
        agent.followUp(user('follow-up'));

        await agent.proceed();

        expect(requests, hasLength(2));
        expect(requests[0], isNot(contains('follow-up')));
        expect(requests[1], contains('follow-up'));
      });
    }

    for (final (mode, expected) in [(QueueMode.oneAtATime, 2), (QueueMode.all, 1)]) {
      test('polls ${mode.name} steering at continuation startup', () async {
        final requests = <List<String>>[];
        final agent = Agent(
          messages: [user('existing')],
          steeringMode: mode,
          stream: replies((_, messages) {
            requests.add(users(messages));
            return assistant('done');
          }),
        );
        agent.steer(user('first'));
        agent.steer(user('second'));

        await agent.proceed();

        expect(requests, hasLength(expected));
        expect(requests[0], contains('first'));
        if (mode == QueueMode.oneAtATime) {
          expect(requests[0], isNot(contains('second')));
          expect(requests[1], contains('second'));
        } else {
          expect(requests[0], contains('second'));
        }
      });
    }

    test('keeps steering ahead of follow-up from a non-assistant continuation tail', () async {
      final requests = <List<String>>[];
      final agent = Agent(
        messages: [user('existing')],
        stream: replies((_, messages) {
          requests.add(users(messages));
          return assistant('done');
        }),
      );
      agent.steer(user('steering'));
      agent.followUp(user('follow-up'));

      await agent.proceed();

      expect(requests, hasLength(2));
      expect(requests[0], contains('steering'));
      expect(requests[0], isNot(contains('follow-up')));
      expect(requests[1], contains('follow-up'));
    });

    for (final reason in [StopReason.error, StopReason.aborted]) {
      test('keeps queues on a ${reason.name} response even when finishTurn requests continuation', () async {
        final steering = user('steering');
        final followUp = user('follow-up');
        final agent = Agent(
          finishTurn: (_, _) => TurnDecision.proceed,
          stream: replies((_, _) {
            return assistant(reason.name)
              ..stopReason = reason
              ..errorMessage = reason.name;
          }),
        );
        agent.followUp(followUp);
        agent.subscribe((event, _) {
          if (event is MessageEnd && event.message is AssistantMessage) {
            agent.steer(steering);
          }
        });

        await agent.promptText('start');

        expect(agent.peekQueuedMessages(), [steering]);
        agent.clearSteeringQueue();
        expect(agent.peekQueuedMessages(), [followUp]);
      });
    }

    test('keeps queues when finishTurn ends the run', () async {
      final steering = user('steering');
      final followUp = user('follow-up');
      final agent = Agent(finishTurn: (_, _) => TurnDecision.end, stream: replies((_, _) => assistant('done')));
      agent.followUp(followUp);
      agent.subscribe((event, _) {
        if (event is MessageEnd && event.message is AssistantMessage) {
          agent.steer(steering);
        }
      });

      await agent.promptText('start');

      expect(agent.peekQueuedMessages(), [steering]);
      agent.clearSteeringQueue();
      expect(agent.peekQueuedMessages(), [followUp]);
    });

    test('previews the next selected queued messages without consuming them', () {
      final agent = Agent(stream: unused, followUpMode: QueueMode.all);
      final first = user('first steering');
      final followUp = user('follow-up');
      agent.steer(first);
      agent.steer(user('second steering'));
      agent.followUp(followUp);

      expect(agent.peekQueuedMessages(), [first]);
      expect(agent.peekQueuedMessages(), [first]);
      agent.clearSteeringQueue();
      expect(agent.peekQueuedMessages(), [followUp]);
    });

    test('forwards provider stream event observers through the options', () async {
      final seen = <Json>[];

      await Agent(
        options: SimpleStreamOptions(onEvent: (event, _) => seen.add(event)),
        stream: (model, messages, options) {
          final stream = AssistantMessageEventStream();
          scheduleMicrotask(() async {
            await options.onEvent?.call({'request_cost': 0.01}, model);
            stream.push(DoneEvent(StopReason.stop, assistant('ok')));
          });
          return stream;
        },
      ).promptText('hello');
      expect(seen, [
        {'request_cost': 0.01},
      ]);
    });

    test('forwards sessionId to the stream function', () async {
      String? received;
      final agent = Agent(
        options: const SimpleStreamOptions(sessionId: 'session-abc'),
        stream: (model, messages, options) {
          received = options.sessionId;
          return replies((_, _) => assistant('ok'))(model, messages, options);
        },
      );

      await agent.promptText('hello');
      expect(received, 'session-abc');

      agent.options = const SimpleStreamOptions(sessionId: 'session-def');
      await agent.promptText('hello again');
      expect(received, 'session-def');
    });
  });
}
