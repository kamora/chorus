import 'dart:async';

import 'package:chorus/chorus.dart';
import 'package:test/test.dart';

// Ported from pi's packages/agent/test/agent-loop.test.ts. Left out: the default stream function (chorus has none)
// and structured tool content (not ported). pi validates tool arguments with pi-schema; here a tool's `validate`
// hook stands in where a test needs validation.

const model = Model(
  id: 'mock',
  name: 'mock',
  api: 'openai-responses',
  provider: 'openai',
  baseUrl: 'https://example.invalid',
  cost: ModelCost(input: 0, output: 0, cacheRead: 0, cacheWrite: 0),
  contextWindow: 8192,
  maxTokens: 2048,
);

AssistantMessage assistant(List<Content> content, [StopReason stop = StopReason.stop]) {
  return AssistantMessage(
    content: content,
    api: 'openai-responses',
    provider: 'openai',
    model: 'mock',
    stopReason: stop,
    timestamp: DateTime.now().millisecondsSinceEpoch,
  );
}

AssistantMessage calls(List<(String, String, Json)> calls) {
  return assistant([for (final (id, name, args) in calls) ToolCall(id, name, args)], StopReason.toolUse);
}

UserMessage user(String text) => UserMessage([TextContent(text)], timestamp: DateTime.now().millisecondsSinceEpoch);

/// The text of a user message, or `null`.
String? userText(Object? message) {
  if (message is! UserMessage) {
    return null;
  }

  return (message.content.single as TextContent).text;
}

List<Message> identity(List<AgentMessage> messages) => messages.whereType<Message>().toList();

/// A stream function answering request number `n` (from 0) with `reply(n, messages)`, a microtask later.
StreamFn scripted(AssistantMessage Function(int n, List<Message> messages) reply) {
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

/// A first request that calls tools, then plain answers.
StreamFn callsThenDone(List<(String, String, Json)> first) {
  return scripted((n, _) => n == 0 ? calls(first) : assistant([TextContent('done')]));
}

const _schema = {
  'type': 'object',
  'properties': {
    'value': {'type': 'string'},
  },
};

AgentTool tool(
  String name,
  Future<AgentToolResult> Function(Object? arguments) run, {
  ToolExecution? mode,
  Json Function(Json)? prepare,
  Object? Function(Json)? validate,
}) {
  return AgentTool(
    name: name,
    description: '$name tool',
    parameters: _schema,
    executionMode: mode,
    prepareArguments: prepare,
    validate: validate,
    execute: (id, arguments, signal, onUpdate) => run(arguments),
  );
}

String value(Object? arguments) => (arguments as Json)['value'] as String;

AgentToolResult text(String text, {bool terminate = false}) =>
    AgentToolResult([TextContent(text)], terminate: terminate);

Future<(List<AgentEvent>, List<AgentMessage>)> drain(AgentEventStream stream) async {
  return (await stream.toList(), await stream.result);
}

List<(String, String)> resultTexts(List<AgentMessage> messages) {
  return [for (final m in messages.whereType<ToolResultMessage>()) (m.id, (m.content.single as TextContent).text)];
}

List<String> toolIds<T extends AgentEvent>(List<AgentEvent> events) {
  return [
    for (final e in events.whereType<T>())
      switch (e) {
        ToolExecutionStart(:final call) || ToolExecutionEnd(:final call) => call.id,
        _ => '',
      },
  ];
}

List<String> roles(List<AgentMessage> messages) {
  return [
    for (final m in messages)
      switch (m) {
        SystemMessage() => 'system',
        UserMessage() => 'user',
        AssistantMessage() => 'assistant',
        ToolResultMessage() => 'toolResult',
        _ => 'custom',
      },
  ];
}

/// One turn whose assistant message calls [t] three times, then an aborted turn.
Future<(List<AgentEvent>, List<AgentMessage>)> runBatch(AgentTool t, AbortSignal signal, [ToolHooks? hooks]) {
  return drain(
    agentLoop(
      [user('run three')],
      AgentContext([], tools: [t]),
      AgentLoopConfig(
        model: model,
        convertToLlm: identity,
        beforeToolCall: hooks?.beforeToolCall,
        afterToolCall: hooks?.afterToolCall,
      ),
      scripted((n, _) {
        if (n > 0) {
          return assistant([], StopReason.aborted);
        }

        return calls([
          ('tool-1', t.name, {'value': 'first'}),
          ('tool-2', t.name, {'value': 'second'}),
          ('tool-3', t.name, {'value': 'third'}),
        ]);
      }),
      signal,
    ),
  );
}

void main() {
  group('agentLoop with AgentMessage', () {
    test('should emit events with AgentMessage types', () async {
      final (events, messages) = await drain(
        agentLoop(
          [user('Hello')],
          AgentContext([]),
          const AgentLoopConfig(model: model, convertToLlm: identity),
          scripted((_, _) => assistant([TextContent('Hi there!')])),
        ),
      );

      expect(roles(messages), ['user', 'assistant']);
      expect(events.first, isA<AgentStart>());
      expect(events[1], isA<TurnStart>());
      expect(events, contains(isA<MessageStart>()));
      expect(events, contains(isA<MessageEnd>()));
      expect(events, contains(isA<TurnEnd>()));
      expect(events.last, isA<AgentEnd>());
    });

    test('should build provider context exclusively from transcript messages', () async {
      final system = SystemMessage([TextContent('Transcript prompt')], timestamp: 1);
      Object? first;
      await agentLoop(
        [system, user('Hello')],
        AgentContext([]),
        const AgentLoopConfig(model: model, convertToLlm: identity),
        scripted((_, messages) {
          first = messages.first;
          return assistant([TextContent('done')]);
        }),
      ).result;

      expect(first, same(system));
    });

    test('should handle custom message types via convertToLlm', () async {
      var converted = <Message>[];
      await agentLoop(
        [user('Hello')],
        AgentContext(['a notification']),
        AgentLoopConfig(model: model, convertToLlm: (messages) => converted = identity(messages)),
        scripted((_, _) => assistant([TextContent('Response')])),
      ).result;

      expect(converted.map(userText), ['Hello']);
    });

    test('should apply transformContext before convertToLlm', () async {
      var transformed = <AgentMessage>[];
      var converted = <Message>[];
      await agentLoop(
        [user('new message')],
        AgentContext([
          user('old message 1'),
          assistant([TextContent('old response 1')]),
          user('old message 2'),
          assistant([TextContent('old response 2')]),
        ]),
        AgentLoopConfig(
          model: model,
          transformContext: (messages, _) async => transformed = messages.sublist(messages.length - 2),
          convertToLlm: (messages) => converted = identity(messages),
        ),
        scripted((_, _) => assistant([TextContent('Response')])),
      ).result;

      expect(transformed, hasLength(2));
      expect(converted, hasLength(2));
    });

    test('should handle tool calls and results', () async {
      final executed = <String>[];
      final usage = Usage(input: 1, output: 2, cacheRead: 3, cacheWrite: 4, totalTokens: 10);
      final patched = Usage(input: 5, output: 6, cacheRead: 7, cacheWrite: 8, totalTokens: 26);
      Usage? observed;
      final (events, messages) = await drain(
        agentLoop(
          [user('echo something')],
          AgentContext(
            [],
            tools: [
              tool('echo', (args) async {
                executed.add(value(args));
                return AgentToolResult(
                  [TextContent('echoed: ${value(args)}')],
                  details: {'value': value(args)},
                  usage: usage,
                );
              }),
            ],
          ),
          AgentLoopConfig(
            model: model,
            convertToLlm: identity,
            afterToolCall: (call, result, isError, _) async {
              observed = result.usage;
              return AfterToolCall(usage: patched);
            },
          ),
          callsThenDone([
            ('tool-1', 'echo', {'value': 'hello'}),
          ]),
        ),
      );

      expect(executed, ['hello']);
      expect(events.whereType<ToolExecutionStart>(), hasLength(1));
      expect(events.whereType<ToolExecutionEnd>().single.isError, isFalse);
      expect(observed, same(usage));
      expect(messages.whereType<ToolResultMessage>().single.usage, same(patched));
    });

    // pi's #10549
    test('records how long execute() took on the tool result, excluding hooks', () async {
      final [ran, blocked] = (await agentLoop(
        [user('go')],
        AgentContext(
          [],
          tools: [
            tool('echo', (args) async {
              await Future<void>.delayed(const Duration(milliseconds: 30));
              return text(value(args));
            }),
          ],
        ),
        AgentLoopConfig(
          model: model,
          convertToLlm: identity,
          beforeToolCall: (call, _) async {
            await Future<void>.delayed(const Duration(milliseconds: 100));
            return call.call.id == 'blocked' ? const BeforeToolCall(block: true, reason: 'no') : null;
          },
        ),
        callsThenDone([
          ('ran', 'echo', {'value': 'a'}),
          ('blocked', 'echo', {'value': 'b'}),
        ]),
      ).result).whereType<ToolResultMessage>().toList();
      expect(ran.durationMs, inInclusiveRange(25, 99));
      expect(blocked.isError, isTrue);
      expect(blocked.durationMs, isNull);
    });

    test('should not execute tool calls from a length-truncated assistant message', () async {
      final executed = <String>[];
      var requests = 0;
      final (events, messages) = await drain(
        agentLoop(
          [user('echo something')],
          AgentContext(
            [],
            tools: [
              tool('echo', (args) async {
                executed.add(value(args));
                return text('echoed');
              }),
            ],
          ),
          const AgentLoopConfig(model: model, convertToLlm: identity),
          scripted((n, _) {
            requests++;
            if (n > 0) {
              return assistant([TextContent('done')]);
            }

            // The output hit the token limit mid tool call, so its arguments may be cut off.
            return assistant([
              ToolCall('tool-1', 'echo', {'value': 'hel'}),
            ], StopReason.length);
          }),
        ),
      );

      expect(executed, isEmpty);
      final end = events.whereType<ToolExecutionEnd>().single;
      expect(end.isError, isTrue);
      expect((end.result.content.single as TextContent).text, contains('output token limit'));
      expect(requests, 2);
      expect(messages.last, isA<AssistantMessage>());
    });

    test('should execute mutated beforeToolCall args without revalidation', () async {
      final executed = <Object?>[];
      await agentLoop(
        [user('echo something')],
        AgentContext(
          [],
          tools: [
            tool('echo', (args) async {
              executed.add((args as Json)['value']);
              return text('echoed');
            }),
          ],
        ),
        AgentLoopConfig(
          model: model,
          convertToLlm: identity,
          beforeToolCall: (call, _) async {
            (call.arguments as Json)['value'] = 123;
            return null;
          },
        ),
        callsThenDone([
          ('tool-1', 'echo', {'value': 'hello'}),
        ]),
      ).result;

      expect(executed, [123]);
    });

    test('should prepare tool arguments for validation', () async {
      final executed = <Object?>[];
      await agentLoop(
        [user('edit something')],
        AgentContext(
          [],
          tools: [
            tool(
              'edit',
              (args) async {
                executed.add((args as Json)['edits']);
                return text('edited');
              },
              prepare: (args) {
                if (args['oldText'] is! String || args['newText'] is! String) {
                  return args;
                }

                return {
                  'edits': [
                    ...?args['edits'] as List?,
                    {'oldText': args['oldText'], 'newText': args['newText']},
                  ],
                };
              },
            ),
          ],
        ),
        const AgentLoopConfig(model: model, convertToLlm: identity),
        callsThenDone([
          ('tool-1', 'edit', {'oldText': 'before', 'newText': 'after'}),
        ]),
      ).result;

      expect(executed, [
        [
          {'oldText': 'before', 'newText': 'after'},
        ],
      ]);
    });

    test('should emit tool_execution_end in completion order but persist tool results in source order', () async {
      final release = Completer<void>();
      var firstDone = false;
      var parallel = false;
      final (events, _) = await drain(
        agentLoop(
          [user('echo both')],
          AgentContext(
            [],
            tools: [
              tool('echo', (args) async {
                if (value(args) == 'first') {
                  await release.future;
                  firstDone = true;
                }

                if (value(args) == 'second' && !firstDone) {
                  parallel = true;
                }

                return text('echoed: ${value(args)}');
              }),
            ],
          ),
          const AgentLoopConfig(model: model, convertToLlm: identity),
          scripted((n, _) {
            if (n > 0) {
              return assistant([TextContent('done')]);
            }

            Timer(const Duration(milliseconds: 20), release.complete);
            return calls([
              ('tool-1', 'echo', {'value': 'first'}),
              ('tool-2', 'echo', {'value': 'second'}),
            ]);
          }),
        ),
      );

      expect(parallel, isTrue);
      expect(toolIds<ToolExecutionEnd>(events), ['tool-2', 'tool-1']);
      expect(
        [
          for (final e in events.whereType<MessageEnd>())
            if (e.message case ToolResultMessage(:final id)) id,
        ],
        ['tool-1', 'tool-2'],
      );
      expect(events.whereType<TurnEnd>().first.results.map((r) => r.id), ['tool-1', 'tool-2']);
    });

    test('should inject queued messages after all tool calls complete', () async {
      final executed = <String>[];
      var delivered = false;
      var sawInterrupt = false;
      final (events, _) = await drain(
        agentLoop(
          [user('start')],
          AgentContext(
            [],
            tools: [
              tool('echo', (args) async {
                executed.add(value(args));
                return text('ok:${value(args)}');
              }),
            ],
          ),
          AgentLoopConfig(
            model: model,
            convertToLlm: identity,
            toolExecution: ToolExecution.sequential,
            getSteeringMessages: () async {
              if (executed.isNotEmpty && !delivered) {
                delivered = true;
                return [user('interrupt')];
              }

              return [];
            },
          ),
          scripted((n, messages) {
            if (n == 1) {
              sawInterrupt = messages.any((m) => userText(m) == 'interrupt');
            }

            if (n > 0) {
              return assistant([TextContent('done')]);
            }

            return calls([
              ('tool-1', 'echo', {'value': 'first'}),
              ('tool-2', 'echo', {'value': 'second'}),
            ]);
          }),
        ),
      );

      expect(executed, ['first', 'second']);
      expect(events.whereType<ToolExecutionEnd>().map((e) => e.isError), [false, false]);
      final sequence = [
        for (final e in events.whereType<MessageStart>())
          if (e.message case ToolResultMessage(:final id)) 'tool:$id' else ?userText(e.message),
      ];
      expect(sequence.indexOf('tool:tool-1'), lessThan(sequence.indexOf('interrupt')));
      expect(sequence.indexOf('tool:tool-2'), lessThan(sequence.indexOf('interrupt')));
      expect(sawInterrupt, isTrue);
    });

    for (final (name, tools) in [
      ('a tool has executionMode=sequential even with default parallel config', ['slow']),
      ('one of multiple tools has executionMode=sequential', ['slow', 'fast']),
    ]) {
      test('should force sequential execution when $name', () async {
        final release = Completer<void>();
        var firstDone = false;
        var parallel = false;

        Future<AgentToolResult> run(Object? args) async {
          if (value(args) == 'first') {
            await release.future;
            firstDone = true;
          }

          if (value(args) == 'second' && !firstDone) {
            parallel = true;
          }

          return text(value(args));
        }

        final (events, _) = await drain(
          agentLoop(
            [user('run both')],
            AgentContext(
              [],
              tools: [
                tool('slow', run, mode: ToolExecution.sequential),
                tool('fast', run, mode: ToolExecution.parallel),
              ],
            ),
            const AgentLoopConfig(model: model, convertToLlm: identity),
            scripted((n, _) {
              if (n > 0) {
                return assistant([TextContent('done')]);
              }

              Timer(const Duration(milliseconds: 20), release.complete);
              return calls([
                ('tool-1', tools.first, {'value': 'first'}),
                ('tool-2', tools.last, {'value': 'second'}),
              ]);
            }),
          ),
        );

        expect(parallel, isFalse);
        expect(toolIds<ToolExecutionEnd>(events), ['tool-1', 'tool-2']);
      });
    }

    test('should allow parallel execution when all tools have executionMode=parallel', () async {
      final release = Completer<void>();
      var firstDone = false;
      var parallel = false;

      Future<AgentToolResult> run(Object? args) async {
        if (value(args) == 'first') {
          await release.future;
          firstDone = true;
        }

        if (value(args) == 'second' && !firstDone) {
          parallel = true;
        }

        return text(value(args));
      }

      await agentLoop(
        [user('run both')],
        AgentContext(
          [],
          tools: [
            tool('a', run, mode: ToolExecution.parallel),
            tool('b', run, mode: ToolExecution.parallel),
          ],
        ),
        const AgentLoopConfig(model: model, convertToLlm: identity),
        scripted((n, _) {
          if (n > 0) {
            return assistant([TextContent('done')]);
          }

          Timer(const Duration(milliseconds: 20), release.complete);
          return calls([
            ('tool-1', 'a', {'value': 'first'}),
            ('tool-2', 'b', {'value': 'second'}),
          ]);
        }),
      ).result;

      expect(parallel, isTrue);
    });

    // pi's #9986
    test('answers every tool call of a sequential batch aborted mid-batch', () async {
      final signal = AbortSignal();
      final executed = <String>[];
      final (events, messages) = await runBatch(
        tool('slow', (args) async {
          executed.add(value(args));

          // The user aborts while the first call runs.
          signal.abort();
          throw Exception('interrupted');
        }, mode: ToolExecution.sequential),
        signal,
      );

      expect(executed, ['first']);
      expect(resultTexts(messages), [
        ('tool-1', 'interrupted'),
        ('tool-2', 'Operation aborted'),
        ('tool-3', 'Operation aborted'),
      ]);
      expect(toolIds<ToolExecutionStart>(events), ['tool-1', 'tool-2', 'tool-3']);
      expect(toolIds<ToolExecutionEnd>(events), ['tool-1', 'tool-2', 'tool-3']);
    });

    // pi's #9986
    test('answers every tool call of a parallel batch aborted while calls are prepared', () async {
      final signal = AbortSignal();
      final executed = <String>[];
      final asked = <String>[];
      final (events, messages) = await runBatch(
        tool('slow', (args) async {
          executed.add(value(args));
          return text(value(args));
        }),
        signal,

        // The user aborts while a hook decides about the second call, at a permission prompt say.
        ToolHooks(
          beforeToolCall: (call, _) async {
            asked.add(call.call.id);
            if (call.call.id == 'tool-2') {
              signal.abort();
            }

            return null;
          },
        ),
      );

      expect(executed, isEmpty);
      expect(asked, ['tool-1', 'tool-2']);
      expect(resultTexts(messages), [
        ('tool-1', 'Operation aborted'),
        ('tool-2', 'Operation aborted'),
        ('tool-3', 'Operation aborted'),
      ]);
      expect(toolIds<ToolExecutionStart>(events), ['tool-1', 'tool-2', 'tool-3']);
      expect(toolIds<ToolExecutionEnd>(events)..sort(), ['tool-1', 'tool-2', 'tool-3']);
    });

    test('runs finishTurn after tool-result messages and before turn_end', () async {
      final order = <String>[];
      await runAgentLoop(
        [user('echo')],
        AgentContext([], tools: [tool('echo', (args) async => text(value(args), terminate: true))]),
        AgentLoopConfig(
          model: model,
          convertToLlm: identity,
          finishTurn: (turn, _) {
            order.add('finishTurn');
            expect(turn.results, hasLength(1));
            expect(turn.context.messages.last, isA<ToolResultMessage>());
            return null;
          },
        ),
        (event) {
          if (event is MessageEnd) {
            order.add('message_end:${roles([event.message]).single}');
          }

          if (event is TurnEnd) {
            order.add('turn_end');
          }
        },
        scripted(
          (_, _) => calls([
            ('tool-1', 'echo', {'value': 'hello'}),
          ]),
        ),
      );

      expect(order.sublist(order.length - 3), ['message_end:toolResult', 'finishTurn', 'turn_end']);
    });

    for (final reason in [StopReason.error, StopReason.aborted]) {
      test('runs finishTurn for a ${reason.name} assistant before turn_end without changing the hard exit', () async {
        final order = <String>[];
        var requests = 0;
        var steering = 0;
        var followUps = 0;
        await runAgentLoop(
          [user('run')],
          AgentContext([]),
          AgentLoopConfig(
            model: model,
            convertToLlm: identity,
            finishTurn: (turn, _) {
              expect(turn.message.stopReason, reason);
              order.add('finishTurn');
              return TurnDecision.proceed;
            },
            getSteeringMessages: () async {
              steering++;
              return [];
            },
            getFollowUpMessages: () async {
              followUps++;
              return [user('queued')];
            },
          ),
          (event) {
            if (event is TurnEnd) {
              order.add('turn_end');
            }
          },
          scripted((_, _) {
            requests++;
            return assistant([], reason)..errorMessage = reason.name;
          }),
        );

        expect(order, ['finishTurn', 'turn_end']);
        expect(requests, 1);
        expect(steering, 1);
        expect(followUps, 0);
      });
    }

    test('action:end skips queue polling and next-turn preparation', () async {
      var requests = 0;
      var steering = 0;
      var followUps = 0;
      var prepares = 0;
      await agentLoop(
        [user('run')],
        AgentContext([], tools: [tool('noop', (_) async => text('done'))]),
        AgentLoopConfig(
          model: model,
          convertToLlm: identity,
          finishTurn: (_, _) => TurnDecision.end,
          prepareNextTurn: (_) {
            prepares++;
            return null;
          },
          getSteeringMessages: () async {
            steering++;
            return [];
          },
          getFollowUpMessages: () async {
            followUps++;
            return [user('queued')];
          },
        ),
        scripted((_, _) {
          requests++;
          return calls([('tool-1', 'noop', {})]);
        }),
      ).result;

      expect((requests, steering, followUps, prepares), (1, 1, 0, 0));
    });

    test('makes exactly one context-only request when no natural request satisfies continuation', () async {
      var requests = 0;
      var finishes = 0;
      await agentLoop(
        [user('run')],
        AgentContext([]),
        AgentLoopConfig(
          model: model,
          convertToLlm: identity,
          finishTurn: (_, _) => ++finishes == 1 ? TurnDecision.proceed : null,
        ),
        scripted((n, _) {
          requests++;
          return assistant([TextContent('response $requests')]);
        }),
      ).result;

      expect((requests, finishes), (2, 2));
    });

    test('lets a natural tool-result request satisfy continuation', () async {
      var requests = 0;
      var finishes = 0;
      await agentLoop(
        [user('run')],
        AgentContext([], tools: [tool('noop', (_) async => text('done'))]),
        AgentLoopConfig(
          model: model,
          convertToLlm: identity,
          finishTurn: (_, _) => ++finishes == 1 ? TurnDecision.proceed : null,
        ),
        scripted((n, _) {
          requests++;
          return n == 0 ? calls([('tool-1', 'noop', {})]) : assistant([TextContent('done')]);
        }),
      ).result;

      expect((requests, finishes), (2, 2));
    });

    for (final kind in ['steering', 'follow-up']) {
      test('lets a natural $kind request satisfy continuation', () async {
        var requests = 0;
        var finishes = 0;
        var steeringPolls = 0;
        var followUpDelivered = false;
        final second = <String>[];
        await agentLoop(
          [user('run')],
          AgentContext([]),
          AgentLoopConfig(
            model: model,
            convertToLlm: identity,
            finishTurn: (_, _) => ++finishes == 1 ? TurnDecision.proceed : null,
            getSteeringMessages: () async {
              steeringPolls++;
              if (kind == 'steering' && steeringPolls == 2) {
                return [user(kind)];
              }

              return [];
            },
            getFollowUpMessages: () async {
              if (kind != 'follow-up' || followUpDelivered) {
                return [];
              }

              followUpDelivered = true;
              return [user(kind)];
            },
          ),
          scripted((n, messages) {
            requests++;
            if (n == 1) {
              second.addAll(messages.map(userText).nonNulls);
            }

            return assistant([TextContent('done')]);
          }),
        ).result;

        expect((requests, finishes), (2, 2));
        expect(second, contains(kind));
      });
    }

    test('prepares the initial request after pending messages and can replace request state', () async {
      const replacement = Model(
        id: 'replacement',
        name: 'replacement',
        api: 'openai-responses',
        provider: 'openai',
        baseUrl: 'https://example.invalid',
        cost: ModelCost(input: 0, output: 0, cacheRead: 0, cacheWrite: 0),
        contextWindow: 8192,
        maxTokens: 2048,
      );
      final canonical = user('canonical projection');
      final steering = user('steering');
      final completed = <AgentMessage>[];
      var delivered = false;
      var prepares = 0;
      await runAgentLoop(
        [user('prompt')],
        AgentContext([]),
        AgentLoopConfig(
          model: model,
          convertToLlm: identity,
          getSteeringMessages: () async {
            if (delivered) {
              return [];
            }

            delivered = true;
            return [steering];
          },
          prepareRequest: (context, model, level, _) {
            prepares++;
            expect(completed, contains(steering));
            expect(context.messages, contains(steering));
            return TurnUpdate(context: AgentContext([canonical]), model: replacement, level: ThinkingLevel.high);
          },
        ),
        (event) {
          if (event is MessageEnd) {
            completed.add(event.message);
          }
        },
        (m, messages, options) {
          expect(m, same(replacement));
          expect(messages, [canonical]);
          expect(options.reasoning, ThinkingLevel.high);
          return scripted((_, _) => assistant([TextContent('done')]))(m, messages, options);
        },
      );

      expect(prepares, 1);
    });

    test('does not poll steering after prepareRequest', () async {
      final queued = <AgentMessage>[];
      final late = user('late steering');
      final included = <bool>[];
      var prepares = 0;
      var polls = 0;
      await agentLoop(
        [user('run')],
        AgentContext([]),
        AgentLoopConfig(
          model: model,
          convertToLlm: identity,
          getSteeringMessages: () async {
            polls++;
            final taken = [...queued];
            queued.clear();
            return taken;
          },
          prepareRequest: (_, _, _, _) {
            if (++prepares == 1) {
              queued.add(late);
            }

            return null;
          },
        ),
        scripted((_, messages) {
          included.add(messages.contains(late));
          return assistant([TextContent('done')]);
        }),
      ).result;

      expect(included, [false, true]);
      expect(prepares, 2);

      // Startup, post-turn delivery, then the final natural-stop check.
      expect(polls, 3);
    });

    test('should use prepareNextTurn snapshot before continuing', () async {
      var updated = false;
      var prepares = 0;
      var prepared = false;
      var requests = 0;
      await agentLoop(
        [user('echo something')],
        AgentContext([], tools: [tool('echo', (args) async => text('echoed: ${value(args)}'))]),
        AgentLoopConfig(
          model: model,
          convertToLlm: identity,
          prepareNextTurn: (turn) async {
            prepares++;
            if (prepared) {
              return null;
            }

            prepared = true;
            return TurnUpdate(
              context: AgentContext([...turn.context.messages], tools: turn.context.tools),
              messages: [
                SystemMessage([TextContent('updated guidance')], timestamp: 1),
              ],
            );
          },
        ),
        scripted((n, messages) {
          requests++;
          if (n == 1) {
            updated = messages.whereType<SystemMessage>().any(
              (m) => m.content.any((c) => c.text == 'updated guidance'),
            );
          }

          return n == 0
              ? calls([
                  ('tool-1', 'echo', {'value': 'hello'}),
                ])
              : assistant([TextContent('done')]);
        }),
      ).result;

      expect((requests, prepares, updated), (2, 1, true));
    });

    test('picks up steering queued during prepareNextTurn before the next request', () async {
      final queued = <AgentMessage>[];
      final late = user('late steering');
      var included = false;
      await agentLoop(
        [user('run')],
        AgentContext([], tools: [tool('noop', (_) async => text('done'))]),
        AgentLoopConfig(
          model: model,
          convertToLlm: identity,
          prepareNextTurn: (_) {
            queued.add(late);
            return null;
          },
          getSteeringMessages: () async {
            final taken = [...queued];
            queued.clear();
            return taken;
          },
        ),
        scripted((n, messages) {
          if (n == 1) {
            included = messages.contains(late);
          }

          return n == 0 ? calls([('tool-1', 'noop', {})]) : assistant([TextContent('done')]);
        }),
      ).result;

      expect(included, isTrue);
    });

    test('action:end receives finalized turn context and stops before queue polling', () async {
      final executed = <String>[];
      var steering = 0;
      var followUps = 0;
      var requests = 0;
      List<String>? resultIds;
      List<String>? contextRoles;
      final (events, messages) = await drain(
        agentLoop(
          [user('echo something')],
          AgentContext(
            [],
            tools: [
              tool('echo', (args) async {
                executed.add(value(args));
                return text('echoed: ${value(args)}');
              }),
            ],
          ),
          AgentLoopConfig(
            model: model,
            convertToLlm: identity,
            finishTurn: (turn, _) async {
              resultIds = turn.results.map((r) => r.id).toList();
              contextRoles = roles(turn.context.messages);
              return TurnDecision.end;
            },
            getSteeringMessages: () async {
              steering++;
              return [];
            },
            getFollowUpMessages: () async {
              followUps++;
              return [user('follow up should stay queued')];
            },
          ),
          scripted((n, _) {
            requests++;
            return n == 0
                ? calls([
                    ('tool-1', 'echo', {'value': 'hello'}),
                  ])
                : assistant([TextContent('should not run')]);
          }),
        ),
      );

      expect((requests, steering, followUps), (1, 1, 0));
      expect(executed, ['hello']);
      expect(resultIds, ['tool-1']);
      expect(contextRoles, ['system', 'user', 'assistant', 'toolResult']);

      // The context declares no tools, so the loop announces them with a system message.
      expect(roles(messages), ['system', 'user', 'assistant', 'toolResult']);
      expect(events.map((e) => e.runtimeType), [
        AgentStart,
        TurnStart,
        MessageStart,
        MessageEnd,
        MessageStart,
        MessageEnd,
        MessageStart,
        MessageEnd,
        ToolExecutionStart,
        ToolExecutionEnd,
        MessageStart,
        MessageEnd,
        TurnEnd,
        AgentEnd,
      ]);
    });

    test('should stop after a tool batch when every tool result sets terminate=true', () async {
      var requests = 0;
      final (events, messages) = await drain(
        agentLoop(
          [user('echo something')],
          AgentContext([], tools: [tool('echo', (args) async => text('echoed', terminate: true))]),
          const AgentLoopConfig(model: model, convertToLlm: identity),
          scripted((_, _) {
            requests++;
            return calls([
              ('tool-1', 'echo', {'value': 'hello'}),
            ]);
          }),
        ),
      );

      expect(requests, 1);
      expect(roles(messages), ['system', 'user', 'assistant', 'toolResult']);
      expect(events.whereType<TurnEnd>(), hasLength(1));
    });

    test('should stop after a blocked tool call when beforeToolCall sets terminate=true', () async {
      var executed = false;
      var requests = 0;
      final result = (await agentLoop(
        [user('echo something')],
        AgentContext(
          [],
          tools: [
            tool('echo', (_) async {
              executed = true;
              return text('should not execute');
            }),
          ],
        ),
        AgentLoopConfig(
          model: model,
          convertToLlm: identity,
          beforeToolCall: (_, _) async =>
              const BeforeToolCall(block: true, reason: 'Blocked by policy', terminate: true),
        ),
        scripted((n, _) {
          requests++;
          return n == 0
              ? calls([
                  ('tool-1', 'echo', {'value': 'hello'}),
                ])
              : assistant([TextContent('should not run')]);
        }),
      ).result).whereType<ToolResultMessage>().single;
      expect(executed, isFalse);
      expect(requests, 1);
      expect(result.isError, isTrue);
      expect((result.content.single as TextContent).text, 'Blocked by policy');
    });

    test('should continue after a mixed batch with one terminating blocked call', () async {
      final executed = <String>[];
      var requests = 0;
      await agentLoop(
        [user('echo both')],
        AgentContext(
          [],
          tools: [
            tool('echo', (args) async {
              executed.add(value(args));
              return text('echoed');
            }),
          ],
        ),
        AgentLoopConfig(
          model: model,
          convertToLlm: identity,
          beforeToolCall: (call, _) async {
            if (value(call.arguments) == 'first') {
              return const BeforeToolCall(block: true, reason: 'Blocked first', terminate: true);
            }

            return null;
          },
        ),
        scripted((n, _) {
          requests++;
          if (n > 0) {
            return assistant([TextContent('done')]);
          }

          return calls([
            ('tool-1', 'echo', {'value': 'first'}),
            ('tool-2', 'echo', {'value': 'second'}),
          ]);
        }),
      ).result;

      expect(executed, ['second']);
      expect(requests, 2);
    });

    test('should continue after parallel tool calls when not all tool results terminate', () async {
      expect(
        roles(
          (await agentLoop(
            [user('echo both')],
            AgentContext([], tools: [tool('echo', (args) async => text('echoed', terminate: value(args) == 'first'))]),
            const AgentLoopConfig(model: model, convertToLlm: identity),
            callsThenDone([
              ('tool-1', 'echo', {'value': 'first'}),
              ('tool-2', 'echo', {'value': 'second'}),
            ]),
          ).result),
        ),
        ['system', 'user', 'assistant', 'toolResult', 'toolResult', 'assistant'],
      );
    });

    test('should allow afterToolCall to mark a tool batch as terminating', () async {
      var requests = 0;
      await agentLoop(
        [user('echo something')],
        AgentContext([], tools: [tool('echo', (_) async => text('echoed'))]),
        AgentLoopConfig(
          model: model,
          convertToLlm: identity,
          afterToolCall: (_, _, _, _) async => const AfterToolCall(terminate: true),
        ),
        scripted((_, _) {
          requests++;
          return calls([
            ('tool-1', 'echo', {'value': 'hello'}),
          ]);
        }),
      ).result;

      expect(requests, 1);
    });
  });

  group('chorus', () {
    test('records the thinking level asked for on the reply', () async {
      for (final level in [ThinkingLevel.off, ThinkingLevel.high]) {
        expect(
          ((await agentLoop(
                    [user('think')],
                    AgentContext([]),
                    AgentLoopConfig(model: model, convertToLlm: identity, level: level),
                    scripted((_, _) => assistant([TextContent('done')])),
                  ).result).last
                  as AssistantMessage)
              .thinkingLevel,
          level,
        );
      }
    });

    test('lets afterToolCall turn a result into an error', () async {
      expect(
        (await agentLoop(
          [user('echo')],
          AgentContext([], tools: [tool('echo', (_) async => text('looks fine'))]),
          AgentLoopConfig(
            model: model,
            convertToLlm: identity,
            afterToolCall: (_, _, _, _) async => const AfterToolCall(isError: true),
          ),
          callsThenDone([
            ('tool-1', 'echo', {'value': 'hello'}),
          ]),
        ).result).whereType<ToolResultMessage>().single.isError,
        isTrue,
      );
    });
  });

  group('agentLoopContinue with AgentMessage', () {
    test('should throw when context has no messages', () {
      expect(
        () => agentLoopContinue(
          AgentContext([]),
          const AgentLoopConfig(model: model, convertToLlm: identity),
          (_, _, _) => throw StateError('Unexpected stream call'),
        ),
        throwsA(isA<StateError>().having((e) => e.message, 'message', 'Cannot continue: no messages in context')),
      );
    });

    test('should continue from existing context without emitting user message events', () async {
      final (events, messages) = await drain(
        agentLoopContinue(
          AgentContext([user('Hello')]),
          const AgentLoopConfig(model: model, convertToLlm: identity),
          scripted((_, _) => assistant([TextContent('Response')])),
        ),
      );

      expect(roles(messages), ['assistant']);
      expect(events.whereType<MessageEnd>().map((e) => roles([e.message]).single), ['assistant']);
    });

    test('should allow custom message types as last message (caller responsibility)', () async {
      final messages = await agentLoopContinue(
        AgentContext(['Hook content']),
        AgentLoopConfig(
          model: model,
          convertToLlm: (messages) => [for (final m in messages) m is String ? user(m) : m as Message],
        ),
        scripted((_, messages) {
          expect(messages.map(userText), ['Hook content']);
          return assistant([TextContent('Response to custom message')]);
        }),
      ).result;

      expect(roles(messages), ['assistant']);
    });
  });

  group('runToolCall', () {
    final echo = AgentTool(
      name: 'echo',
      description: 'Echo',
      parameters: _schema,

      // Stands in for pi-schema's validation of `{value: string}`.
      validate: (args) {
        if (args['value'] is! String) {
          throw 'value: must be string';
        }

        return args;
      },
      execute: (id, args, signal, onUpdate) async {
        onUpdate(text('partial'));
        return text(value(args));
      },
    );
    final failing = AgentTool(
      name: 'failing',
      description: 'Fails',
      parameters: const {'type': 'object'},
      execute: (id, args, signal, onUpdate) async {
        return AgentToolResult([TextContent('bad')], details: {'partial': true}, isError: true);
      },
    );

    test('validates, runs the hooks, and reports failures as error outcomes', () async {
      final hooks = <String>[];
      final updates = <AgentToolResult>[];

      Future<ToolCallOutcome> run(String id, String name, Json args) {
        return runToolCall(
          ToolCall(id, name, args),
          tools: [echo, failing],
          message: assistant([]),
          context: AgentContext([]),
          hooks: ToolHooks(
            beforeToolCall: (call, _) async {
              hooks.add('before ${call.call.id}');
              if ((call.arguments as Json)['value'] == 'blocked') {
                return const BeforeToolCall(block: true, reason: 'nope');
              }

              return null;
            },
            afterToolCall: (call, result, isError, _) async {
              hooks.add('after ${call.call.id}');
              return null;
            },
          ),
          onUpdate: updates.add,
        );
      }

      final a = await run('a', 'echo', {'value': 'a'});
      expect((a.call.id, a.isError, (a.result.content.single as TextContent).text), ('a', false, 'a'));
      expect(
        (await run('b', 'echo', {
          'value': {'nested': true},
        })).isError,
        isTrue,
      );
      final c = await run('c', 'echo', {'value': 'blocked'});
      expect((c.isError, (c.result.content.single as TextContent).text), (true, 'nope'));
      final d = await run('d', 'missing', {});
      expect((d.isError, (d.result.content.single as TextContent).text), (true, 'Tool missing not found'));

      // Error results keep their details.
      final e = await run('e', 'failing', {});
      expect(e.isError, isTrue);
      expect(e.result.details, {'partial': true});
      expect(updates.map((u) => (u.content.single as TextContent).text), ['partial']);

      // Validation failures and unknown tools never reach the hooks; blocked calls skip afterToolCall.
      expect(hooks, ['before a', 'after a', 'before c', 'before e', 'after e']);
    });
  });
}
