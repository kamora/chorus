// Ported from pi's packages/ai/test/faux-provider.test.ts, through a Models collection (pi's compat global registry is
// deprecated) and without deferred responses.

import 'dart:convert';
import 'dart:math';

import 'package:chorus/chorus.dart';
import 'package:test/test.dart';

final messages = <Message>[
  UserMessage([TextContent('hi')], timestamp: 0),
];

UserMessage user(String text) => UserMessage([TextContent(text)], timestamp: 1);

/// A faux provider registered in its own collection.
(FauxProvider, Models) register([FauxProvider? provider]) {
  final faux = provider ?? FauxProvider();
  return (faux, Models()..setProvider(faux.provider));
}

List<Json> content(AssistantMessage m) => [for (final c in m.content) c.toJson()];

void main() {
  test('registers a custom provider and estimates usage', () async {
    final (faux, models) = register();
    faux.setResponses([fauxAssistantMessage('hello world')]);
    final response = await models.completeSimple(faux.getModel()!, [
      SystemMessage([TextContent('Be concise.')], timestamp: 0),
      user('hi there'),
    ]);
    expect(content(response), [
      {'type': 'text', 'text': 'hello world'},
    ]);
    expect(response.usage.input, greaterThan(0));
    expect(response.usage.output, greaterThan(0));
    expect(response.usage.totalTokens, response.usage.input + response.usage.output);
    expect(faux.state.callCount, 1);
  });

  test('supports helper blocks for text, thinking, and tool calls', () async {
    final (faux, models) = register();
    faux.setResponses([
      fauxAssistantMessage([
        fauxThinking('think'),
        fauxToolCall('echo', {'text': 'hi'}),
        fauxText('done'),
      ], stopReason: StopReason.toolUse),
    ]);
    final response = await models.completeSimple(faux.getModel()!, messages);
    expect(response.content.map((c) => c.runtimeType), [ThinkingContent, ToolCall, TextContent]);
    expect((response.content[0] as ThinkingContent).thinking, 'think');
    final call = response.content[1] as ToolCall;
    expect((call.name, call.arguments['text'], call.id.isNotEmpty), ('echo', 'hi', true));
    expect((response.content[2] as TextContent).text, 'done');
    expect(response.stopReason, StopReason.toolUse);
  });

  test('supports multiple models with per-model reasoning and model-aware factories', () async {
    final (faux, models) = register(
      FauxProvider(
        models: const [
          FauxModel('faux-fast', name: 'Faux Fast'),
          FauxModel('faux-thinker', name: 'Faux Thinker', reasoning: true),
        ],
      ),
    );
    AssistantMessage answer(List<Message> m, SimpleStreamOptions o, FauxState s, Model model) {
      return fauxAssistantMessage('${model.id}:${model.reasoning}');
    }

    faux.setResponses([answer, answer]);
    expect(faux.models.map((m) => m.id), ['faux-fast', 'faux-thinker']);
    expect(faux.getModel(), same(faux.models.first));
    expect(faux.getModel('faux-fast')?.reasoning, isFalse);
    expect(faux.getModel('faux-thinker')?.reasoning, isTrue);
    expect(faux.getModel('missing'), isNull);
    expect(
      ((await models.completeSimple(faux.getModel('faux-fast')!, messages)).content.single as TextContent).text,
      'faux-fast:false',
    );
    expect(
      ((await models.completeSimple(faux.getModel('faux-thinker')!, messages)).content.single as TextContent).text,
      'faux-thinker:true',
    );
  });

  test('rewrites api, provider, and model on returned messages', () async {
    final (faux, models) = register(
      FauxProvider(api: 'faux:test', id: 'faux-provider', models: const [FauxModel('faux-model')]),
    );
    faux.setResponses([fauxAssistantMessage('hello')]);
    final response = await models.completeSimple(faux.getModel()!, messages);
    expect((response.api, response.provider, response.model), ('faux:test', 'faux-provider', 'faux-model'));
  });

  test('consumes queued responses in order and errors when exhausted', () async {
    final (faux, models) = register();
    faux.setResponses([fauxAssistantMessage('first'), fauxAssistantMessage('second')]);
    Future<String> next() async =>
        ((await models.completeSimple(faux.getModel()!, messages)).content.single as TextContent).text;

    expect(await next(), 'first');
    expect(await next(), 'second');
    final exhausted = await models.completeSimple(faux.getModel()!, messages);
    expect((exhausted.stopReason, exhausted.errorMessage), (StopReason.error, 'No more faux responses queued'));
    expect(faux.pendingResponseCount, 0);
    expect(faux.state.callCount, 3);
  });

  test('can replace and append queued responses', () async {
    final (faux, models) = register();
    Future<String> next() async =>
        ((await models.completeSimple(faux.getModel()!, messages)).content.single as TextContent).text;

    faux.setResponses([fauxAssistantMessage('first')]);
    expect(await next(), 'first');
    expect(faux.pendingResponseCount, 0);
    faux.setResponses([fauxAssistantMessage('second')]);
    expect(faux.pendingResponseCount, 1);
    expect(await next(), 'second');
    faux.appendResponses([fauxAssistantMessage('third'), fauxAssistantMessage('fourth')]);
    expect(faux.pendingResponseCount, 2);
    expect(await next(), 'third');
    expect(await next(), 'fourth');
    expect(faux.pendingResponseCount, 0);
  });

  test('supports async response factories', () async {
    final (faux, models) = register();
    faux.setResponses([
      (List<Message> m, SimpleStreamOptions o, FauxState state, Model model) async =>
          fauxAssistantMessage('${m.length}:${state.callCount}'),
    ]);
    final response = await models.completeSimple(faux.getModel()!, messages);
    expect((response.content.single as TextContent).text, '1:1');
  });

  test('emits an error when a response factory throws', () async {
    final (faux, models) = register();
    faux.setResponses([(List<Message> m, SimpleStreamOptions o, FauxState s, Model model) => throw Exception('boom')]);
    final events = await models.streamSimple(faux.getModel()!, messages).toList();
    expect(events, hasLength(1));
    final error = (events.single as ErrorEvent).error;
    expect((error.stopReason, error.errorMessage), (StopReason.error, 'boom'));
  });

  test('rejects a queued response without a terminal stop reason', () async {
    final (faux, models) = register();
    faux.setResponses([fauxAssistantMessage('partial', stopReason: StopReason.pending)]);
    final events = await models.streamSimple(faux.getModel()!, messages).toList();
    expect(events.whereType<DoneEvent>(), isEmpty);
    final error = (events.last as ErrorEvent).error;
    expect((error.stopReason, error.errorMessage), (StopReason.error, 'Faux response ended without a stop reason'));
  });

  test('estimates prompt and output tokens from serialized context', () async {
    final (faux, models) = register();
    faux.setResponses([fauxAssistantMessage('done')]);
    const tool = Tool('echo', 'Echo back text', {
      'type': 'object',
      'properties': {
        'text': {'type': 'string'},
      },
      'required': ['text'],
    });
    final response = await models.completeSimple(faux.getModel()!, [
      SystemMessage([TextContent('sys')], added: const [tool], timestamp: 0),
      UserMessage([TextContent('hello'), ImageContent('abcd', 'image/png')], timestamp: 1),
      fauxAssistantMessage('prior'),
      ToolResultMessage(id: 'tool-1', name: 'echo', content: [TextContent('tool out')], isError: false, timestamp: 2),
    ]);
    final promptTokens =
        ([
                  'system:sys\ntool+:${jsonEncode(tool.toJson())}',
                  'user:hello\n[image:image/png:4]',
                  'assistant:prior',
                  'toolResult:echo\ntool out',
                ].join('\n\n').length /
                4)
            .ceil();
    final outputTokens = ('done'.length / 4).ceil();
    expect((response.usage.input, response.usage.output), (promptTokens, outputTokens));
    expect((response.usage.cacheRead, response.usage.cacheWrite), (0, 0));
    expect(response.usage.totalTokens, promptTokens + outputTokens);
  });

  test('does not share cache across sessions or requests without sessionId', () async {
    final (faux, models) = register();
    faux.setResponses([fauxAssistantMessage('first'), fauxAssistantMessage('second'), fauxAssistantMessage('third')]);
    final transcript = <Message>[user('hello')];
    final first = await models.completeSimple(
      faux.getModel()!,
      transcript,
      const SimpleStreamOptions(sessionId: 'session-1', cacheRetention: CacheRetention.short),
    );
    expect(first.usage.cacheWrite, greaterThan(0));
    transcript.addAll([first, user('follow up')]);
    final second = await models.completeSimple(
      faux.getModel()!,
      transcript,
      const SimpleStreamOptions(sessionId: 'session-2', cacheRetention: CacheRetention.short),
    );
    expect(second.usage.cacheRead, 0);
    expect(second.usage.cacheWrite, greaterThan(0));
    final third = await models.completeSimple(faux.getModel()!, transcript);
    expect((third.usage.cacheRead, third.usage.cacheWrite), (0, 0));
  });

  test('simulates prompt caching per sessionId', () async {
    final (faux, models) = register();
    faux.setResponses([fauxAssistantMessage('first'), fauxAssistantMessage('second')]);
    const options = SimpleStreamOptions(sessionId: 'session-1', cacheRetention: CacheRetention.short);
    final transcript = <Message>[
      SystemMessage([TextContent('Be concise.')], timestamp: 0),
      user('hello'),
    ];
    final first = await models.completeSimple(faux.getModel()!, transcript, options);
    expect(first.usage.cacheRead, 0);
    expect(first.usage.cacheWrite, greaterThan(0));
    transcript.addAll([first, user('follow up')]);
    final second = await models.completeSimple(faux.getModel()!, transcript, options);
    expect(second.usage.cacheRead, greaterThan(0));
  });

  test('counts cached characters up to the first difference in the joined prompt', () async {
    final (faux, models) = register();
    faux.setResponses([fauxAssistantMessage('a'), fauxAssistantMessage('b'), fauxAssistantMessage('c')]);
    const options = SimpleStreamOptions(sessionId: 'session-1', cacheRetention: CacheRetention.short);
    Future<(int, int, int)> usage(List<String> texts) async {
      final u = (await models.completeSimple(faux.getModel()!, [for (final t in texts) user(t)], options)).usage;
      return (u.input, u.cacheRead, u.cacheWrite);
    }

    // Prompt texts: "user:hello world" (16 characters), then 16 + 2 + "user:next" (9) = 27.
    expect(await usage(['hello world']), (0, 0, 4));
    expect(await usage(['hello world', 'next']), (0, 4, 3));

    // The first message now differs after "user:hello w" (12 characters): "user:hello wide" + 2 + 9 = 26.
    expect(await usage(['hello wide', 'next']), (0, 3, 4));
  });

  test('does not simulate caching when cacheRetention is none', () async {
    final (faux, models) = register();
    faux.setResponses([fauxAssistantMessage('first'), fauxAssistantMessage('second')]);
    const options = SimpleStreamOptions(sessionId: 'session-1', cacheRetention: CacheRetention.none);
    final transcript = <Message>[user('hello')];
    await models.completeSimple(faux.getModel()!, transcript, options);
    transcript.addAll([fauxAssistantMessage('first'), user('follow up')]);
    final second = await models.completeSimple(faux.getModel()!, transcript, options);
    expect((second.usage.cacheRead, second.usage.cacheWrite), (0, 0));
  });

  test('streams thinking, text, and partial tool call deltas', () async {
    final (faux, models) = register();
    faux.setResponses([
      fauxAssistantMessage([
        fauxThinking('thinking text'),
        fauxText('answer text'),
        fauxToolCall('echo', {'text': 'hi', 'count': 12}, id: 'tool-1'),
      ], stopReason: StopReason.toolUse),
    ]);
    final events = await models.streamSimple(faux.getModel()!, messages).toList();
    final types = events.map((e) => e.runtimeType).toSet();
    expect(types, containsAll([ThinkingStartEvent, ThinkingDeltaEvent, TextStartEvent, TextDeltaEvent]));
    expect(types, containsAll([ToolCallStartEvent, ToolCallDeltaEvent, ToolCallEndEvent]));
    final deltas = [for (final e in events.whereType<ToolCallDeltaEvent>()) e.delta];
    expect(deltas.length, greaterThan(1));
    expect(jsonDecode(deltas.join()), {'text': 'hi', 'count': 12});

    // The partial call's arguments grow as the deltas arrive; paced, so the reader keeps up with the provider, which
    // grows the partial message in place.
    final (paced, pacedModels) = register(FauxProvider(tokensPerSecond: 10000, minTokenSize: 1, maxTokenSize: 1));
    final seen = <Object?>[];
    paced.setResponses([
      fauxAssistantMessage([
        fauxToolCall('echo', {'text': 'hi', 'count': 12}, id: 'tool-1'),
      ], stopReason: StopReason.toolUse),
    ]);
    await for (final event in pacedModels.streamSimple(paced.getModel()!, messages)) {
      if (event is ToolCallDeltaEvent) {
        seen.add(jsonEncode((event.partial.content.single as ToolCall).arguments));
      }
    }

    expect(seen, contains('{"text":"hi"}'));
  });

  test('streams an exact event order for fixed-size chunks', () async {
    final (faux, models) = register(FauxProvider(minTokenSize: 1, maxTokenSize: 1));
    faux.setResponses([
      fauxAssistantMessage([
        fauxThinking('go'),
        fauxText('ok'),
        fauxToolCall('echo', {}, id: 'tool-1'),
      ], stopReason: StopReason.toolUse),
    ]);
    final events = await models.streamSimple(faux.getModel()!, messages).toList();
    expect((events.first as StartEvent).partial.stopReason, StopReason.pending);
    expect(events.map((e) => e.runtimeType), [
      StartEvent,
      ThinkingStartEvent,
      ThinkingDeltaEvent,
      ThinkingEndEvent,
      TextStartEvent,
      TextDeltaEvent,
      TextEndEvent,
      ToolCallStartEvent,
      ToolCallDeltaEvent,
      ToolCallEndEvent,
      DoneEvent,
    ]);
  });

  test('streams multiple tool calls in one message', () async {
    final (faux, models) = register();
    faux.setResponses([
      fauxAssistantMessage([
        fauxToolCall('echo', {'text': 'one'}, id: 'tool-1'),
        fauxToolCall('echo', {'text': 'two'}, id: 'tool-2'),
      ], stopReason: StopReason.toolUse),
    ]);
    final events = await models.streamSimple(faux.getModel()!, messages).toList();
    expect(events.whereType<ToolCallStartEvent>(), hasLength(2));
    expect(events.whereType<ToolCallEndEvent>(), hasLength(2));
  });

  for (final (reason, error) in [(StopReason.error, 'upstream failed'), (StopReason.aborted, 'Request was aborted')]) {
    test('streams an explicit assistant ${reason.name} message as a terminal error', () async {
      final (faux, models) = register(FauxProvider(minTokenSize: 2, maxTokenSize: 2));
      faux.setResponses([fauxAssistantMessage('partial', stopReason: reason, errorMessage: error)]);
      final events = await models.streamSimple(faux.getModel()!, messages).toList();
      expect(events.map((e) => e.runtimeType), [StartEvent, TextStartEvent, TextDeltaEvent, TextEndEvent, ErrorEvent]);
      final terminal = events.last as ErrorEvent;
      expect((terminal.reason, terminal.error.stopReason, terminal.error.errorMessage), (reason, reason, error));
    });
  }

  test('supports aborting before the first chunk', () async {
    final (faux, models) = register(FauxProvider(tokensPerSecond: 50, minTokenSize: 3, maxTokenSize: 3));
    faux.setResponses([fauxAssistantMessage('abcdefghijklmnopqrstuvwxyz')]);
    final signal = AbortSignal()..abort();
    final events = await models.streamSimple(faux.getModel()!, messages, SimpleStreamOptions(signal: signal)).toList();
    expect(events, hasLength(1));
    final error = events.single as ErrorEvent;
    expect((error.reason, error.error.stopReason), (StopReason.aborted, StopReason.aborted));
  });

  for (final (name, block, delta, end) in [
    ('text', fauxText('abcdefghijklmnopqrstuvwxyz'), TextDeltaEvent, TextEndEvent),
    ('thinking', fauxThinking('abcdefghijklmnopqrstuvwxyz'), ThinkingDeltaEvent, ThinkingEndEvent),
    (
      'tool call',
      fauxToolCall('echo', {'text': 'abcdefghijklmnopqrstuvwxyz', 'count': 123456789}, id: 'tool-1'),
      ToolCallDeltaEvent,
      ToolCallEndEvent,
    ),
  ]) {
    test('supports aborting mid-$name stream when paced', () async {
      final (faux, models) = register(FauxProvider(tokensPerSecond: 100, minTokenSize: 3, maxTokenSize: 3));
      faux.setResponses([
        fauxAssistantMessage([block], stopReason: StopReason.toolUse),
      ]);
      final signal = AbortSignal();
      final types = <Type>[];
      await for (final event in models.streamSimple(faux.getModel()!, messages, SimpleStreamOptions(signal: signal))) {
        types.add(event.runtimeType);
        if (event.runtimeType == delta) {
          signal.abort();
        }
      }

      expect(types.where((t) => t == delta), hasLength(1));
      expect(types, contains(ErrorEvent));
      expect(types, isNot(contains(end)));
    });
  }

  test('fails requests once the provider is removed from the collection', () async {
    final (faux, models) = register();
    faux.setResponses([fauxAssistantMessage('hello')]);
    models.deleteProvider(faux.id);
    final result = await models.completeSimple(faux.getModel()!, messages);
    expect((result.stopReason, result.errorMessage), (StopReason.error, 'Unknown provider: faux'));
  });

  test('drives an agent through a tool call', () async {
    final faux = FauxProvider(random: Random(1));
    final models = Models()..setProvider(faux.provider);
    faux.setResponses([
      fauxAssistantMessage([
        fauxToolCall('add', {'a': 2, 'b': 3}),
      ], stopReason: StopReason.toolUse),
      (List<Message> m, SimpleStreamOptions o, FauxState s, Model model) =>
          fauxAssistantMessage(((m.last as ToolResultMessage).content.single as TextContent).text),
    ]);
    final agent = Agent(
      model: faux.getModel()!,
      tools: [
        AgentTool(
          name: 'add',
          description: 'Adds two numbers.',
          parameters: const {
            'type': 'object',
            'properties': {
              'a': {'type': 'number'},
              'b': {'type': 'number'},
            },
            'required': ['a', 'b'],
          },
          execute: (id, args, signal, onUpdate) async {
            final json = args as Json;
            return AgentToolResult([TextContent('${(json['a'] as num) + (json['b'] as num)}')]);
          },
        ),
      ],
      stream: models.streamSimple,
    );
    await agent.promptText('What is 2 + 3?');
    expect(((agent.messages.last as AssistantMessage).content.single as TextContent).text, '5');
    expect(faux.state.callCount, 2);
  });
}
