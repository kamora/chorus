import 'dart:convert';

import 'package:chorus/chorus.dart';
import 'package:chorus/src/ai/transform.dart';
import 'package:test/test.dart';

import 'anthropic_fake.dart';

// Ported from pi's test/anthropic-sse-parsing.test.ts and test/anthropic-cache-write-1h-cost.test.ts. Left out with
// their features: OpenRouter's reported cost, fallback models, managed betas and input transformations.

Future<AssistantMessage> run(Model model, List<Message> messages, FakeServer server, [AnthropicOptions? options]) {
  return streamAnthropic(
    model,
    messages,
    AnthropicOptions(
      client: server.client(),
      onEvent: options?.onEvent,
      onPayload: options?.onPayload,
      thinkingEnabled: options?.thinkingEnabled,
    ),
  ).result;
}

Json start(String id, {String? model, Json usage = const {'input_tokens': 12, 'output_tokens': 0}}) {
  return {
    'type': 'message_start',
    'message': {'id': id, 'model': ?model, 'usage': usage},
  };
}

Json delta(String reason, [Json? usage, Json? details]) {
  return {
    'type': 'message_delta',
    'delta': {'stop_reason': reason, 'stop_details': ?details},
    'usage': ?usage,
  };
}

void main() {
  group('Anthropic raw SSE parsing', () {
    test('forwards parsed provider stream events in order', () async {
      final model = claude();
      final types = <Object?>[];
      final models = <Model>[];
      expect(
        (await streamAnthropic(
          model,
          [user('Hello')],
          AnthropicOptions(
            client: FakeServer(sse(minimalEvents())).client(),
            onEvent: (event, m) async {
              await Future<void>.value();
              types.add(event['type']);
              models.add(m);
            },
          ),
        ).result).stopReason,
        StopReason.stop,
      );
      expect(types, [
        'message_start',
        'content_block_start',
        'content_block_delta',
        'content_block_stop',
        'message_delta',
        'message_stop',
      ]);
      expect(models, everyElement(same(model)));
    });

    test('keeps signed thinking replayable when a proxy relabels the model', () async {
      final model = claude();
      final first = await run(
        model,
        [user('Hello')],
        FakeServer(
          sse([
            start('msg_response_model', model: 'kimi-for-coding', usage: {'input_tokens': 100, 'output_tokens': 0}),
            {
              'type': 'content_block_start',
              'index': 0,
              'content_block': {'type': 'thinking', 'thinking': 'reasoning', 'signature': 'signature'},
            },
            {'type': 'content_block_stop', 'index': 0},
            delta('end_turn', {'input_tokens': 100, 'output_tokens': 20}),
            {'type': 'message_stop'},
          ]),
        ),
      );

      expect(first.model, model.id);
      expect(first.responseModel, 'kimi-for-coding');
      expect(
        transformMessages([
          user('Hello'),
          first,
        ], model).whereType<AssistantMessage>().single.content.map((c) => c.toJson()),
        [
          {'type': 'thinking', 'thinking': 'reasoning', 'thinkingSignature': 'signature'},
        ],
      );
    });

    test('uses a returned fallback model for cost attribution', () async {
      final model = claude(
        id: 'claude-opus-5',
        compat: const {
          'allowedFallbackModels': [
            {
              'provider': 'anthropic',
              'model': 'fallback-model',
              'cost': {'input': 3, 'output': 5, 'cacheRead': 0, 'cacheWrite': 0},
            },
          ],
        },
      );
      final result = await run(
        model,
        [user('Hello')],
        FakeServer(
          sse([
            start('msg_response_model', model: 'fallback-model', usage: {'input_tokens': 100, 'output_tokens': 0}),
            {
              'type': 'content_block_start',
              'index': 0,
              'content_block': {'type': 'text', 'text': 'done'},
            },
            {'type': 'content_block_stop', 'index': 0},
            delta('end_turn', {'input_tokens': 100, 'output_tokens': 20}),
            {'type': 'message_stop'},
          ]),
        ),
      );

      expect(result.model, 'claude-opus-5');
      expect(result.responseModel, 'fallback-model');
      expect(result.usage.cost.input, closeTo(0.0003, 1e-10));
      expect(result.usage.cost.output, closeTo(0.0001, 1e-10));
    });

    test('bills its own prices when a model outside the fallback list answers', () async {
      // The answering model is listed, but for another provider.
      final result = await run(
        claude(
          id: 'claude-opus-5',
          compat: const {
            'allowedFallbackModels': [
              {
                'provider': 'other-provider',
                'model': 'another-model',
                'cost': {'input': 3, 'output': 5, 'cacheRead': 0, 'cacheWrite': 0},
              },
            ],
          },
        ),
        [user('Hello')],
        FakeServer(
          sse([
            start('msg', model: 'another-model', usage: {'input_tokens': 100, 'output_tokens': 0}),
            delta('end_turn', {'input_tokens': 100, 'output_tokens': 20}),
            {'type': 'message_stop'},
          ]),
        ),
      );

      // claude()'s prices: $5 input and $25 output per million tokens.
      expect(result.usage.cost.input, closeTo(0.0005, 1e-10));
      expect(result.usage.cost.output, closeTo(0.0005, 1e-10));
    });

    test('offers the allowed fallback models with their beta', () async {
      final server = FakeServer(sse(minimalEvents()));
      await run(getBuiltinModel('anthropic', 'claude-fable-5-2')!, [user('Hello')], server);
      expect(server.body['fallbacks'], [
        {'model': 'claude-opus-5'},
      ]);
      expect(server.requests.single.headers['anthropic-beta'], contains('server-side-fallback-2026-07-01'));

      final plain = FakeServer(sse(minimalEvents()));
      await run(claude(), [user('Hello')], plain);
      expect(plain.body.containsKey('fallbacks'), isFalse);
      expect(plain.requests.single.headers['anthropic-beta'] ?? '', isNot(contains('server-side-fallback')));
    });

    test('uses the serving model input transformations from the final stream event', () async {
      final events = minimalEvents();
      (events[0]['message'] as Json)['input_transformations'] = [
        {'type': 'thinking_dropped', 'path': 'messages.1.content.0', 'reason': 'prefix_binding_mismatch'},
      ];
      events[4]['input_transformations'] = [
        {'type': 'thinking_dropped', 'path': 'messages.3.content.0', 'reason': 'model_binding_mismatch'},
      ];
      final result = await run(claude(id: 'claude-fable-5-1'), [user('Hello')], FakeServer(sse(events)));
      expect(result.diagnostics, hasLength(1));
      expect(result.diagnostics!.single.type, 'anthropic_input_transformations');
      expect(result.diagnostics!.single.details, {
        'transformations': [
          {'type': 'thinking_dropped', 'path': 'messages.3.content.0', 'reason': 'model_binding_mismatch'},
        ],
      });

      final plain = await run(claude(), [user('Hello')], FakeServer(sse(minimalEvents())));
      expect(plain.diagnostics, isNull);
    });

    test('fails safely when Anthropic falls back after output begins', () async {
      final result = await run(
        claude(),
        [user('Hello')],
        FakeServer(
          sse([
            start('msg_fallback', model: 'claude-test', usage: {'input_tokens': 1, 'output_tokens': 0}),
            {
              'type': 'content_block_start',
              'index': 0,
              'content_block': {'type': 'text', 'text': 'partial'},
            },
            {'type': 'content_block_stop', 'index': 0},
            {
              'type': 'content_block_start',
              'index': 1,
              'content_block': {
                'type': 'fallback',
                'from': {'model': 'claude-test'},
                'to': {'model': 'claude-older'},
              },
            },
          ]),
        ),
      );

      expect(result.stopReason, StopReason.error);
      expect(result.errorMessage, contains('unsupported mid-output model fallback'));
    });

    test('forces streaming after an onPayload replacement', () async {
      final server = FakeServer(sse(minimalEvents()));
      await run(
        claude(),
        [user('Hello')],
        server,
        AnthropicOptions(onPayload: (payload, model) => {...payload, 'stream': false}),
      );

      expect(server.body['stream'], isTrue);
    });

    test('omits the interleaved-thinking beta when thinking is disabled', () async {
      final server = FakeServer(sse(minimalEvents()));
      await run(claude(), [user('Hello')], server, const AnthropicOptions(thinkingEnabled: false));

      expect(server.requests.single.headers['anthropic-beta'] ?? '', isNot(contains('interleaved-thinking')));
    });

    test('repairs malformed SSE JSON and malformed streamed tool JSON', () async {
      final result = await run(
        claude(),
        [
          SystemMessage(
            const [],
            added: const [
              Tool('edit', 'Edit a file.', {
                'type': 'object',
                'properties': {
                  'path': {'type': 'string'},
                  'text': {'type': 'string'},
                },
                'required': ['path', 'text'],
              }),
            ],
            timestamp: 0,
          ),
          user('Use the edit tool.'),
        ],
        FakeServer(
          [
            'event: message_start\ndata: ${jsonEncode(start('msg_test'))}\n',
            'event: content_block_start\ndata: ${jsonEncode({
              'type': 'content_block_start',
              'index': 0,
              'content_block': {'type': 'tool_use', 'id': 'toolu_test', 'name': 'edit', 'input': <String, Object?>{}},
            })}\n',

            // pi's raw event: an invalid `\H` escape and a raw tab inside the arguments' JSON text.
            'event: content_block_delta\ndata: '
                r'{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"path\":\"A\H\",\"text\":\"col1	col2\"}"}}'
                '\n',
            'event: content_block_stop\ndata: ${jsonEncode({'type': 'content_block_stop', 'index': 0})}\n',
            'event: message_delta\ndata: ${jsonEncode(delta('tool_use', {'input_tokens': 12, 'output_tokens': 5}))}\n',
            'event: message_stop\ndata: ${jsonEncode({'type': 'message_stop'})}\n',
          ].join('\n'),
        ),
      );

      expect(result.stopReason, StopReason.toolUse);
      expect(result.errorMessage, isNull);
      expect(result.content.whereType<ToolCall>().single.arguments, {'path': r'A\H', 'text': 'col1\tcol2'});
    });

    test('preserves content from content_block_start events', () async {
      expect(
        (await run(
          claude(),
          [user('Say hello.')],
          FakeServer(
            sse([
              start('msg_initial_content'),
              {
                'type': 'content_block_start',
                'index': 0,
                'content_block': {'type': 'text', 'text': 'Initial text'},
              },
              {
                'type': 'content_block_delta',
                'index': 0,
                'delta': {'type': 'text_delta', 'text': ' plus delta'},
              },
              {'type': 'content_block_stop', 'index': 0},
              {
                'type': 'content_block_start',
                'index': 1,
                'content_block': {'type': 'thinking', 'thinking': 'Initial thinking', 'signature': 'initial signature'},
              },
              {
                'type': 'content_block_delta',
                'index': 1,
                'delta': {'type': 'thinking_delta', 'thinking': ' plus delta'},
              },
              {
                'type': 'content_block_delta',
                'index': 1,
                'delta': {'type': 'signature_delta', 'signature': ' plus delta'},
              },
              {'type': 'content_block_stop', 'index': 1},
              delta('end_turn', {'input_tokens': 12, 'output_tokens': 5}),
              {'type': 'message_stop'},
            ]),
          ),
        )).content.map((c) => c.toJson()),
        [
          {'type': 'text', 'text': 'Initial text plus delta'},
          {
            'type': 'thinking',
            'thinking': 'Initial thinking plus delta',
            'thinkingSignature': 'initial signature plus delta',
          },
        ],
      );
    });

    test('preserves refusal stop details from message_delta', () async {
      const explanation = 'This request triggered restrictions on violative cyber content.';
      final result = await run(
        claude(),
        [user('blocked request')],
        FakeServer(
          sse([
            start('msg_refusal', usage: {'input_tokens': 412, 'output_tokens': 0}),
            delta(
              'refusal',
              {'input_tokens': 412, 'output_tokens': 0},
              {'type': 'refusal', 'category': 'cyber', 'explanation': explanation},
            ),
            {'type': 'message_stop'},
          ]),
        ),
      );

      expect(result.stopReason, StopReason.error);
      expect(result.rawStopReason, 'refusal');
      expect(result.errorMessage, explanation);
    });

    test('preserves sensitive stop reasons with a descriptive error message', () async {
      final result = await run(
        claude(),
        [user('blocked request')],
        FakeServer(
          sse([
            start('msg_sensitive'),
            delta('sensitive', {'output_tokens': 0}),
            {'type': 'message_stop'},
          ]),
        ),
      );

      expect(result.stopReason, StopReason.error);
      expect(result.rawStopReason, 'sensitive');
      expect(result.errorMessage, 'Provider stopped with: sensitive');
    });

    test('treats message_delta without usage as a no-op for usage accumulation', () async {
      final events = minimalEvents();
      events[4] = delta('end_turn');
      final result = await run(claude(), [user('Say hello.')], FakeServer(sse(events)));

      expect(result.stopReason, StopReason.stop);
      expect(result.errorMessage, isNull);
      expect(result.content.map((c) => c.toJson()), [
        {'type': 'text', 'text': 'Hello'},
      ]);
      expect(result.usage.input, 12);
      expect(result.usage.totalTokens, 12);
    });

    test('ignores unknown SSE events after message_stop', () async {
      final result = await run(claude(), [
        user('Say hello.'),
      ], FakeServer('${sse(minimalEvents())}\nevent: done\ndata: [DONE]\n\nevent: proxy.stats\ndata: not json\n'));

      expect(result.stopReason, StopReason.stop);
      expect(result.errorMessage, isNull);
      expect(result.content.map((c) => c.toJson()), [
        {'type': 'text', 'text': 'Hello'},
      ]);
    });
  });

  // pi's test/anthropic-cache-write-1h-cost.test.ts. The model is priced like Claude Opus 4.8: $5 input and $6.25
  // 5m cache write per million tokens, so a 1h write costs $10.
  group('Anthropic 1h cache write cost', () {
    List<Json> events(Json? creation) {
      return [
        start(
          'msg_test',
          usage: {
            'input_tokens': 100,
            'output_tokens': 0,
            'cache_read_input_tokens': 0,
            'cache_creation_input_tokens': 1000000,
            'cache_creation': ?creation,
          },
        ),
        {
          'type': 'content_block_start',
          'index': 0,
          'content_block': {'type': 'text', 'text': ''},
        },
        {
          'type': 'content_block_delta',
          'index': 0,
          'delta': {'type': 'text_delta', 'text': 'Hi'},
        },
        {'type': 'content_block_stop', 'index': 0},
        delta('end_turn', {
          'input_tokens': 100,
          'output_tokens': 5,
          'cache_read_input_tokens': 0,
          'cache_creation_input_tokens': 1000000,
        }),
        {'type': 'message_stop'},
      ];
    }

    test('prices the 1h portion at 2x input and the rest at the 5m rate', () async {
      final result = await run(claude(), [
        user('hi'),
      ], FakeServer(sse(events({'ephemeral_5m_input_tokens': 600000, 'ephemeral_1h_input_tokens': 400000}))));

      expect(result.usage.cacheWrite, 1000000);
      expect(result.usage.cacheWrite1h, 400000);
      expect(result.usage.cost.cacheWrite, closeTo(7.75, 1e-10));
    });

    // pi's regression for #9210: Vercel AI Gateway sends cache usage in message_delta, not message_start.
    test('prices 1h cache writes reported only in message_delta', () async {
      final result = await run(
        claude(),
        [user('hi')],
        FakeServer(
          sse([
            start('msg_test', usage: {'input_tokens': 0, 'output_tokens': 0}),
            delta('end_turn', {
              'input_tokens': 3,
              'output_tokens': 4,
              'cache_creation_input_tokens': 6535,
              'cache_creation': {'ephemeral_5m_input_tokens': 0, 'ephemeral_1h_input_tokens': 6535},
            }),
            {'type': 'message_stop'},
          ]),
        ),
      );

      expect(result.usage.cacheWrite, 6535);
      expect(result.usage.cacheWrite1h, 6535);
      expect(result.usage.cost.cacheWrite, closeTo(6535 * 5 * 2 / 1e6, 1e-10));
    });

    test('falls back to the 5m rate when no breakdown is reported', () async {
      final result = await run(claude(), [user('hi')], FakeServer(sse(events(null))));

      expect(result.usage.cacheWrite, 1000000);
      expect(result.usage.cacheWrite1h ?? 0, 0);
      expect(result.usage.cost.cacheWrite, closeTo(6.25, 1e-10));
    });
  });
}
