// Ported from pi's packages/ai/test: openai-responses-terminal-event, -namespace, -partial-json-cleanup,
// -prompt-cache, -compat and -mid-convo-effort. The SDK and fetch mocks become http's MockClient answering with the
// events as server-sent events. Not ported: custom (grammar) tool calls and Sign in with ChatGPT, which chorus leaves
// out, and models missing from chorus's catalog are built in code.

import 'dart:convert';

import 'package:chorus/chorus.dart';
import 'package:chorus/src/ai/providers/openai_responses.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:test/test.dart';

const _free = ModelCost(input: 0, output: 0, cacheRead: 0, cacheWrite: 0);

/// A model built in code, as pi's tests build theirs.
Model model({
  String id = 'gpt-5-mini',
  String provider = 'openai',
  String baseUrl = 'https://api.openai.com/v1',
  bool reasoning = true,
  Map<String, String?>? levels,
  Json compat = const {},
}) {
  return Model(
    id: id,
    name: id,
    api: 'openai-responses',
    provider: provider,
    baseUrl: baseUrl,
    reasoning: reasoning,
    levels: levels,
    cost: _free,
    contextWindow: 400000,
    maxTokens: 128000,
    compat: compat,
  );
}

/// [base] with the given fields replaced.
Model like(Model base, {String? id, String? provider, String? baseUrl, String? api, Json? compat}) {
  return Model(
    id: id ?? base.id,
    name: base.name,
    api: api ?? base.api,
    provider: provider ?? base.provider,
    baseUrl: baseUrl ?? base.baseUrl,
    input: base.input,
    cost: base.cost,
    headers: base.headers,
    reasoning: base.reasoning,
    levels: base.levels,
    contextWindow: base.contextWindow,
    maxTokens: base.maxTokens,
    compat: compat ?? base.compat,
  );
}

Model catalog(String id) => getBuiltinModel('openai', id)!;

AssistantMessage output(Model m) {
  return AssistantMessage(api: m.api, provider: m.provider, model: m.id, timestamp: 0);
}

UserMessage user(String text) => UserMessage([TextContent(text)], timestamp: 0);

final hi = <Message>[
  SystemMessage([TextContent('sys')], timestamp: 0),
  user('hi'),
];

String sse(List<Json> events) => events.map((e) => 'data: ${jsonEncode(e)}\n\n').join();

/// Records each event and the stop reason its partial had when it was pushed.
class Recorder extends AssistantMessageEventStream {
  final events = <AssistantMessageEvent>[];
  final stopReasons = <StopReason>[];

  @override
  void push(AssistantMessageEvent event) {
    events.add(event);
    if (event case BlockEvent(:final partial) || StartEvent(:final partial)) {
      stopReasons.add(partial.stopReason);
    }

    super.push(event);
  }
}

Future<void> process(List<Json> events, AssistantMessage out, Model m, [AssistantMessageEventStream? stream]) {
  return processResponsesStream(Stream.fromIterable(events), out, stream ?? AssistantMessageEventStream(), m);
}

/// A request through the provider, answered with [body]: the sent body and headers, and the result.
Future<({Json payload, Map<String, String> headers, AssistantMessage message})> run(
  Model m,
  List<Message> messages, {
  String body = 'data: [DONE]\n\n',
  OpenAIResponsesOptions Function(http.Client client)? options,
}) async {
  late Json payload;
  late Map<String, String> headers;
  final client = MockClient((request) async {
    payload = jsonDecode(request.body) as Json;
    headers = request.headers;
    return http.Response(body, 200, headers: {'content-type': 'text/event-stream'});
  });
  final message = await streamOpenAIResponses(
    m,
    messages,
    options?.call(client) ?? OpenAIResponsesOptions(apiKey: 'sk-test-key', httpClient: client),
  ).result;
  return (payload: payload, headers: headers, message: message);
}

List<Json> functionCall({String? namespace, String arguments = '{"value":"hello"}', bool index = true}) {
  return [
    {
      'type': 'response.output_item.added',
      if (index) 'output_index': 0,
      'item': {'type': 'function_call', 'id': 'fc_test', 'call_id': 'call_test', 'name': 'lookup', 'arguments': ''},
    },
    {
      'type': 'response.output_item.done',
      if (index) 'output_index': 0,
      'item': {
        'type': 'function_call',
        'id': 'fc_test',
        'call_id': 'call_test',
        'name': 'lookup',
        'arguments': arguments,
        'namespace': ?namespace,
      },
    },
    {
      'type': 'response.completed',
      'response': {'id': 'resp_test', 'status': 'completed'},
    },
  ];
}

void main() {
  group('terminal event handling', () {
    final m = model();
    final earlyEof = <Json>[
      {
        'type': 'response.created',
        'response': {'id': 'resp_early_eof'},
      },
      {
        'type': 'response.output_item.added',
        'output_index': 0,
        'item': {'type': 'reasoning', 'id': 'rs_early_eof', 'summary': <Object?>[]},
      },
      {
        'type': 'response.reasoning_text.delta',
        'output_index': 0,
        'content_index': 0,
        'item_id': 'rs_early_eof',
        'delta': 'partial reasoning before the stream ends',
      },
    ];

    List<Json> incomplete([String reason = 'max_output_tokens']) {
      return [
        {
          'type': 'response.incomplete',
          'response': {
            'id': 'resp_incomplete',
            'status': 'incomplete',
            'incomplete_details': {'reason': reason},
            'usage': {
              'input_tokens': 30,
              'output_tokens': 12,
              'total_tokens': 42,
              'input_tokens_details': {'cached_tokens': 5},
            },
          },
        },
      ];
    }

    List<Json> phased(String first, String second, [String terminal = 'completed']) {
      Json item(String status, String phase, List<Object?> content) {
        return {
          'type': 'message',
          'id': 'msg_phase',
          'role': 'assistant',
          'status': status,
          'content': content,
          'phase': phase,
        };
      }

      return [
        {'type': 'response.output_item.added', 'output_index': 0, 'item': item('in_progress', first, [])},
        {
          'type': 'response.output_item.done',
          'output_index': 0,
          'item': item('completed', second, [
            {'type': 'output_text', 'text': 'answer', 'annotations': <Object?>[]},
          ]),
        },
        if (terminal == 'incomplete')
          {
            'type': 'response.incomplete',
            'response': {
              'id': 'resp_phase',
              'status': 'incomplete',
              'incomplete_details': {'reason': 'max_output_tokens'},
            },
          }
        else
          {
            'type': 'response.completed',
            'response': {'id': 'resp_phase', 'status': 'completed'},
          },
      ];
    }

    test('rejects streams that end before a terminal response event', () async {
      await expectLater(
        process(earlyEof, output(m), m),
        throwsA(
          isA<ProviderException>().having(
            (e) => e.message,
            'message',
            'OpenAI Responses stream ended before a terminal response event',
          ),
        ),
      );
    });

    test('rejects completed streams whose tool call never received output_item.done', () async {
      await expectLater(
        process(
          [
            {
              'type': 'response.output_item.added',
              'output_index': 0,
              'item': {'type': 'function_call', 'id': 'fc_1', 'call_id': 'call_1', 'name': 'bash', 'arguments': ''},
            },
            {
              'type': 'response.function_call_arguments.delta',
              'output_index': 0,
              'item_id': 'fc_1',
              'delta': '{"command":"rm -rf /tmp/build',
            },
            {
              'type': 'response.completed',
              'response': {'id': 'resp_unfinished', 'status': 'completed'},
            },
          ],
          output(m),
          m,
        ),
        throwsA(
          isA<ProviderException>().having(
            (e) => e.message,
            'message',
            'OpenAI Responses stream completed with an unfinished tool call: bash (call_1|fc_1)',
          ),
        ),
      );
    });

    // llama.cpp leaves output_index out of every event and sends both done events after all deltas.
    test('rejects parallel tool calls without output_index instead of running mixed-up calls', () async {
      Json call(String n, [String arguments = '']) {
        return {'type': 'function_call', 'id': 'fc_$n', 'call_id': 'call_$n', 'name': 'bash', 'arguments': arguments};
      }

      await expectLater(
        process(
          [
            {'type': 'response.output_item.added', 'item': call('a')},
            {'type': 'response.function_call_arguments.delta', 'item_id': 'fc_a', 'delta': '{"command":"echo a"}'},
            {'type': 'response.output_item.added', 'item': call('b')},
            {'type': 'response.function_call_arguments.delta', 'item_id': 'fc_b', 'delta': '{"command":"echo b"}'},
            {'type': 'response.output_item.done', 'item': call('a', '{"command":"echo a"}')},
            {'type': 'response.output_item.done', 'item': call('b', '{"command":"echo b"}')},
            {
              'type': 'response.completed',
              'response': {'id': 'resp_no_output_index', 'status': 'completed'},
            },
          ],
          output(m),
          m,
        ),
        throwsA(
          isA<ProviderException>().having(
            (e) => e.message,
            'message',
            'OpenAI Responses stream completed with an unfinished tool call: bash (call_a|fc_a)',
          ),
        ),
      );
    });

    test('forwards parsed provider stream events in order', () async {
      final seen = <String>[];
      final models = <Model>[];
      await run(
        m,
        [user('hi')],
        body: sse(earlyEof),
        options: (client) => OpenAIResponsesOptions(
          apiKey: 'test',
          httpClient: client,
          onEvent: (event, eventModel) async {
            await Future<void>.value();
            seen.add(event['type'] as String);
            models.add(eventModel);
          },
        ),
      );
      expect(seen, ['response.created', 'response.output_item.added', 'response.reasoning_text.delta']);
      expect(models, [same(m), same(m), same(m)]);
    });

    test('emits an error final result when the stream ends before a terminal response event', () async {
      final events = <AssistantMessageEvent>[];
      StopReason? initial;
      final client = MockClient(
        (_) async => http.Response(sse(earlyEof), 200, headers: {'content-type': 'text/event-stream'}),
      );
      final stream = streamOpenAIResponses(m, [user('hi')], OpenAIResponsesOptions(apiKey: 'test', httpClient: client));
      await for (final event in stream) {
        if (event is StartEvent) {
          initial = event.partial.stopReason;
        }

        events.add(event);
      }

      final result = await stream.result;
      expect(initial, StopReason.pending);
      expect(events.last, isA<ErrorEvent>());
      expect(result.stopReason, StopReason.error);
      expect(result.errorMessage, 'OpenAI Responses stream ended before a terminal response event');
    });

    for (final (phases, expected) in [
      (('commentary', 'commentary'), [StopReason.pending, StopReason.pending]),
      (('final_answer', 'final_answer'), [StopReason.stop, StopReason.stop]),
      (('commentary', 'final_answer'), [StopReason.pending, StopReason.stop]),
    ]) {
      test('tracks message phases $phases', () async {
        final out = output(m);
        final recorder = Recorder();
        await process(phased(phases.$1, phases.$2), out, m, recorder);
        expect(recorder.stopReasons, expected);
        expect(out.stopReason, StopReason.stop);
      });
    }

    test('replaces a provisional final-answer stop with an incomplete terminal reason', () async {
      final out = output(m);
      final recorder = Recorder();
      await process(phased('final_answer', 'final_answer', 'incomplete'), out, m, recorder);
      expect(recorder.stopReasons, [StopReason.stop, StopReason.stop]);
      expect(out.stopReason, StopReason.length);
    });

    test('finalizes completed terminal events as stop', () async {
      final out = output(m);
      await process(
        [
          {
            'type': 'response.completed',
            'response': {
              'id': 'resp_completed',
              'status': 'completed',
              'usage': {
                'input_tokens': 20,
                'output_tokens': 7,
                'total_tokens': 27,
                'input_tokens_details': {'cached_tokens': 2, 'cache_write_tokens': 3},
              },
            },
          },
        ],
        out,
        m,
      );
      expect((out.responseId, out.stopReason, out.rawStopReason), ('resp_completed', StopReason.stop, 'completed'));
      final u = out.usage;
      expect((u.input, u.output, u.cacheRead, u.cacheWrite, u.totalTokens), (15, 7, 2, 3, 27));
    });

    test('finalizes incomplete terminal events as length stops', () async {
      final out = output(m);
      await process(incomplete(), out, m);
      expect(
        (out.responseId, out.stopReason, out.rawStopReason),
        ('resp_incomplete', StopReason.length, 'incomplete.max_output_tokens'),
      );
      final u = out.usage;
      expect((u.input, u.output, u.cacheRead, u.cacheWrite, u.totalTokens), (25, 12, 5, 0, 42));
    });

    for (final reason in ['content_filter', 'max_time_limit']) {
      test('finalizes $reason incomplete responses as non-retryable errors', () async {
        final out = output(m);
        await process(incomplete(reason), out, m);
        expect(
          (out.stopReason, out.rawStopReason, out.errorMessage),
          (StopReason.error, 'incomplete.$reason', 'Response incomplete: $reason'),
        );
      });
    }

    test('rejects failed terminal events with the provider error', () async {
      final out = output(m);
      await expectLater(
        process(
          [
            {
              'type': 'response.failed',
              'response': {
                'id': 'resp_failed',
                'status': 'failed',
                'error': {'code': 'server_error', 'message': 'boom'},
              },
            },
          ],
          out,
          m,
        ),
        throwsA(isA<ProviderException>().having((e) => e.message, 'message', 'server_error: boom')),
      );
      expect(out.rawStopReason, 'failed');
    });
  });

  group('tool-call namespaces', () {
    final m = model(id: 'gpt-5.4');

    test('omits an absent error message', () async {
      final out = output(m);
      await process(functionCall(namespace: 'dynamic_tools'), out, m);
      expect(out.errorMessage, isNull);
      expect(out.toJson().containsKey('errorMessage'), isFalse);
    });

    test('round-trips a function namespace received only on output_item.done', () async {
      final out = output(m);
      await process(functionCall(namespace: 'dynamic_tools'), out, m);
      final call = out.content.single as ToolCall;
      expect((call.id, call.name, call.namespace), ('call_test|fc_test', 'lookup', 'dynamic_tools'));
      expect(call.arguments, {'value': 'hello'});
      expect(convertResponsesMessages(m, [out]).singleWhere((i) => i['type'] == 'function_call'), {
        'type': 'function_call',
        'id': 'fc_test',
        'call_id': 'call_test',
        'name': 'lookup',
        'arguments': '{"value":"hello"}',
        'namespace': 'dynamic_tools',
      });
    });

    test('drops namespaces when the target cannot replay their load items', () {
      final out = output(m)
        ..content.add(ToolCall('call_function|fc_test', 'lookup', {'value': 'hello'}, namespace: 'dynamic_tools'));
      for (final target in [
        like(m, id: 'gpt-5.2'),
        like(m, provider: 'azure'),
        like(m, api: 'openai-codex-responses', provider: 'openai-codex', id: 'gpt-5.3-codex-spark'),
      ]) {
        final call = convertResponsesMessages(target, [out]).singleWhere((i) => i['type'] == 'function_call');
        expect(call.containsKey('namespace'), isFalse, reason: target.id);
      }
    });

    test('does not add a namespace to ordinary function calls', () {
      final out = output(m)..content.add(ToolCall('call_test|fc_test', 'lookup', {'value': 'hello'}));
      final call = convertResponsesMessages(m, [out]).singleWhere((i) => i['type'] == 'function_call');
      expect(call.containsKey('namespace'), isFalse);
    });
  });

  group('partial JSON cleanup', () {
    final m = model();

    List<Json> edit(String arguments) {
      return [
        {
          'type': 'response.output_item.added',
          'item': {'type': 'function_call', 'id': 'fc_test', 'call_id': 'call_test', 'name': 'edit', 'arguments': ''},
        },
        {'type': 'response.function_call_arguments.delta', 'delta': '{"path":"README.md"'},
        {'type': 'response.function_call_arguments.delta', 'delta': ',"content":"updated"}'},
        {'type': 'response.function_call_arguments.done', 'arguments': arguments},
        {
          'type': 'response.output_item.done',
          'item': {
            'type': 'function_call',
            'id': 'fc_test',
            'call_id': 'call_test',
            'name': 'edit',
            'arguments': arguments,
          },
        },
        {
          'type': 'response.completed',
          'response': {'id': 'resp_test', 'status': 'completed'},
        },
      ];
    }

    test('the persisted tool call is the one the end event carries, without stream state', () async {
      final out = output(m);
      final recorder = Recorder();
      await process(edit('{"path":"README.md","content":"updated"}'), out, m, recorder);
      final call = out.content.single as ToolCall;
      expect(call.arguments, {'path': 'README.md', 'content': 'updated'});
      expect(call.toJson().containsKey('partialJson'), isFalse);
      expect(recorder.events.whereType<ToolCallEndEvent>().single.call, same(call));
    });

    test('marks malformed final arguments instead of keeping the parsed prefix', () async {
      final out = output(m);
      const arguments = '{"path":"README.md","content":"updated","mode":.5}';
      await process(edit(arguments), out, m);
      final call = out.content.single as ToolCall;
      expect(call.arguments, isEmpty);
      expect(call.argumentsError, endsWith('Received arguments:\n$arguments'));
    });
  });

  group('prompt cache key policy', () {
    Future<Json> payload([String? mode, CacheRetention? retention]) async {
      final m = model(
        id: 'test-model',
        provider: 'test-proxy',
        baseUrl: 'https://proxy.example.com/v1',
        reasoning: false,
        compat: {'promptCacheKeyMode': ?mode},
      );
      return (await run(
        m,
        [user('hello')],
        body: sse([
          {
            'type': 'response.completed',
            'response': {
              'status': 'completed',
              'usage': {
                'input_tokens': 1,
                'output_tokens': 1,
                'total_tokens': 2,
                'input_tokens_details': {'cached_tokens': 0},
              },
            },
          },
        ]),
        options: (client) => OpenAIResponsesOptions(
          apiKey: 'test-key',
          httpClient: client,
          sessionId: 'session-123',
          cacheRetention: retention,
        ),
      )).payload;
    }

    test('sends the session by default', () async {
      expect((await payload())['prompt_cache_key'], 'session-123');
    });

    test('sends the key when enabled', () async {
      expect((await payload('enabled'))['prompt_cache_key'], 'session-123');
    });

    test('omits the key when disabled', () async {
      expect((await payload('disabled')).containsKey('prompt_cache_key'), isFalse);
    });

    test('lets cacheRetention none override enabled mode', () async {
      expect((await payload('enabled', CacheRetention.none)).containsKey('prompt_cache_key'), isFalse);
    });
  });

  group('provider defaults', () {
    Future<Map<String, String>> headers({
      Model? m,
      String? sessionId,
      CacheRetention? retention,
      Map<String, String?>? extra,
    }) async {
      return (await run(
        m ?? catalog('gpt-5.4'),
        hi,
        options: (client) => OpenAIResponsesOptions(
          apiKey: 'sk-test-key',
          httpClient: client,
          sessionId: sessionId,
          cacheRetention: retention,
          headers: extra,
        ),
      )).headers;
    }

    // Copilot rejected reasoning.effort "none".
    test('omits reasoning when no reasoning is requested on Copilot', () async {
      final m = model(id: 'gpt-6-sol', provider: 'github-copilot', levels: {'off': 'none'});
      expect((await run(m, hi)).payload.containsKey('reasoning'), isFalse);
    });

    test('forwards required tool choice', () async {
      final payload = (await run(
        catalog('gpt-5.6-luna'),
        [
          SystemMessage(
            [],
            added: const [
              Tool('ping', 'Ping', {
                'type': 'object',
                'properties': {
                  'value': {'type': 'string'},
                },
                'required': ['value'],
              }),
            ],
            timestamp: 0,
          ),
          user('Do not call ping. Respond with text instead.'),
        ],
        options: (client) => OpenAIResponsesOptions(apiKey: 'sk-test-key', httpClient: client, toolChoice: 'required'),
      )).payload;

      // gpt-5.6-luna declares its tools in additional_tools items; tool_choice applies to them.
      expect(payload['tool_choice'], 'required');
      final input = payload['input'] as List<Object?>;
      expect(input, hasLength(2));
      expect((input.first as Json)['type'], 'additional_tools');
      expect([for (final t in (input.first as Json)['tools'] as List<Object?>) (t as Json)['name']], ['ping']);
      expect(payload.containsKey('tools'), isFalse);
    });

    test('sets strict mode explicitly for tools', () async {
      // The Cloudflare AI Gateway's model: strict mode, but tools in the top-level list.
      final m = model(id: 'gpt-5.6-sol', provider: 'cloudflare-ai-gateway', compat: {'supportsStrictMode': true});
      final payload = (await run(m, [
        SystemMessage(
          [],
          added: const [
            Tool('ordinary', 'An ordinary tool', {
              'type': 'object',
              'properties': {
                'path': {'type': 'string'},
                'offset': {'type': 'number'},
              },
              'required': ['path'],
            }),
            Tool('constrained', 'A constrained tool', {
              'type': 'object',
              'properties': {
                'value': {'type': 'string'},
              },
              'required': ['value'],
            }, constrainedSampling: JsonSchemaSampling(Strictness.prefer)),
          ],
          timestamp: 0,
        ),
        user('Use a tool.'),
      ])).payload;
      expect(m.compat['supportsStrictMode'], isTrue);
      expect(
        [for (final t in payload['tools'] as List<Object?>) ((t as Json)['name'], t['strict'])],
        [('ordinary', false), ('constrained', true)],
      );
    });

    for (final id in ['gpt-5.1', 'gpt-5.2', 'gpt-5.3-codex', 'gpt-5.4', 'gpt-5.4-mini', 'gpt-5.4-nano', 'gpt-5.5']) {
      test('sends none reasoning effort for $id when no reasoning is requested', () async {
        expect((await run(catalog(id), hi)).payload['reasoning'], {'effort': 'none'});
      });
    }

    for (final id in ['gpt-5', 'gpt-5-mini', 'gpt-5-nano', 'gpt-5-pro', 'gpt-5.2-pro', 'gpt-5.4-pro', 'gpt-5.5-pro']) {
      test('omits reasoning effort for $id when off is unsupported', () async {
        expect((await run(catalog(id), hi)).payload.containsKey('reasoning'), isFalse);
      });
    }

    test('sets cache-affinity headers for OpenAI requests with a session', () async {
      final h = await headers(sessionId: 'session-123');
      expect((h['session-id'], h['session_id'], h['x-client-request-id']), ('session-123', null, 'session-123'));
    });

    test("clamps prompt_cache_key to OpenAI's 64-character limit", () async {
      final payload = (await run(
        catalog('gpt-5.4'),
        hi,
        options: (client) => OpenAIResponsesOptions(apiKey: 'sk-test-key', httpClient: client, sessionId: 'x' * 67),
      )).payload;
      expect(payload['prompt_cache_key'], 'x' * 64);
    });

    test('sets cache-affinity headers for proxy requests with a session', () async {
      final h = await headers(
        m: like(catalog('gpt-5.4'), provider: 'opencode', baseUrl: 'https://proxy.example.com/v1'),
        sessionId: 'session-123',
      );
      expect((h['session-id'], h['x-client-request-id']), ('session-123', 'session-123'));
    });

    for (final (name, m) in [
      (
        'configured',
        like(
          catalog('gpt-5.4'),
          provider: 'proxy',
          baseUrl: 'https://proxy.example.com/v1',
          compat: {'sessionAffinityFormat': 'openrouter'},
        ),
      ),
      ('detected', like(catalog('gpt-5.4'), provider: 'openrouter', baseUrl: 'https://openrouter.ai/api/v1')),
    ]) {
      test('uses the OpenRouter session header when $name', () async {
        late Json payload;
        late Map<String, String> h;
        (:payload, headers: h, message: _) = await run(
          m,
          hi,
          options: (client) =>
              OpenAIResponsesOptions(apiKey: 'sk-test-key', httpClient: client, sessionId: 'session-x'),
        );
        expect((h['session-id'], h['x-client-request-id'], h['x-session-id']), (null, null, 'session-x'));
        expect(payload.containsKey('session_id'), isFalse);
        expect(payload['prompt_cache_key'], 'session-x');
      });
    }

    test('uses the OpenAI no-session format when configured', () async {
      final m = like(
        catalog('gpt-5.4'),
        provider: 'proxy',
        baseUrl: 'https://proxy.example.com/v1',
        compat: {'sessionAffinityFormat': 'openai-nosession'},
      );
      final (:payload, headers: h, message: _) = await run(
        m,
        hi,
        options: (client) => OpenAIResponsesOptions(apiKey: 'sk-test-key', httpClient: client, sessionId: 'session-p'),
      );
      expect((h['session-id'], h['x-client-request-id'], h['x-session-id']), (null, 'session-p', null));
      expect(payload['prompt_cache_key'], 'session-p');
    });

    test('lets explicit headers override the default cache-affinity headers', () async {
      final h = await headers(
        sessionId: 'session-123',
        extra: {'session-id': 'override-session', 'X-Client-Request-Id': 'override-request'},
      );
      expect((h['session-id'], h['x-client-request-id']), ('override-session', 'override-request'));
    });

    test('a null header removes a default one', () async {
      final h = await headers(sessionId: 'session-123', extra: {'session-id': null});
      expect((h['session-id'], h['x-client-request-id']), (null, 'session-123'));
    });

    test('omits cache-affinity headers when cacheRetention is none', () async {
      final h = await headers(sessionId: 'session-123', retention: CacheRetention.none);
      expect((h['session-id'], h['x-client-request-id']), (null, null));
    });

    for (final (id, tier, returned, multiplier) in [
      ('gpt-5.4', 'priority', 'priority', 2.0),
      ('gpt-5.5', 'priority', 'priority', 2.5),
      ('gpt-5.5', 'flex', 'flex', 0.5),

      // GPT-6 models report Fast mode as "fast" even when "priority" is asked for.
      ('gpt-6-luna', 'priority', 'fast', 2.0),
      ('gpt-6-luna', 'fast', 'fast', 2.0),
    ]) {
      test('applies the $id cost multiplier for $tier answered as $returned', () async {
        final m = catalog(id);
        const tokens = 100000;
        final result = (await run(
          m,
          hi,
          body: sse([
            {
              'type': 'response.completed',
              'response': {
                'status': 'completed',
                'service_tier': returned,
                'usage': {
                  'input_tokens': tokens,
                  'output_tokens': tokens,
                  'total_tokens': tokens * 2,
                  'input_tokens_details': {'cached_tokens': 0},
                },
              },
            },
          ]),
          options: (client) => OpenAIResponsesOptions(apiKey: 'sk-test-key', httpClient: client, serviceTier: tier),
        )).message;
        const scale = tokens / 1e6;
        final cost = result.usage.cost;
        expect(cost.input, closeTo(m.cost.input * multiplier * scale, 1e-12));
        expect(cost.output, closeTo(m.cost.output * multiplier * scale, 1e-12));
        expect(cost.total, closeTo((m.cost.input + m.cost.output) * multiplier * scale, 1e-12));
      });
    }

    test('sends max_output_tokens by default, and not when the model does not take it', () async {
      Future<Json> payload(Model m) async {
        return (await run(
          m,
          hi,
          options: (client) => OpenAIResponsesOptions(apiKey: 'sk-test-key', httpClient: client, maxTokens: 1024),
        )).payload;
      }

      final m = catalog('gpt-5.4');
      expect((await payload(m))['max_output_tokens'], 1024);
      final without = like(m, compat: {...m.compat, 'supportsMaxOutputTokens': false});
      expect((await payload(without)).containsKey('max_output_tokens'), isFalse);
    });
  });

  group('configuration_update', () {
    const levels = {
      'off': 'none',
      'minimal': null,
      'low': 'low',
      'medium': 'medium',
      'high': 'high',
      'xhigh': 'xhigh',
      'max': 'max',
    };

    Model luna([Json compat = const {'supportsMidConvoEffort': true}]) {
      return model(id: 'gpt-6-luna', levels: levels, compat: compat);
    }

    Json update(String effort) => {
      'type': 'configuration_update',
      'reasoning': {'effort': effort},
    };

    List<Json> answer(bool toolCall) {
      final item = toolCall
          ? {
              'type': 'function_call',
              'id': 'fc_1',
              'call_id': 'call_1',
              'name': 'lookup',
              'arguments': '{}',
              'status': 'completed',
            }
          : {
              'type': 'message',
              'id': 'msg_resp_1',
              'role': 'assistant',
              'status': 'completed',
              'content': [
                {'type': 'output_text', 'text': 'answer', 'annotations': <Object?>[]},
              ],
            };
      return [
        {
          'type': 'response.created',
          'response': {'id': 'resp_1'},
        },
        {'type': 'response.output_item.added', 'output_index': 0, 'item': item},
        {'type': 'response.output_item.done', 'output_index': 0, 'item': item},
        {
          'type': 'response.completed',
          'response': {'id': 'resp_1', 'status': 'completed'},
        },
      ];
    }

    Future<({Json payload, AssistantMessage message})> ask(
      List<Message> messages,
      String? effort, {
      Model? m,
      bool toolCall = false,
    }) async {
      final r = await run(
        m ?? luna(),
        messages,
        body: sse(answer(toolCall)),
        options: (client) => OpenAIResponsesOptions(apiKey: 'sk-test', httpClient: client, reasoningEffort: effort),
      );
      expect(r.message.stopReason, isNot(StopReason.error), reason: r.message.errorMessage);
      return (payload: r.payload, message: r.message);
    }

    List<Object?> input(Json payload) => payload['input'] as List<Object?>;

    test('keeps reasoning.effort constant and only appends input when the level changes', () async {
      final history = <Message>[user('one')];
      final first = await ask(history, 'high');
      history.addAll([first.message, user('two')]);
      final second = await ask(history, 'high');
      history.addAll([second.message, user('three')]);
      final third = await ask(history, 'low');
      expect(input(first.payload), [
        {
          'role': 'user',
          'content': [
            {'type': 'input_text', 'text': 'one'},
          ],
        },
        update('high'),
      ]);
      for (final r in [first, second, third]) {
        expect((r.payload['reasoning'] as Json)['effort'], 'medium');
      }

      expect(input(second.payload).take(input(first.payload).length), input(first.payload));
      expect(input(third.payload).take(input(second.payload).length), input(second.payload));
      expect(input(third.payload).where((i) => (i as Json)['type'] == 'configuration_update'), [
        update('high'),
        update('high'),
        update('low'),
      ]);
      expect(input(third.payload).last, update('low'));
      expect(
        [
          for (final r in [first, second, third]) r.message.providerThinkingLevel,
        ],
        ['high', 'high', 'low'],
      );
    });

    test('applies a level change made during a tool loop to the next reply', () async {
      final history = <Message>[user('one')];
      final call = await ask(history, 'high', toolCall: true);
      history.addAll([
        call.message,
        ToolResultMessage(
          id: 'call_1|fc_1',
          name: 'lookup',
          content: [TextContent('result')],
          isError: false,
          timestamp: 0,
        ),
      ]);
      final after = await ask(history, 'low');
      expect(input(after.payload).take(input(call.payload).length), input(call.payload));
      final tail = input(after.payload).skip(input(after.payload).length - 2).toList();
      expect(((tail.first as Json)['type'], (tail.first as Json)['call_id']), ('function_call_output', 'call_1'));
      expect(tail.last, update('low'));
      expect(after.message.providerThinkingLevel, 'low');
    });

    test('adds items only before replies recorded by this model', () async {
      final unstamped = (await ask([user('one')], 'high', m: luna({}))).message;

      AssistantMessage copy({String? provider, String? model, String? level}) {
        return AssistantMessage.fromJson({
          ...unstamped.toJson(),
          'provider': ?provider,
          'model': ?model,
          'providerThinkingLevel': ?level,
        });
      }

      final payload = (await ask([
        user('one'),
        unstamped,
        user('two'),
        copy(provider: 'opencode', level: 'high'),
        user('three'),
        copy(model: 'gpt-6-sol', level: 'high'),
        user('four'),
        copy(level: 'high'),
        user('five'),
      ], 'low')).payload;
      expect(
        [for (final i in input(payload)) (i as Json)['type'] ?? i['role']],
        [
          'user',
          'message',
          'user',
          'message',
          'user',
          'message',
          'user',
          'configuration_update',
          'message',
          'user',
          'configuration_update',
        ],
      );
      expect(input(payload).where((i) => (i as Json)['type'] == 'configuration_update'), [
        update('high'),
        update('low'),
      ]);
    });

    for (final (name, m, level, effort) in [
      ('the compat flag turned off', luna({'supportsMidConvoEffort': false}), 'low', 'low'),
      ('thinking off', luna(), null, 'none'),
    ]) {
      test('sends the level as reasoning.effort with $name', () async {
        final first = await ask([user('one')], 'high', m: m);
        final (:payload, :message) = await ask([user('one'), first.message, user('two')], level, m: m);
        expect((payload['reasoning'] as Json)['effort'], effort);
        expect(input(payload).any((i) => (i as Json)['type'] == 'configuration_update'), isFalse);
        expect(message.providerThinkingLevel, isNull);
      });
    }
  });

  // pi's cases without PI_CACHE_RETENTION, which chorus does not read: long retention is asked for explicitly.
  group('cache retention', () {
    Future<Json> payload(Model m, CacheRetention retention, String session) async {
      return (await run(
        m,
        [user('hi')],
        options: (client) => OpenAIResponsesOptions(
          apiKey: 'sk-fake-key',
          httpClient: client,
          cacheRetention: retention,
          sessionId: session,
        ),
      )).payload;
    }

    for (final id in [
      'gpt-5.6-luna',
      'gpt-5.6-sol',
      'gpt-5.6-terra',
      'gpt-6-astra',
      'gpt-6-luna',
      'gpt-6-sol',
      'gpt-6.1-terra',
      'gpt-6.2-astra',
      'gpt-6.2-sol',
    ]) {
      test('does not enable cache warming from the documented TTL alone for $id', () {
        expect(catalog(id).promptCache, isNull);
      });
    }

    test('sets prompt_cache_retention for a proxy base URL', () async {
      final m = like(catalog('gpt-4o-mini'), baseUrl: 'https://my-proxy.example.com/v1');
      expect((await payload(m, CacheRetention.long, 's'))['prompt_cache_retention'], '24h');
    });

    test('omits prompt_cache_retention when supportsLongCacheRetention is false', () async {
      final m = like(catalog('gpt-4o-mini'), compat: {'supportsLongCacheRetention': false});
      expect(
        (await payload(m, CacheRetention.long, 'session-compat-false')).containsKey('prompt_cache_retention'),
        isFalse,
      );
    });

    test('omits prompt_cache_key and disables implicit writes when cacheRetention is none', () async {
      final p = await payload(catalog('gpt-5.6-sol'), CacheRetention.none, 'session-1');
      expect(p.containsKey('prompt_cache_key'), isFalse);
      expect(p.containsKey('prompt_cache_retention'), isFalse);
      expect(p['prompt_cache_options'], {'mode': 'explicit'});
    });

    test('omits prompt_cache_options for models that reject it', () async {
      final p = await payload(catalog('gpt-4o-mini'), CacheRetention.none, 'session-1');
      expect(p.containsKey('prompt_cache_key'), isFalse);
      expect(p.containsKey('prompt_cache_options'), isFalse);
    });

    for (final (id, retention, options) in [
      ('gpt-4o-mini', '24h', null),
      ('gpt-6-astra', null, {'ttl': '30m'}),
      ('gpt-6-sol', null, {'ttl': '30m'}),
      ('gpt-6-luna', null, {'ttl': '30m'}),
    ]) {
      test('uses the supported long cache field for $id', () async {
        final p = await payload(catalog(id), CacheRetention.long, 'session-2');
        expect(p['prompt_cache_key'], 'session-2');
        expect(p['prompt_cache_retention'], retention);
        expect(p['prompt_cache_options'], options);
      });
    }
  });

  group('error responses', () {
    Future<AssistantMessage> fail(http.Response response, [Model? m]) {
      return streamOpenAIResponses(
        m ?? catalog('gpt-5.4'),
        hi,
        OpenAIResponsesOptions(apiKey: 'test', httpClient: MockClient((_) async => response)),
      ).result;
    }

    // From pi's provider-error-body-regression test, through the real SDK error path.
    test('keeps the prefix and surfaces the body', () async {
      final out = await fail(http.Response(jsonEncode({'error': 'blocked by gateway WAF'}), 403));
      expect(out.stopReason, StopReason.error);
      expect(out.errorMessage, contains('OpenAI API error (403)'));
      expect(out.errorMessage, contains('blocked by gateway WAF'));
    });

    test("shows an error object's body, and the provider's name for other providers", () async {
      const body = {
        'error': {'message': 'Incorrect API key provided', 'type': 'invalid_request_error', 'code': 'invalid_api_key'},
      };
      expect(
        (await fail(http.Response(jsonEncode(body), 401))).errorMessage,
        'OpenAI API error (401): {"message":"Incorrect API key provided","type":"invalid_request_error","code":"invalid_api_key"}',
      );
      expect(
        (await fail(http.Response('', 500), like(catalog('gpt-5.4'), provider: 'proxy'))).errorMessage,
        'proxy API error (500): 500 status code (no body)',
      );
    });

    test('a connection failure keeps its cause', () async {
      final out = await streamOpenAIResponses(
        catalog('gpt-5.4'),
        hi,
        OpenAIResponsesOptions(
          apiKey: 'test',
          httpClient: MockClient((_) async => throw http.ClientException('connect ECONNREFUSED 127.0.0.1:9')),
        ),
      ).result;
      expect(out.errorMessage, 'Connection error.: connect ECONNREFUSED 127.0.0.1:9');
    });

    test('an error event in the stream ends the response with its message', () async {
      final out = await run(
        catalog('gpt-5.4'),
        hi,
        body: 'event: error\ndata: {"type":"error","code":"server_error","message":"The server had an error"}\n\n',
      );
      expect((out.message.stopReason, out.message.errorMessage), (StopReason.error, 'The server had an error'));
    });

    test('a missing key fails the request, and streamSimple at once', () async {
      final out = await streamOpenAIResponses(catalog('gpt-5.4'), hi).result;
      expect(out.errorMessage, 'No API key for provider: openai');
      expect(() => streamSimpleOpenAIResponses(catalog('gpt-5.4'), hi), throwsA(isA<ProviderException>()));
      final viaHeader = await run(
        catalog('gpt-5.4'),
        hi,
        options: (client) => OpenAIResponsesOptions(httpClient: client, headers: {'Authorization': 'Bearer gateway'}),
      );
      expect(viaHeader.headers['authorization'], 'Bearer gateway');
    });
  });
}
