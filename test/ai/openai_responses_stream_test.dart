// chorus's own tests of the OpenAI Responses stream and request path, for what pi's unit tests leave to its live
// tests: a full streamed reply and its replay, reasoning backfill, service tiers, events without SSE names, headers,
// hooks, retries, long error bodies and streamSimple.

import 'dart:convert';

import 'package:chorus/chorus.dart';
import 'package:chorus/src/ai/providers/openai_responses.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:test/test.dart';

const _cost = ModelCost(input: 1, output: 2, cacheRead: 0.5, cacheWrite: 0);

Model model({Json compat = const {}, Map<String, String>? headers, Map<String, String?>? levels, int window = 400000}) {
  return Model(
    id: 'gpt-test',
    name: 'GPT Test',
    api: 'openai-responses',
    provider: 'openai',
    baseUrl: 'https://api.openai.com/v1',
    reasoning: true,
    levels: levels,
    headers: headers,
    cost: _cost,
    contextWindow: window,
    maxTokens: 128000,
    compat: compat,
  );
}

UserMessage user(String text) => UserMessage([TextContent(text)], timestamp: 0);

String sse(List<Json> events) => events.map((e) => 'data: ${jsonEncode(e)}\n\n').join();

http.Response ok(String body) => http.Response(body, 200, headers: {'content-type': 'text/event-stream'});

Json completed([Json extra = const {}]) => {
  'type': 'response.completed',
  'response': {'id': 'resp_1', 'status': 'completed', ...extra},
};

/// The events and the result of [events] streamed through the provider.
Future<(List<AssistantMessageEvent>, AssistantMessage)> stream(
  List<Json> events, {
  Model? m,
  OpenAIResponsesOptions Function(http.Client client)? options,
}) async {
  final client = MockClient((_) async => ok(sse(events)));
  final s = streamOpenAIResponses(m ?? model(), [
    user('hi'),
  ], options?.call(client) ?? OpenAIResponsesOptions(apiKey: 'sk-test', httpClient: client));
  final seen = await s.toList();
  return (seen, await s.result);
}

void main() {
  group('a full streamed reply', () {
    const reasoning = {
      'type': 'reasoning',
      'id': 'rs_1',
      'summary': [
        {'type': 'summary_text', 'text': 'First thought.'},
        {'type': 'summary_text', 'text': 'Second thought.'},
      ],
      'encrypted_content': 'gAAAA-secret',
    };
    const message = {
      'type': 'message',
      'id': 'msg_1',
      'role': 'assistant',
      'status': 'completed',
      'phase': 'final_answer',
      'content': [
        {'type': 'output_text', 'text': 'Hello', 'annotations': <Object?>[]},
        {'type': 'refusal', 'refusal': ' but no.'},
      ],
    };
    const call = {
      'type': 'function_call',
      'id': 'fc_1',
      'call_id': 'call_1',
      'name': 'lookup',
      'arguments': '{"q":"x"}',
    };
    final events = <Json>[
      {
        'type': 'response.created',
        'response': {'id': 'resp_1'},
      },
      {
        'type': 'response.output_item.added',
        'output_index': 0,
        'item': {'type': 'reasoning', 'id': 'rs_1', 'summary': <Object?>[]},
      },
      {'type': 'response.reasoning_summary_text.delta', 'output_index': 0, 'delta': 'First thought.'},
      {'type': 'response.reasoning_summary_part.done', 'output_index': 0},
      {'type': 'response.reasoning_summary_text.delta', 'output_index': 0, 'delta': 'Second thought.'},
      {'type': 'response.output_item.done', 'output_index': 0, 'item': reasoning},
      {
        'type': 'response.output_item.added',
        'output_index': 1,
        'item': {'type': 'message', 'id': 'msg_1', 'role': 'assistant', 'content': <Object?>[], 'phase': 'commentary'},
      },
      {'type': 'response.output_text.delta', 'output_index': 1, 'delta': 'Hello'},
      {'type': 'response.refusal.delta', 'output_index': 1, 'delta': ' but no.'},
      {'type': 'response.output_item.done', 'output_index': 1, 'item': message},
      {
        'type': 'response.output_item.added',
        'output_index': 2,
        'item': {...call, 'arguments': ''},
      },
      {'type': 'response.function_call_arguments.delta', 'output_index': 2, 'delta': '{"q":'},
      {'type': 'response.function_call_arguments.done', 'output_index': 2, 'arguments': '{"q":"x"}'},
      {'type': 'response.output_item.done', 'output_index': 2, 'item': call},
      completed({
        'usage': {
          'input_tokens': 100,
          'output_tokens': 50,
          'total_tokens': 150,
          'input_tokens_details': {'cached_tokens': 40},
          'output_tokens_details': {'reasoning_tokens': 30},
        },
      }),
    ];

    test('streams every block with its deltas and ends as a tool use', () async {
      final (seen, out) = await stream(events);
      expect(seen.map((e) => e.runtimeType), [
        StartEvent,
        ThinkingStartEvent,
        ThinkingDeltaEvent,
        ThinkingDeltaEvent,
        ThinkingDeltaEvent,
        ThinkingEndEvent,
        TextStartEvent,
        TextDeltaEvent,
        TextDeltaEvent,
        TextEndEvent,
        ToolCallStartEvent,
        ToolCallDeltaEvent,
        ToolCallDeltaEvent,
        ToolCallEndEvent,
        DoneEvent,
      ]);
      expect(
        [for (final e in seen.whereType<ThinkingDeltaEvent>()) e.delta],
        ['First thought.', '\n\n', 'Second thought.'],
      );
      expect([for (final e in seen.whereType<ToolCallDeltaEvent>()) e.delta], ['{"q":', '"x"}']);
      expect((out.stopReason, out.responseId), (StopReason.toolUse, 'resp_1'));
      final [thinking as ThinkingContent, text as TextContent, tool as ToolCall] = out.content;
      expect(thinking.thinking, 'First thought.\n\nSecond thought.');
      expect(jsonDecode(thinking.signature!), reasoning);
      expect(text.text, 'Hello but no.');
      expect(jsonDecode(text.signature!), {'v': 1, 'id': 'msg_1', 'phase': 'final_answer'});
      expect((tool.id, tool.name), ('call_1|fc_1', 'lookup'));
      expect(tool.arguments, {'q': 'x'});
      final u = out.usage;
      expect((u.input, u.output, u.cacheRead, u.reasoning, u.totalTokens), (60, 50, 40, 30, 150));
      expect(u.cost.total, closeTo((60 * 1 + 50 * 2 + 40 * 0.5) / 1e6, 1e-15));
    });

    test('is replayed as the same items', () async {
      final (_, out) = await stream(events);
      // The call has no result yet, so a placeholder output follows; it is not part of the reply.
      expect(convertResponsesMessages(model(), [user('hi'), out]).skip(1).take(3), [
        reasoning,
        {
          'type': 'message',
          'role': 'assistant',
          'content': [
            {'type': 'output_text', 'text': 'Hello but no.', 'annotations': <Object?>[]},
          ],
          'status': 'completed',
          'id': 'msg_1',
          'phase': 'final_answer',
        },
        {'type': 'function_call', 'id': 'fc_1', 'call_id': 'call_1', 'name': 'lookup', 'arguments': '{"q":"x"}'},
      ]);
    });
  });

  test('reasoning text deltas stream, and content stands in for a missing summary', () async {
    final (seen, out) = await stream([
      {
        'type': 'response.output_item.added',
        'output_index': 0,
        'item': {'type': 'reasoning', 'id': 'rs_1'},
      },
      {'type': 'response.reasoning_text.delta', 'output_index': 0, 'delta': 'raw'},
      {
        'type': 'response.output_item.done',
        'output_index': 0,
        'item': {
          'type': 'reasoning',
          'id': 'rs_1',
          'summary': <Object?>[],
          'content': [
            {'type': 'reasoning_text', 'text': 'raw'},
            {'type': 'reasoning_text', 'text': 'more'},
          ],
        },
      },
      completed(),
    ]);
    expect(seen.whereType<ThinkingDeltaEvent>().single.delta, 'raw');
    expect((out.content.single as ThinkingContent).thinking, 'raw\n\nmore');
  });

  // Azure OpenAI can send the encrypted reasoning only with the terminal response.
  test('backfills encrypted reasoning from the terminal response', () async {
    const item = {'type': 'reasoning', 'id': 'rs_1', 'summary': <Object?>[]};
    final (_, out) = await stream([
      {'type': 'response.output_item.added', 'output_index': 0, 'item': item},
      {'type': 'response.output_item.done', 'output_index': 0, 'item': item},
      completed({
        'output': [
          {...item, 'encrypted_content': 'late-secret'},
        ],
      }),
    ]);
    expect(jsonDecode((out.content.single as ThinkingContent).signature!), {
      ...item,
      'encrypted_content': 'late-secret',
    });
  });

  test('a terminal reasoning item without encrypted content leaves the signature alone', () async {
    const item = {'type': 'reasoning', 'id': 'rs_1', 'summary': <Object?>[]};
    final (_, out) = await stream([
      {'type': 'response.output_item.added', 'output_index': 0, 'item': item},
      {'type': 'response.output_item.done', 'output_index': 0, 'item': item},
      completed({
        'output': [item],
      }),
    ]);
    expect(jsonDecode((out.content.single as ThinkingContent).signature!), item);
  });

  test('OpenAIResponsesOptions.from copies every common option', () {
    final signal = AbortSignal();
    final client = MockClient((_) async => ok(''));

    Json? payload(Json p, Model m) => p;

    void response(ProviderResponse r, Model m) {}

    void event(Json e, Model m) {}
    final o = OpenAIResponsesOptions.from(
      StreamOptions(
        signal: signal,
        apiKey: 'k',
        headers: const {'h': 'v'},
        httpClient: client,
        timeout: const Duration(seconds: 3),
        maxRetries: 2,
        maxRetryDelay: const Duration(seconds: 4),
        temperature: 0.5,
        maxTokens: 99,
        cacheRetention: CacheRetention.long,
        sessionId: 's',
        metadata: const {'m': 1},
        onPayload: payload,
        onResponse: response,
        onEvent: event,
        disableContinuation: true,
      ),
    );
    expect(
      [
        o.signal,
        o.apiKey,
        o.headers,
        o.httpClient,
        o.timeout,
        o.maxRetries,
        o.maxRetryDelay,
        o.temperature,
        o.maxTokens,
        o.cacheRetention,
        o.sessionId,
        o.metadata,
        o.onPayload,
        o.onResponse,
        o.onEvent,
        o.disableContinuation,
      ],
      [
        same(signal),
        'k',
        {'h': 'v'},
        same(client),
        const Duration(seconds: 3),
        2,
        const Duration(seconds: 4),
        0.5,
        99,
        CacheRetention.long,
        's',
        {'m': 1},
        payload,
        response,
        event,
        true,
      ],
    );
  });

  test("prices the tier the response reports over the requested one", () async {
    final (_, out) = await stream([
      completed({
        'service_tier': 'default',
        'usage': {'input_tokens': 1000000, 'output_tokens': 0, 'total_tokens': 1000000},
      }),
    ], options: (client) => OpenAIResponsesOptions(apiKey: 'sk-test', httpClient: client, serviceTier: 'flex'));
    expect(out.usage.cost.input, 1);
  });

  for (final status in ['failed', 'cancelled']) {
    test('a $status terminal status ends with an error', () async {
      final (_, out) = await stream([
        completed({'status': status}),
      ]);
      expect((out.stopReason, out.errorMessage), (StopReason.error, 'An unknown error occurred'));
    });
  }

  for (final (response, message) in [
    (<String, Object?>{'status': 'failed', 'error': <String, Object?>{}}, 'unknown: no message'),
    (
      {
        'status': 'failed',
        'incomplete_details': {'reason': 'max_time'},
      },
      'incomplete: max_time',
    ),
    (<String, Object?>{'status': 'failed'}, 'Unknown error (no error details in response)'),
  ]) {
    test('a failed response says why: $message', () async {
      final (_, out) = await stream([
        {'type': 'response.failed', 'response': response},
      ]);
      expect((out.stopReason, out.errorMessage, out.rawStopReason), (StopReason.error, message, 'failed'));
    });
  }

  test('an error event without an SSE name reports its code', () async {
    final (_, out) = await stream([
      {'type': 'error', 'code': 'rate_limit', 'message': 'Slow down'},
    ]);
    expect(out.errorMessage, 'Error Code rate_limit: Slow down');
  });

  group('requests', () {
    Future<http.Request> capture(Model m, [SimpleStreamOptions? options]) async {
      late http.Request sent;
      final client = MockClient((request) async {
        sent = request;
        return ok(sse([completed()]));
      });
      final base = options ?? const SimpleStreamOptions();
      await streamSimpleOpenAIResponses(
        m,
        [user('hi')],
        SimpleStreamOptions(
          apiKey: 'sk-test',
          httpClient: client,
          headers: base.headers,
          reasoning: base.reasoning,
          maxTokens: base.maxTokens,
          toolChoice: base.toolChoice,
        ),
      ).result;
      return sent;
    }

    test("sends the model's headers, lowercased, under the caller's", () async {
      final sent = await capture(
        model(headers: {'X-Team': 'a', 'X-Other': 'b'}),
        const SimpleStreamOptions(headers: {'x-team': 'c'}),
      );
      expect((sent.headers['x-team'], sent.headers['x-other']), ('c', 'b'));
    });

    test('onPayload can replace the body, and onResponse sees the response', () async {
      late http.Request sent;
      ProviderResponse? seen;
      final client = MockClient((request) async {
        sent = request;
        return http.Response(
          sse([completed()]),
          200,
          headers: {'content-type': 'text/event-stream', 'x-request-id': 'r1'},
        );
      });
      await streamOpenAIResponses(
        model(),
        [user('hi')],
        OpenAIResponsesOptions(
          apiKey: 'sk-test',
          httpClient: client,
          onPayload: (body, _) => {
            ...body,
            'metadata': {'trace': 't'},
          },
          onResponse: (response, _) => seen = response,
        ),
      ).result;
      expect((jsonDecode(sent.body) as Json)['metadata'], {'trace': 't'});
      expect((seen?.status, seen?.headers['x-request-id']), (200, 'r1'));
    });

    test('retries a failed request as often as asked', () async {
      var calls = 0;
      final client = MockClient((_) async {
        calls++;
        if (calls == 1) {
          return http.Response('{"error":{"message":"busy"}}', 503, headers: {'retry-after-ms': '1'});
        }

        return ok(sse([completed()]));
      });
      final out = await streamOpenAIResponses(model(), [
        user('hi'),
      ], OpenAIResponsesOptions(apiKey: 'sk-test', httpClient: client, maxRetries: 1)).result;
      expect((out.stopReason, calls), (StopReason.stop, 2));
    });

    test('cuts a long error body', () async {
      final long = 'x' * 5000;
      final out = await streamOpenAIResponses(
        model(),
        [user('hi')],
        OpenAIResponsesOptions(
          apiKey: 'sk-test',
          httpClient: MockClient(
            (_) async => http.Response(
              jsonEncode({
                'error': {'detail': long},
              }),
              400,
            ),
          ),
        ),
      ).result;
      final body = jsonEncode({'detail': long});
      expect(
        out.errorMessage,
        'OpenAI API error (400): ${body.substring(0, 4000)}... [truncated ${body.length - 4000} chars]',
      );
    });

    test('streamSimple clamps the level to the model and maps it', () async {
      final m = model(levels: {'off': 'none', 'low': 'low', 'medium': 'medium', 'high': 'deep', 'xhigh': null});
      final body =
          jsonDecode((await capture(m, const SimpleStreamOptions(reasoning: ThinkingLevel.xhigh))).body) as Json;
      expect(body['reasoning'], {'effort': 'deep', 'summary': 'auto'});
    });

    test('streamSimple fits max tokens into the context window and passes the tool choice', () async {
      final body =
          jsonDecode((await capture(model(window: 20000), const SimpleStreamOptions(toolChoice: 'none'))).body) as Json;
      expect(body['max_output_tokens'], lessThan(20000));
      expect(body['tool_choice'], 'none');
    });

    test('streamSimple sends no effort for thinking off', () async {
      final body = jsonDecode(
        (await capture(model(levels: {'off': 'none'}), const SimpleStreamOptions(reasoning: ThinkingLevel.off))).body,
      ) as Json;
      expect(body['reasoning'], {'effort': 'none'});
    });
  });
}
