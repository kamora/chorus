import 'dart:convert';
import 'dart:io';

import 'package:chorus/chorus.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:test/test.dart';

import 'anthropic_fake.dart';

// From pi's test/anthropic-auth-token.test.ts: header auth and the anthropic-beta header. Its environment, OAuth and
// User-Agent cases are left out with those features. The rest is chorus's own: requests built from options, through
// a mock HTTP client.

final _body = sse([
  {
    'type': 'message_start',
    'message': {
      'id': 'msg_test',
      'usage': {'input_tokens': 1, 'output_tokens': 0},
    },
  },
  {
    'type': 'message_delta',
    'delta': {'stop_reason': 'end_turn'},
    'usage': {'output_tokens': 1},
  },
  {'type': 'message_stop'},
]);

/// Runs one request through a mock HTTP client answering [replies] in turn; returns the requests and the result.
Future<(List<http.Request>, AssistantMessage)> send(
  Model model,
  AnthropicOptions Function(http.Client client) options, {
  List<(int, String, Map<String, String>)>? replies,
  List<Message>? messages,
}) async {
  final requests = <http.Request>[];
  return (
    requests,
    await streamAnthropic(
      model,
      messages ??
          [
            SystemMessage([TextContent('System prompt.')], timestamp: 0),
            user('Hello'),
          ],
      options(
        MockClient((request) async {
          requests.add(request);
          final (status, body, headers) = (replies ?? [(200, _body, <String, String>{})])[requests.length - 1];
          return http.Response(body, status, headers: headers);
        }),
      ),
    ).result,
  );
}

void main() {
  group('Anthropic auth', () {
    test('uses Authorization headers without an API key', () async {
      final (requests, result) = await send(
        claude(),
        (c) => AnthropicOptions(httpClient: c, headers: {'Authorization': 'Bearer gateway-token'}),
      );

      expect(result.stopReason, StopReason.stop);
      expect(requests.single.headers['authorization'], 'Bearer gateway-token');
      expect((jsonDecode(requests.single.body) as Json)['system'], [
        {
          'type': 'text',
          'text': 'System prompt.',
          'cache_control': {'type': 'ephemeral'},
        },
      ]);
    });

    test('sends an API key as a bearer token', () async {
      final (requests, _) = await send(claude(), (c) => AnthropicOptions(httpClient: c, apiKey: 'anthropic-key'));
      expect(requests.single.headers['authorization'], 'Bearer anthropic-key');
    });

    test('fails without credentials', () async {
      final (requests, result) = await send(claude(), (c) => AnthropicOptions(httpClient: c));

      expect(requests, isEmpty);
      expect(result.stopReason, StopReason.error);
      expect(result.errorMessage, 'No API key for provider: anthropic');
    });

    test('streamSimple throws at once without credentials', () {
      expect(() => streamSimpleAnthropic(claude(), [user('Hello')]), throwsA(isA<ProviderException>()));
    });
  });

  group('Anthropic beta header', () {
    test('preserves explicit Anthropic beta header replacement', () async {
      final (requests, _) = await send(
        claude(),
        (c) => AnthropicOptions(httpClient: c, apiKey: 'k', headers: {'anthropic-beta': 'custom-beta'}),
      );
      expect(requests.single.headers['anthropic-beta'], 'custom-beta');
      expect(jsonDecode(requests.single.body) as Json, isNot(contains('betas')));
    });

    test('preserves explicit Anthropic beta header suppression', () async {
      final (requests, _) = await send(
        claude(),
        (c) => AnthropicOptions(httpClient: c, apiKey: 'k', thinkingEnabled: true, headers: {'anthropic-beta': null}),
      );
      expect(requests.single.headers, isNot(contains('anthropic-beta')));
    });

    test('asks for interleaved thinking with budget-based thinking', () async {
      final (requests, _) = await send(
        claude(),
        (c) => AnthropicOptions(httpClient: c, apiKey: 'k', thinkingEnabled: true),
      );
      expect(requests.single.headers['anthropic-beta'], 'interleaved-thinking-2025-05-14');
    });

    test('leaves interleaved thinking to adaptive models', () async {
      final (requests, _) = await send(
        claude(compat: {'forceAdaptiveThinking': true}),
        (c) => AnthropicOptions(httpClient: c, apiKey: 'k', thinkingEnabled: true),
      );
      expect(requests.single.headers, isNot(contains('anthropic-beta')));
    });
  });

  group('Anthropic stream', () {
    test('keeps the input tokens of message_start when message_delta leaves them out', () async {
      final (_, result) = await send(claude(), (c) => AnthropicOptions(httpClient: c, apiKey: 'k'));
      expect(result.usage.input, 1);
      expect(result.usage.output, 1);
      expect(result.usage.totalTokens, 2);
    });

    test('fails a stream that ends before message_stop', () async {
      final (_, result) = await send(
        claude(),
        (c) => AnthropicOptions(httpClient: c, apiKey: 'k'),
        replies: [(200, sse(minimalEvents().take(5).toList()), {})],
      );
      expect(result.stopReason, StopReason.error);
      expect(result.errorMessage, 'Anthropic stream ended before message_stop');
    });
  });

  group('Anthropic requests', () {
    test('sends session affinity headers to OpenRouter', () async {
      final (requests, _) = await send(
        claude(provider: 'openrouter', baseUrl: 'https://openrouter.ai/api'),
        (c) => AnthropicOptions(httpClient: c, apiKey: 'k', sessionId: 'session-1'),
      );
      expect(requests.single.headers['x-session-id'], 'session-1');
      expect(requests.single.url.toString(), 'https://openrouter.ai/api/v1/messages');
    });

    test('asks for one-hour caching with long retention', () async {
      final (requests, _) = await send(
        claude(),
        (c) => AnthropicOptions(httpClient: c, apiKey: 'k', cacheRetention: CacheRetention.long),
      );
      expect(
        (((jsonDecode(requests.single.body) as Json)['messages'] as List).last['content'] as List)
            .last['cache_control'],
        {'type': 'ephemeral', 'ttl': '1h'},
      );
    });

    test('retries an overloaded server and reports the API error when retries run out', () async {
      const overloaded = '{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}';
      final (requests, result) = await send(
        claude(),
        (c) => AnthropicOptions(httpClient: c, apiKey: 'k', maxRetries: 1),
        replies: [
          (529, overloaded, {'retry-after-ms': '10'}),
          (529, overloaded, {'retry-after-ms': '10'}),
        ],
      );

      expect(requests, hasLength(2));
      expect(result.stopReason, StopReason.error);
      expect(result.errorMessage, contains('Overloaded'));
    });

    test('aborts a streaming response', () async {
      // A real server, since the mock client can't abort: it sends the start of a stream and never ends it.
      final server = await HttpServer.bind(InternetAddress.loopbackIPv4, 0);
      addTearDown(() => server.close(force: true));
      server.listen((request) async {
        request.response
          ..bufferOutput = false
          ..headers.contentType = ContentType('text', 'event-stream')
          ..write('${sse(minimalEvents().take(3).toList())}\n');
        await request.response.flush();
      });

      final signal = AbortSignal();
      final stream = streamAnthropic(claude(baseUrl: 'http://127.0.0.1:${server.port}'), [
        user('Hello'),
      ], AnthropicOptions(apiKey: 'k', signal: signal));
      final seen = <AssistantMessageEvent>[];
      await for (final event in stream) {
        seen.add(event);
        if (event is TextDeltaEvent) {
          signal.abort();
        }
      }

      final result = await stream.result;
      expect(seen.last, isA<ErrorEvent>());
      expect(result.stopReason, StopReason.aborted);
      expect(result.errorMessage, 'Request was aborted');
      expect((result.content.single as TextContent).text, 'Hello');
    });
  });
}
