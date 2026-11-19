import 'dart:convert';
import 'dart:math';

import 'package:anthropic_sdk_lite_dart/anthropic_sdk_lite_dart.dart';
import 'package:chorus/chorus.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';

/// A server-sent events body, as pi's tests build it: `event:` and `data:` lines, events separated by a blank line.
String sse(List<Json> events) {
  return events.map((e) => 'event: ${e['type']}\ndata: ${jsonEncode(e)}\n').join('\n');
}

/// A lite client whose requests are recorded in [requests] and answered with [body].
class FakeServer {
  FakeServer(String body, {int status = 200}) : this.sequence([(status, body)]);

  /// Answers each request with the next reply; the last one repeats.
  FakeServer.sequence(this._replies);

  final List<(int, String)> _replies;
  final requests = <http.Request>[];

  /// The JSON body of the last request.
  Json get body => jsonDecode(requests.last.body) as Json;

  Anthropic client() {
    return Anthropic(
      apiKey: 'test-key',
      baseURL: 'https://api.anthropic.com',
      httpClient: MockClient((request) async {
        requests.add(request);
        final (status, body) = _replies[min(requests.length, _replies.length) - 1];
        return http.Response(body, status, headers: {'content-type': 'text/event-stream'});
      }),
    );
  }
}

/// pi's minimal Anthropic stream: one text block "Hello", stop reason `end_turn`.
List<Json> minimalEvents() {
  return [
    {
      'type': 'message_start',
      'message': {
        'id': 'msg_test',
        'usage': {
          'input_tokens': 12,
          'output_tokens': 0,
          'cache_read_input_tokens': 0,
          'cache_creation_input_tokens': 0,
        },
      },
    },
    {
      'type': 'content_block_start',
      'index': 0,
      'content_block': {'type': 'text', 'text': ''},
    },
    {
      'type': 'content_block_delta',
      'index': 0,
      'delta': {'type': 'text_delta', 'text': 'Hello'},
    },
    {'type': 'content_block_stop', 'index': 0},
    {
      'type': 'message_delta',
      'delta': {'stop_reason': 'end_turn'},
      'usage': {'input_tokens': 12, 'output_tokens': 5, 'cache_read_input_tokens': 0, 'cache_creation_input_tokens': 0},
    },
    {'type': 'message_stop'},
  ];
}

/// A Claude model for tests; prices are Claude Opus 4.8's ($5 input, $6.25 5m cache write per million tokens).
Model claude({
  String id = 'claude-test',
  bool reasoning = true,
  Json compat = const {},
  List<String> input = const ['text', 'image'],
  Map<String, String?>? levels,
  Map<String, String>? headers,
  String provider = 'anthropic',
  String baseUrl = 'https://api.anthropic.com',
}) {
  return Model(
    id: id,
    name: id,
    api: 'anthropic-messages',
    provider: provider,
    baseUrl: baseUrl,
    input: input,
    cost: const ModelCost(input: 5, output: 25, cacheRead: 0.5, cacheWrite: 6.25),
    headers: headers,
    reasoning: reasoning,
    levels: levels,
    contextWindow: 200000,
    maxTokens: 64000,
    compat: compat,
  );
}

UserMessage user(String text, {int timestamp = 1}) => UserMessage([TextContent(text)], timestamp: timestamp);
