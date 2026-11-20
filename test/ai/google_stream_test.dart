// chorus's own tests of the Google provider, for what pi's unit tests leave out: the streamed reply (blocks,
// signatures, tool call ids, usage), the REST body, the client setup, streamSimple, and the helpers' edge cases.

import 'dart:convert';

import 'package:chorus/chorus.dart';
import 'package:chorus/src/ai/providers/google.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:test/test.dart';

Model model(
  String id, {
  String baseUrl = 'https://example.com/v1beta',
  bool reasoning = true,
  List<String> input = const ['text'],
  Map<String, String?>? levels,
  Map<String, String>? headers,
  int window = 128000,
}) {
  return Model(
    id: id,
    name: id,
    api: 'google-generative-ai',
    provider: 'google',
    baseUrl: baseUrl,
    reasoning: reasoning,
    input: input,
    levels: levels,
    headers: headers,
    cost: const ModelCost(input: 1, output: 2, cacheRead: 0.5, cacheWrite: 0),
    contextWindow: window,
    maxTokens: 8192,
  );
}

UserMessage user(String text) => UserMessage([TextContent(text)], timestamp: 0);

String sse(List<Json> chunks) => chunks.map((c) => 'data: ${jsonEncode(c)}\n\n').join();

Json text(String text, {bool thought = false, String? signature, String? finish}) {
  return {
    'candidates': [
      {
        'content': {
          'parts': [
            {'text': text, if (thought) 'thought': true, 'thoughtSignature': ?signature},
          ],
        },
        'finishReason': ?finish,
      },
    ],
  };
}

Json call(String? id, {String name = 'bash', String? signature, String? finish}) {
  return {
    'candidates': [
      {
        'content': {
          'parts': [
            {
              'functionCall': {
                'id': ?id,
                'name': name,
                'args': {'n': 1},
              },
              'thoughtSignature': ?signature,
            },
          ],
        },
        'finishReason': ?finish,
      },
    ],
  };
}

const _stop = {
  'candidates': [
    {'finishReason': 'STOP'},
  ],
};

/// The events, the result and the request of [chunks] streamed through streamGoogle.
Future<(List<AssistantMessageEvent>, AssistantMessage, http.Request)> stream(
  List<Json> chunks, {
  Model? m,
  List<Message>? messages,
  GoogleOptions Function(http.Client client)? options,
}) async {
  late http.Request sent;
  final client = MockClient((request) async {
    sent = request;
    return http.Response(sse(chunks), 200, headers: {'content-type': 'text/event-stream'});
  });
  final s = streamGoogle(
    m ?? model('gemini-3-flash-preview'),
    messages ?? [user('hi')],
    options?.call(client) ?? GoogleOptions(apiKey: 'k', httpClient: client),
  );
  return (await s.toList(), await s.result, sent);
}

/// The REST body streamSimple builds.
Future<Json> simple(
  Model m,
  SimpleStreamOptions Function(http.Client client) options, [
  List<Message>? messages,
]) async {
  return jsonDecode((await streamSimpleRequest(m, options, messages)).body) as Json;
}

Future<http.Request> streamSimpleRequest(
  Model m,
  SimpleStreamOptions Function(http.Client client) options, [
  List<Message>? messages,
]) async {
  late http.Request sent;
  await streamSimpleGoogle(
    m,
    messages ?? [user('hi')],
    options(
      MockClient((request) async {
        sent = request;
        return http.Response(sse([_stop]), 200);
      }),
    ),
  ).result;
  return sent;
}

void main() {
  group('a streamed reply', () {
    test('switches between thinking and text blocks, keeping each block signature', () async {
      final (events, out, _) = await stream([
        text('Let me ', thought: true, signature: 'sigA'),
        text('think.', thought: true),
        text('Answer', signature: 'sigB'),
        text(' done', finish: 'STOP'),
      ]);
      expect(events.map((e) => e.runtimeType), [
        StartEvent,
        ThinkingStartEvent,
        ThinkingDeltaEvent,
        ThinkingDeltaEvent,
        ThinkingEndEvent,
        TextStartEvent,
        TextDeltaEvent,
        TextDeltaEvent,
        TextEndEvent,
        DoneEvent,
      ]);
      final [thinking as ThinkingContent, answer as TextContent] = out.content;
      expect((thinking.thinking, thinking.signature), ('Let me think.', 'sigA'));
      expect((answer.text, answer.signature), ('Answer done', 'sigB'));
    });

    test('closes the open block at the end of the stream', () async {
      final (events, _, _) = await stream([text('hi'), _stop]);
      expect(events.whereType<TextEndEvent>().single.content, 'hi');
    });

    test('a tool call closes the open block and carries its signature', () async {
      final (events, out, _) = await stream([text('Calling'), call('c1', signature: 'sigC', finish: 'STOP')]);
      expect(events.map((e) => e.runtimeType).skip(3).take(4), [
        TextEndEvent,
        ToolCallStartEvent,
        ToolCallDeltaEvent,
        ToolCallEndEvent,
      ]);
      expect(events.whereType<ToolCallDeltaEvent>().single.delta, '{"n":1}');
      final tool = out.content.last as ToolCall;
      expect((tool.id, tool.thoughtSignature, out.stopReason), ('c1', 'sigC', StopReason.toolUse));
      expect(tool.arguments, {'n': 1});
    });

    test('a missing or repeated tool call id gets a new one', () async {
      final (_, out, _) = await stream([call('c1'), call('c1'), call(null, name: 'grep', finish: 'STOP')]);
      final ids = [for (final c in out.content.whereType<ToolCall>()) c.id];
      expect(ids.first, 'c1');
      expect(ids[1], allOf(startsWith('bash_'), isNot('c1')));
      expect(ids[2], startsWith('grep_'));
    });

    test('keeps the first response id, and prices the usage', () async {
      final (_, out, _) = await stream([
        {...text('a'), 'responseId': 'r1'},
        {
          ...text('b', finish: 'STOP'),
          'responseId': 'r2',
          'usageMetadata': {
            'promptTokenCount': 100,
            'cachedContentTokenCount': 40,
            'candidatesTokenCount': 20,
            'thoughtsTokenCount': 5,
            'totalTokenCount': 125,
          },
        },
      ]);
      expect(out.responseId, 'r1');
      final u = out.usage;
      expect((u.input, u.output, u.cacheRead, u.reasoning, u.totalTokens), (60, 25, 40, 5, 125));
      expect(u.cost.total, closeTo((60 * 1 + 25 * 2 + 40 * 0.5) / 1e6, 1e-15));
    });

    test('a stream without a finish reason ends with an error', () async {
      final (_, out, _) = await stream([text('a')]);
      expect((out.stopReason, out.errorMessage), (StopReason.error, 'Google stream ended without a finish reason'));
    });

    test('an aborted request ends as aborted', () async {
      final signal = AbortSignal()..abort();
      final out = await streamGoogle(model('gemini-3-flash-preview'), [
        user('hi'),
      ], GoogleOptions(apiKey: 'k', signal: signal)).result;
      expect((out.stopReason, out.errorMessage), (StopReason.aborted, 'Request was aborted'));
    });

    test('an API error gives its message', () async {
      expect(
        (await streamGoogle(
          model('gemini-3-flash-preview'),
          [user('hi')],
          GoogleOptions(
            apiKey: 'k',
            httpClient: MockClient(
              (_) async => http.Response(
                '{"error":{"code":403,"message":"denied","status":"PERMISSION_DENIED"}}',
                403,
                headers: {'content-type': 'application/json'},
              ),
            ),
          ),
        ).result).errorMessage,
        '{"error":{"code":403,"message":"denied","status":"PERMISSION_DENIED"}}',
      );
    });
  });

  group('requests', () {
    test("go to the model's base URL without adding a version, or to the default one", () async {
      final (_, _, custom) = await stream([_stop], m: model('gemini-3-flash-preview'));
      expect(
        custom.url.toString(),
        'https://example.com/v1beta/models/gemini-3-flash-preview:streamGenerateContent?alt=sse',
      );
      final (_, _, fallback) = await stream([_stop], m: model('gemini-3-flash-preview', baseUrl: ''));
      expect(fallback.url.toString(), startsWith('https://generativelanguage.googleapis.com/v1beta/models/'));
    });

    test("send the model's headers, lowercased, under the caller's, and the key", () async {
      final (_, _, sent) = await stream(
        [_stop],
        m: model('gemini-3-flash-preview', headers: {'X-Team': 'a', 'X-Other': 'b'}),
        options: (client) => GoogleOptions(apiKey: 'key', httpClient: client, headers: {'x-team': 'c'}),
      );
      expect((sent.headers['x-team'], sent.headers['x-other'], sent.headers['x-goog-api-key']), ('c', 'b', 'key'));
    });

    test("a caller's null removes a model header, whatever its case", () async {
      final (_, _, sent) = await stream(
        [_stop],
        m: model('gemini-3-flash-preview', headers: {'X-Team': 'a'}),
        options: (client) => GoogleOptions(apiKey: 'key', httpClient: client, headers: {'x-team': null}),
      );
      expect(sent.headers.containsKey('x-team'), isFalse);
    });

    test('carry the system prompt, tools, tool config and generation config', () async {
      final (_, _, sent) = await stream(
        [_stop],
        messages: [
          SystemMessage(
            [TextContent('Be brief.')],
            added: const [
              Tool('bash', 'Runs a command.', {'type': 'object', 'properties': <String, Object?>{}}),
            ],
            timestamp: 0,
          ),
          user('hi'),
        ],
        options: (client) =>
            GoogleOptions(apiKey: 'k', httpClient: client, temperature: 0.2, maxTokens: 300, toolChoice: 'auto'),
      );
      final body = jsonDecode(sent.body) as Json;
      expect(body['systemInstruction'], {
        'role': 'user',
        'parts': [
          {'text': 'Be brief.'},
        ],
      });
      expect(((body['tools'] as List<Object?>).single as Json)['functionDeclarations'], [
        {
          'name': 'bash',
          'description': 'Runs a command.',
          'parametersJsonSchema': {'type': 'object', 'properties': <String, Object?>{}},
        },
      ]);
      expect(body['toolConfig'], {
        'functionCallingConfig': {'mode': 'AUTO'},
      });
      expect(body['generationConfig'], {'temperature': 0.2, 'maxOutputTokens': 300});
      expect(body.keys, ['contents', 'systemInstruction', 'tools', 'toolConfig', 'generationConfig']);
    });

    test('send no system instruction or tool config without them', () async {
      final (_, _, sent) = await stream([_stop]);
      expect((jsonDecode(sent.body) as Json).keys, ['contents', 'generationConfig']);
    });

    test('onPayload can replace the body', () async {
      final (_, _, sent) = await stream(
        [_stop],
        options: (client) => GoogleOptions(
          apiKey: 'k',
          httpClient: client,
          onPayload: (body, _) => {...body, 'cachedContent': 'cachedContents/1'},
        ),
      );
      expect((jsonDecode(sent.body) as Json)['cachedContent'], 'cachedContents/1');
    });

    test('retries as often as asked', () async {
      var calls = 0;
      expect(
        (
          (await streamGoogle(
            model('gemini-3-flash-preview'),
            [user('hi')],
            GoogleOptions(
              apiKey: 'k',
              maxRetries: 1,
              httpClient: MockClient((_) async {
                calls++;
                if (calls == 1) {
                  return http.Response('busy', 503);
                }

                return http.Response(sse([_stop]), 200);
              }),
            ),
          ).result).stopReason,
          calls,
        ),
        (StopReason.stop, 2),
      );
    });
  });

  group('streamSimple', () {
    test('clamps the level to the model before mapping it', () async {
      expect(
        ((await simple(
              model('gemini-3.7-flash', levels: {'high': 'high', 'xhigh': null, 'max': null}),
              (c) => SimpleStreamOptions(apiKey: 'k', httpClient: c, reasoning: ThinkingLevel.xhigh),
            ))['generationConfig']
            as Json)['thinkingConfig'],
        {'includeThoughts': true, 'thinkingLevel': 'HIGH'},
      );
    });

    test('fits max tokens into the context window and passes the tool choice', () async {
      final body = await simple(
        model('gemini-3-flash-preview', window: 9000),
        (c) => SimpleStreamOptions(apiKey: 'k', httpClient: c, toolChoice: 'none'),
        [
          SystemMessage(
            const [],
            added: const [
              Tool('bash', 'b', {'type': 'object'}),
            ],
            timestamp: 0,
          ),
          user('hi'),
        ],
      );
      expect((body['generationConfig'] as Json)['maxOutputTokens'], lessThan(8192));
      expect(body['toolConfig'], {
        'functionCallingConfig': {'mode': 'NONE'},
      });
    });

    for (final (id, level, budget) in [
      ('gemini-2.5-pro', ThinkingLevel.high, 32768),
      ('gemini-2.5-flash-lite', ThinkingLevel.minimal, 512),
      ('gemini-2.5-flash', ThinkingLevel.minimal, 128),
      ('gemini-2.0-flash', ThinkingLevel.medium, -1),
    ]) {
      test('gives $id a ${level.name} budget of $budget', () async {
        expect(
          (((await simple(
                    model(id),
                    (c) => SimpleStreamOptions(apiKey: 'k', httpClient: c, reasoning: level),
                  ))['generationConfig']
                  as Json)['thinkingConfig']
              as Json)['thinkingBudget'],
          budget,
        );
      });
    }
  });

  group('helpers', () {
    test('Gemini thinking levels cover the Flash aliases and Gemma 4', () {
      for (final id in [
        'gemini-flash-latest',
        'gemini-flash-lite-latest',
        'gemma-4-27b',
        'gemma4-e4b',
        'gemini-3.1-pro',
      ]) {
        expect(usesGoogleThinkingLevel(model(id)), isTrue, reason: id);
      }

      expect(usesGoogleThinkingLevel(model('gemini-2.5-flash')), isFalse);
    });

    test('thinking turns off with a zero budget where off is a level', () {
      expect(disabledGoogleThinkingConfig(model('gemini-3-flash-preview', levels: {})), {'thinkingBudget': 0});
      expect(disabledGoogleThinkingConfig(model('gemini-3-flash-preview', levels: {'off': null, 'minimal': 'low'})), {
        'thinkingLevel': 'LOW',
      });
    });

    test('a const next to an enum is left alone', () {
      expect(
        ((((convertGoogleTools(const [
                          Tool('t', 'd', {
                            'type': 'object',
                            'properties': {
                              'x': {
                                'const': 'a',
                                'enum': ['a', 'b'],
                              },
                            },
                          }),
                        ])!.single['functionDeclarations']
                        as List<Object?>)
                    .single
                as Json)['parametersJsonSchema']
            as Json)['properties']),
        {
          'x': {
            'const': 'a',
            'enum': ['a', 'b'],
          },
        },
      );
    });

    test('tool choice maps to a calling mode', () {
      const tools = [
        Tool('t', 'd', {'type': 'object'}),
      ];
      expect(
        [
          for (final choice in [null, 'auto', 'none', 'any', 'other']) googleFunctionCallingMode(tools, choice, false),
        ],
        [null, 'AUTO', 'NONE', 'ANY', 'AUTO'],
      );
    });
  });

  group('conversion', () {
    final gemini3 = model('gemini-3-flash-preview', input: ['text', 'image']);

    AssistantMessage reply(List<Content> content, {String provider = 'google', String? id}) {
      return AssistantMessage(
        content: content,
        api: provider == 'google' ? 'google-generative-ai' : 'anthropic-messages',
        provider: provider,
        model: id ?? gemini3.id,
        stopReason: StopReason.toolUse,
        timestamp: 0,
      );
    }

    test('user images go as inline data, and an empty user message is left out', () {
      expect(
        convertGoogleMessages(gemini3, [
          UserMessage(const [], timestamp: 0),
          UserMessage([TextContent('see'), ImageContent('abc', 'image/png')], timestamp: 0),
        ]),
        [
          {
            'role': 'user',
            'parts': [
              {'text': 'see'},
              {
                'inlineData': {'mimeType': 'image/png', 'data': 'abc'},
              },
            ],
          },
        ],
      );
    });

    test('only base64 signatures of whole groups are kept', () {
      for (final (signature, kept) in [('AAAA', true), ('AAA', false), ('AAA*', false)]) {
        final parts =
            convertGoogleMessages(gemini3, [
                  user('hi'),
                  reply([TextContent('t', signature: signature)]),
                ]).last['parts']
                as List<Object?>;
        expect((parts.single as Json).containsKey('thoughtSignature'), kept, reason: signature);
      }
    });

    test("another provider's tool call ids are cleaned for models that need ids", () {
      final contents = convertGoogleMessages(gemini3, [
        user('hi'),
        reply([ToolCall('toolu.1|x${'y' * 70}', 'bash', const {})], provider: 'anthropic'),
        ToolResultMessage(
          id: 'toolu.1|x${'y' * 70}',
          name: 'bash',
          content: [TextContent('r')],
          isError: true,
          timestamp: 0,
        ),
      ]);
      final call = ((contents[1]['parts'] as List<Object?>).single as Json)['functionCall'] as Json;
      final response = ((contents[2]['parts'] as List<Object?>).single as Json)['functionResponse'] as Json;
      expect(call['id'], allOf(startsWith('toolu_1_x'), hasLength(64)));
      expect(response['id'], call['id']);
      expect(response['response'], {'error': 'r'});
    });

    test("images go inside the function response for models that are not Gemini", () {
      final other = model('custom-model', input: ['text', 'image']);
      final contents = convertGoogleMessages(other, [
        user('hi'),
        reply([ToolCall('c1', 'read', const {})], id: other.id),
        ToolResultMessage(
          id: 'c1',
          name: 'read',
          content: [ImageContent('abc', 'image/png')],
          isError: false,
          timestamp: 0,
        ),
      ]);
      expect(contents, hasLength(3));
      expect(
        (((contents[2]['parts'] as List<Object?>).single as Json)['functionResponse'] as Json)['parts'],
        hasLength(1),
      );
    });
  });
}
