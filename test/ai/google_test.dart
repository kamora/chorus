// Ported from pi's packages/ai/test: google-thinking-signature, google-shared-retry, google-stop-reason-continuation,
// google-shared-signed-empty-blocks, google-shared-image-tool-result-routing, google-shared-gemini3-unsigned-tool-call,
// google-thinking-level-map, google-raw-stop-reason, google-shared-convert-tools and pre-generation-error. The SDK
// mock becomes http's MockClient answering with server-sent chunks, and payloads are the REST body, so pi's
// `config.thinkingConfig` is `generationConfig.thinkingConfig` here. Not ported: the Vertex and Cloud Code Assist cases
// (with the OpenAPI `parameters` form of tool schemas) and the identity User-Agent.

import 'dart:convert';

import 'package:chorus/chorus.dart';
import 'package:chorus/src/ai/providers/google.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:test/test.dart';

const _free = ModelCost(input: 0, output: 0, cacheRead: 0, cacheWrite: 0);
const _signature = 'AAAAAAAAAAAAAAAAAAAAAA==';

/// A level map with only low, medium and high.
const _lowToHigh = {
  'off': null,
  'minimal': null,
  'low': 'low',
  'medium': 'medium',
  'high': 'high',
  'xhigh': null,
  'max': null,
};

Model model(
  String id, {
  String provider = 'google',
  bool reasoning = true,
  List<String> input = const ['text'],
  Map<String, String?>? levels,
}) {
  return Model(
    id: id,
    name: id,
    api: 'google-generative-ai',
    provider: provider,
    baseUrl: 'https://example.com',
    reasoning: reasoning,
    input: input,
    levels: levels,
    cost: _free,
    contextWindow: 128000,
    maxTokens: 8192,
  );
}

UserMessage user(String text) => UserMessage([TextContent(text)], timestamp: 0);

AssistantMessage reply(Model m, List<Content> content, {String? id}) {
  return AssistantMessage(
    content: content,
    api: m.api,
    provider: m.provider,
    model: id ?? m.id,
    stopReason: StopReason.toolUse,
    timestamp: 0,
  );
}

ToolResultMessage result(String id, List<Content> content, {String name = 'bash'}) {
  return ToolResultMessage(id: id, name: name, content: content, isError: false, timestamp: 0);
}

/// The parts of the first turn with [role].
List<Json> parts(List<Json> contents, String role) {
  return (contents.firstWhere((c) => c['role'] == role)['parts'] as List<Object?>).cast<Json>();
}

String sse(List<Json> chunks) => chunks.map((c) => 'data: ${jsonEncode(c)}\n\n').join();

/// A request through streamGoogle answered with [chunks].
Future<AssistantMessage> run(Model m, List<Json> chunks, [GoogleOptions Function(http.Client client)? options]) {
  final client = MockClient(
    (_) async => http.Response(sse(chunks), 200, headers: {'content-type': 'text/event-stream'}),
  );
  return streamGoogle(m, [
    user('hello'),
  ], options?.call(client) ?? GoogleOptions(apiKey: 'test-api-key', httpClient: client)).result;
}

class Captured implements Exception {
  @override
  String toString() => 'payload captured';
}

/// The REST body streamSimple builds for [m] at [reasoning].
Future<Json> payload(Model m, [ThinkingLevel? reasoning, ThinkingBudgets? budgets]) async {
  Json? captured;
  expect(
    (await streamSimpleGoogle(
      m,
      [user('Hello')],
      SimpleStreamOptions(
        apiKey: 'test',
        reasoning: reasoning,
        budgets: budgets,
        onPayload: (body, _) {
          captured = body;
          throw Captured();
        },
      ),
    ).result).errorMessage,
    contains('payload captured'),
  );
  return captured!;
}

Object? thinkingConfig(Json payload) => (payload['generationConfig'] as Json)['thinkingConfig'];

void main() {
  group('thinking detection', () {
    test('treats thought: true as thinking', () {
      expect(isThinkingPart({'thought': true}), isTrue);
      expect(isThinkingPart({'thought': true, 'thoughtSignature': 'opaque-signature'}), isTrue);
    });

    test('does not treat a thoughtSignature alone as thinking', () {
      expect(isThinkingPart({'thoughtSignature': 'opaque-signature'}), isFalse);
      expect(isThinkingPart({'thought': false, 'thoughtSignature': 'opaque-signature'}), isFalse);
      expect(isThinkingPart({}), isFalse);
      expect(isThinkingPart({'thought': false, 'thoughtSignature': ''}), isFalse);
    });

    test('keeps the signature when later deltas leave it out, and takes a new one', () {
      expect(retainThoughtSignature(null, 'sig-1'), 'sig-1');
      expect(retainThoughtSignature('sig-1', null), 'sig-1');
      expect(retainThoughtSignature('sig-1', ''), 'sig-1');
      expect(retainThoughtSignature('sig-1', 'sig-2'), 'sig-2');
    });
  });

  group('retries', () {
    Future<(AssistantMessage, int)> attempt(int first, int? maxRetries) async {
      var calls = 0;
      return (
        await streamGoogle(
          model('gemini-2.5-flash'),
          [user('hi')],
          GoogleOptions(
            apiKey: 'k',
            maxRetries: maxRetries,
            httpClient: MockClient((_) async {
              calls++;
              if (calls == 1) {
                return http.Response('{"error":{"code":$first}}', first, headers: {'content-type': 'application/json'});
              }

              return http.Response(
                sse([
                  {
                    'candidates': [
                      {
                        'content': {
                          'parts': [
                            {'text': 'ok'},
                          ],
                        },
                        'finishReason': 'STOP',
                      },
                    ],
                  },
                ]),
                200,
              );
            }),
          ),
        ).result,
        calls,
      );
    }

    test("retries the SDK's headerless error with a retryable status", () async {
      final (out, calls) = await attempt(429, 1);
      expect((out.stopReason, calls), (StopReason.stop, 2));
    });

    test('does not retry without maxRetries, nor a status that is not retryable', () async {
      final (once, calls) = await attempt(429, null);
      expect((once.stopReason, calls), (StopReason.error, 1));
      final (bad, badCalls) = await attempt(400, 2);
      expect((bad.stopReason, badCalls, bad.errorMessage), (StopReason.error, 1, '{"error":{"code":400}}'));
    });
  });

  group('CONTINUATION finish reason', () {
    test('maps to length, like MAX_TOKENS', () {
      expect(mapGoogleStopReason('CONTINUATION'), StopReason.length);
    });

    test('ends the turn as length without a continuation request', () async {
      var calls = 0;
      expect(
        (
          (await streamGoogle(
            model('test-model', provider: 'test-provider', reasoning: false),
            [user('hello')],
            GoogleOptions(
              apiKey: 'test-key',
              httpClient: MockClient((_) async {
                calls++;
                return http.Response(
                  sse([
                    {
                      'candidates': [
                        {
                          'content': {
                            'role': 'model',
                            'parts': [
                              {'text': 'partial'},
                            ],
                          },
                          'finishReason': 'CONTINUATION',
                          'continuationToken': 'token-1',
                        },
                      ],
                    },
                  ]),
                  200,
                );
              }),
            ),
          ).result).stopReason,
          calls,
        ),
        (StopReason.length, 1),
      );
    });
  });

  group('signed empty blocks', () {
    final m = model('gemini-3-pro-preview');
    final call = ToolCall('call_1', 'bash', {'command': 'ls'});

    test('keeps a signed empty thinking block so its signature is sent back', () {
      final signed = parts(
        convertGoogleMessages(m, [
          user('Hi'),
          reply(m, [ThinkingContent('', signature: _signature), call]),
        ]),
        'model',
      ).where((x) => x['thoughtSignature'] == _signature).toList();
      expect(signed, hasLength(1));
      expect(signed.single['thought'], isTrue);
    });

    test('keeps a signed empty text block the same way', () {
      expect(
        parts(
          convertGoogleMessages(m, [
            user('Hi'),
            reply(m, [TextContent('', signature: _signature), call]),
          ]),
          'model',
        ).where((x) => x['thoughtSignature'] == _signature),
        hasLength(1),
      );
    });

    test('still drops unsigned empty blocks', () {
      expect(
        parts(
          convertGoogleMessages(m, [
            user('Hi'),
            reply(m, [ThinkingContent(''), TextContent('   '), call]),
          ]),
          'model',
        ).single.containsKey('functionCall'),
        isTrue,
      );
    });

    test("still drops signed empty blocks of another model", () {
      final contents = convertGoogleMessages(m, [
        user('Hi'),
        reply(m, [
          ThinkingContent('', signature: _signature),
          TextContent('', signature: _signature),
          call,
        ], id: 'other-model'),
      ]);
      expect(parts(contents, 'model').single.containsKey('functionCall'), isTrue);
      expect(jsonEncode(contents), isNot(contains(_signature)));
    });
  });

  group('image tool results', () {
    List<Message> history(Model m, {bool blank = false}) {
      return [
        user('read the files'),
        reply(m, [
          ToolCall('call_a', 'read', {'path': 'a.txt'}),
          ToolCall('call_img', 'read', {'path': 'image.png'}),
          ToolCall('call_b', 'read', {'path': 'b.txt'}),
        ]),
        result('call_a', [TextContent(blank ? '\r\n' : 'alpha text')], name: 'read'),
        result('call_img', [if (blank) TextContent(' \n'), ImageContent('abc', 'image/png')], name: 'read'),
        result('call_b', [TextContent('beta text')], name: 'read'),
      ];
    }

    test('keep a separate image turn for Gemini 2.x', () {
      final m = model('gemini-2.5-flash', input: ['text', 'image']);
      final contents = convertGoogleMessages(m, history(m));
      expect(contents, hasLength(5));
      expect((contents[2]['parts'] as List<Object?>).every((p) => (p as Json).containsKey('functionResponse')), isTrue);
      final image = (contents[3]['parts'] as List<Object?>).cast<Json>();
      expect(image[0]['text'], 'Tool result image:');
      expect(image[1]['inlineData'], isNotNull);
      expect(((contents[4]['parts'] as List<Object?>).first as Json)['functionResponse'], isNotNull);
    });

    test('nest the image for Gemini 3', () {
      final m = model('gemini-3-pro-preview', input: ['text', 'image']);
      final contents = convertGoogleMessages(m, history(m));
      expect(contents, hasLength(3));
      final turn = (contents[2]['parts'] as List<Object?>).cast<Json>();
      expect(turn, hasLength(3));
      expect(
        ((((turn[1]['functionResponse'] as Json)['parts'] as List<Object?>).single) as Json)['inlineData'],
        isNotNull,
      );
    });

    test('treat whitespace-only text as no text', () {
      final m = model('gemini-3-pro-preview', input: ['text', 'image']);
      expect(
        [
          for (final p in (convertGoogleMessages(m, history(m, blank: true))[2]['parts'] as List<Object?>).cast<Json>())
            (p['functionResponse'] as Json)['response'],
        ],
        [
          {'output': ''},
          {'output': '(see attached image)'},
          {'output': 'beta text'},
        ],
      );
    });
  });

  group('Gemini 3 unsigned tool calls', () {
    List<Message> history(Model m, {String? id, String? signature}) {
      return [
        user('Hi'),
        reply(m, [
          ToolCall('call_1', 'bash', {'command': 'echo hi'}, thoughtSignature: signature),
          ToolCall('call_2', 'bash', {'command': 'ls -la'}),
        ], id: id),
        result('call_1', [TextContent('hi')]),
        result('call_2', [TextContent('files')]),
      ];
    }

    List<Json> all(List<Json> contents) => [for (final c in contents) ...(c['parts'] as List<Object?>).cast<Json>()];

    for (final id in ['gemini-3-pro-preview', 'gemini-3.6-flash']) {
      test('keeps tool call ids for $id', () {
        final m = model(id);
        final every = all(convertGoogleMessages(m, history(m)));
        expect([for (final p in every) ?(p['functionCall'] as Json?)?['id']], ['call_1', 'call_2']);
        expect([for (final p in every) ?(p['functionResponse'] as Json?)?['id']], ['call_1', 'call_2']);
      });
    }

    test('adds no signature or skip marker to unsigned calls of another model', () {
      final m = model('gemini-3-pro-preview');
      final contents = convertGoogleMessages(m, history(m, id: 'other-model'));
      final calls = parts(contents, 'model').where((p) => p.containsKey('functionCall')).toList();
      expect(calls, hasLength(2));
      expect(calls.every((p) => !p.containsKey('thoughtSignature')), isTrue);
      expect(jsonEncode(contents), isNot(contains('skip_thought_signature_validator')));
      expect(parts(contents, 'model').where((p) => '${p['text']}'.contains('Historical context')), isEmpty);
    });

    test('keeps a valid signature of the same provider and model', () {
      final m = model('gemini-3-pro-preview');
      expect(
        [
          for (final p in parts(convertGoogleMessages(m, history(m, signature: _signature)), 'model'))
            p['thoughtSignature'],
        ],
        [_signature, null],
      );
    });

    test('adds no ids before Gemini 3', () {
      final m = model('gemini-2.5-flash');
      final every = all(convertGoogleMessages(m, history(m, id: 'other-model')));
      final calls = every.where((p) => p.containsKey('functionCall')).toList();
      final responses = every.where((p) => p.containsKey('functionResponse')).toList();
      expect((calls.length, responses.length), (2, 2));
      expect(
        calls.every((p) => !(p['functionCall'] as Json).containsKey('id') && !p.containsKey('thoughtSignature')),
        isTrue,
      );
      expect(responses.every((p) => !(p['functionResponse'] as Json).containsKey('id')), isTrue);
    });

    for (final (expected, id) in [
      (false, 'gemini-2.5-flash'),
      (true, 'gemini-3.6-flash'),
      (true, 'claude-sonnet-4-5'),
      (true, 'gpt-oss-120b'),
    ]) {
      test('requiresToolCallId is $expected for $id', () {
        expect(requiresToolCallId(id), expected);
      });
    }
  });

  group('thinking level maps', () {
    Model flash(Map<String, String?> levels, [String id = 'gemini-3.7-flash']) {
      return model(id, provider: 'test-google', levels: levels);
    }

    test('resolves every supported level and mapping value', () {
      for (final level in ['minimal', 'low', 'medium', 'high']) {
        expect(resolveGoogleThinkingLevel(flash({}), ThinkingLevel.values.byName(level)), level);
      }

      for (final (mapped, expected) in [
        ('minimal', 'minimal'),
        ('low', 'low'),
        ('medium', 'medium'),
        ('high', 'high'),
        ('MINIMAL', 'minimal'),
        ('LOW', 'low'),
        ('MEDIUM', 'medium'),
        ('HIGH', 'high'),
      ]) {
        for (final level in [ThinkingLevel.high, ThinkingLevel.xhigh, ThinkingLevel.max]) {
          expect(resolveGoogleThinkingLevel(flash({'high': mapped, 'xhigh': mapped, 'max': mapped}), level), expected);
        }
      }

      expect(
        () => resolveGoogleThinkingLevel(flash({'xhigh': 'extreme'}), ThinkingLevel.xhigh),
        throwsA(
          isA<ProviderException>().having(
            (e) => e.message,
            'message',
            'Unsupported Google thinking level mapping for test-google/gemini-3.7-flash: xhigh -> extreme',
          ),
        ),
      );
      expect(
        () => resolveGoogleThinkingLevel(flash({}), ThinkingLevel.max),
        throwsA(isA<ProviderException>().having((e) => e.message, 'message', endsWith('max -> undefined'))),
      );
      expect(
        () => resolveGoogleThinkingLevel(flash({'max': null}), ThinkingLevel.max),
        throwsA(isA<ProviderException>().having((e) => e.message, 'message', endsWith('max -> null'))),
      );
    });

    test('uses the lowest supported level when reasoning is omitted', () async {
      expect(thinkingConfig(await payload(flash(_lowToHigh, 'gemini-3.8-flash'))), {'thinkingLevel': 'LOW'});
    });

    test('keeps native medium effort for Gemini 3.1 Pro', () async {
      expect(thinkingConfig(await payload(flash(_lowToHigh, 'gemini-3.1-pro-preview'), ThinkingLevel.medium)), {
        'includeThoughts': true,
        'thinkingLevel': 'MEDIUM',
      });
    });

    test('turns Gemini 2.5 thinking off when reasoning is omitted', () async {
      expect(thinkingConfig(await payload(flash({}, 'gemini-2.5-flash'))), {'thinkingBudget': 0});
    });

    test('uses thinking levels for Gemini 4', () async {
      expect(thinkingConfig(await payload(flash({}, 'gemini-4-argon'), ThinkingLevel.medium)), {
        'includeThoughts': true,
        'thinkingLevel': 'MEDIUM',
      });
    });

    for (final level in [ThinkingLevel.xhigh, ThinkingLevel.max]) {
      test('maps ${level.name} to a supported level', () async {
        expect(thinkingConfig(await payload(flash({'xhigh': 'high', 'max': 'high'}), level)), {
          'includeThoughts': true,
          'thinkingLevel': 'HIGH',
        });
      });
    }

    test('honors uppercase provider values', () async {
      expect(
        (thinkingConfig(await payload(flash({'high': 'LOW'}), ThinkingLevel.high)) as Json)['thinkingLevel'],
        'LOW',
      );
    });

    test('uses mapped levels for token budgets', () async {
      expect(
        (thinkingConfig(
          await payload(
            flash({'xhigh': 'high'}, 'gemini-2.5-flash'),
            ThinkingLevel.xhigh,
            const ThinkingBudgets(high: 1234),
          ),
        ) as Json)['thinkingBudget'],
        1234,
      );
    });
  });

  group('raw stop reasons', () {
    final flash = getBuiltinModel('google', 'gemini-2.5-flash')!;

    List<Json> finish(String reason, {bool call = false}) {
      return [
        {
          'responseId': 'google-response-id',
          'candidates': [
            {
              'finishReason': reason,
              if (call)
                'content': {
                  'parts': [
                    {
                      'functionCall': {
                        'id': 'call-1',
                        'name': 'echo',
                        'args': {'value': 'truncated'},
                      },
                    },
                  ],
                },
            },
          ],
          'usageMetadata': {'promptTokenCount': 1, 'candidatesTokenCount': 0, 'totalTokenCount': 1},
        },
      ];
    }

    test('keeps the raw finish reason of an error', () async {
      final out = await run(flash, finish('MALFORMED_FUNCTION_CALL'));
      expect(
        (out.stopReason, out.rawStopReason, out.errorMessage),
        (StopReason.error, 'MALFORMED_FUNCTION_CALL', 'Provider stopped with: MALFORMED_FUNCTION_CALL'),
      );
    });

    test('keeps MAX_TOKENS with a tool call as length', () async {
      final out = await run(flash, finish('MAX_TOKENS', call: true));
      expect((out.stopReason, out.rawStopReason), (StopReason.length, 'MAX_TOKENS'));
      expect(out.content.whereType<ToolCall>(), hasLength(1));
    });

    test('maps STOP with a tool call to toolUse', () async {
      final out = await run(flash, finish('STOP', call: true));
      expect((out.stopReason, out.rawStopReason), (StopReason.toolUse, 'STOP'));
    });

    test('forwards each chunk in order before reading it', () async {
      final chunks = [
        {
          'responseId': 'resp_google',
          'candidates': [
            {
              'content': {
                'parts': [
                  {'text': 'hello'},
                ],
              },
            },
          ],
        },
        {
          'candidates': [
            {'finishReason': 'STOP'},
          ],
          'usageMetadata': {'promptTokenCount': 2, 'candidatesTokenCount': 1, 'totalTokenCount': 3},
        },
      ];
      final received = <Json>[];
      final models = <Model>[];
      final out = await run(
        flash,
        chunks,
        (client) => GoogleOptions(
          apiKey: 'test-api-key',
          httpClient: client,
          onEvent: (chunk, m) async {
            await Future<void>.value();
            received.add(chunk);
            models.add(m);
          },
        ),
      );
      expect(received, chunks);
      expect(models, [same(flash), same(flash)]);
      expect((out.stopReason, out.responseId), (StopReason.stop, 'resp_google'));
      expect(
        [for (final c in out.content) c.toJson()],
        [
          {'type': 'text', 'text': 'hello'},
        ],
      );
    });
  });

  group('tool conversion', () {
    Tool tool(Json parameters, [ConstrainedSampling? sampling]) {
      return Tool('test_tool', 'A test tool', parameters, constrainedSampling: sampling);
    }

    Object? schema(Json parameters) {
      return ((convertGoogleTools([tool(parameters)])!.single['functionDeclarations'] as List<Object?>).single
          as Json)['parametersJsonSchema'];
    }

    test(r'keeps $schema in parametersJsonSchema', () {
      const parameters = {
        r'$schema': 'http://json-schema.org/draft-07/schema#',
        'type': 'object',
        'properties': {
          'command': {'type': 'string'},
        },
        'required': ['command'],
      };
      expect(schema(parameters), parameters);
    });

    test('uses validated function calling for strict tools on Gemini 3', () {
      final strict = tool(const {
        'type': 'object',
        'properties': <String, Object?>{},
      }, const JsonSchemaSampling(Strictness.require));
      expect(supportsGoogleStrictToolSampling('gemini-3.1-pro-preview'), isTrue);
      expect(supportsGoogleStrictToolSampling('gemini-2.5-pro'), isFalse);
      expect(googleFunctionCallingMode([strict], null, true), 'VALIDATED');
      expect(
        () => googleFunctionCallingMode([strict], null, false),
        throwsA(
          isA<StateError>().having(
            (e) => e.message,
            'message',
            contains('Tool "test_tool" requires JSON-schema constrained sampling'),
          ),
        ),
      );
    });

    test('returns nothing for no tools', () {
      expect(convertGoogleTools([]), isNull);
    });

    for (final key in ['default', 'examples', 'enum', 'const', 'x-custom']) {
      test('keeps literal data under $key', () {
        final literal = {
          'const': 'literal-data',
          'properties': {
            'nested': {'const': 1},
          },
        };
        final parameters = {
          'type': 'object',
          'properties': {
            'value': {
              'type': 'object',
              key: const {'enum', 'examples'}.contains(key) ? [literal] : literal,
            },
          },
        };
        final original = jsonDecode(jsonEncode(parameters));
        expect(schema(parameters), original);
        expect(parameters, original);
      });
    }

    for (final key in [
      'properties',
      'patternProperties',
      r'$defs',
      'definitions',
      'dependentSchemas',
      'dependencies',
    ]) {
      test('keeps names while rewriting schemas in $key', () {
        final parameters = {
          'type': 'object',
          key: {
            '__proto__': {'const': 'x'},
            'const': {'const': 'y'},
            'properties': {'const': 'z'},
          },
        };
        final original = jsonDecode(jsonEncode(parameters));
        final expected = {
          'type': 'object',
          key: {
            '__proto__': {
              'enum': ['x'],
            },
            'const': {
              'enum': ['y'],
            },
            'properties': {
              'enum': ['z'],
            },
          },
        };
        final converted = schema(parameters);
        expect(converted, expected);
        expect(jsonEncode(converted), jsonEncode(expected));
        expect(parameters, original);
      });
    }

    test('rewrites only schema-bearing keywords', () {
      expect(
        schema({
          'type': 'object',
          'additionalProperties': {'const': 'x'},
          'unevaluatedProperties': false,
          'propertyNames': {'const': 'name'},
          'allOf': [
            {
              'if': {'const': 'x'},
              'then': {'const': 'y'},
              'else': {
                'not': {'const': 'z'},
              },
            },
          ],
          'properties': {
            'list': {
              'type': 'array',
              'items': [
                {'const': 1},
              ],
              'prefixItems': [
                {'const': 2},
              ],
              'additionalItems': {'const': 3},
              'contains': {'const': 4},
              'unevaluatedItems': {'const': 5},
            },
          },
          'dependencies': {
            'list': ['other'],
          },
        }),
        {
          'type': 'object',
          'additionalProperties': {
            'enum': ['x'],
          },
          'unevaluatedProperties': false,
          'propertyNames': {
            'enum': ['name'],
          },
          'allOf': [
            {
              'if': {
                'enum': ['x'],
              },
              'then': {
                'enum': ['y'],
              },
              'else': {
                'not': {
                  'enum': ['z'],
                },
              },
            },
          ],
          'properties': {
            'list': {
              'type': 'array',
              'items': [
                {
                  'enum': [1],
                },
              ],
              'prefixItems': [
                {
                  'enum': [2],
                },
              ],
              'additionalItems': {
                'enum': [3],
              },
              'contains': {
                'enum': [4],
              },
              'unevaluatedItems': {
                'enum': [5],
              },
            },
          },
          'dependencies': {
            'list': ['other'],
          },
        },
      );
    });

    test('rewrites string and number const values to single-value enums', () {
      final parameters = {
        'type': 'object',
        'properties': {
          'origin': {'type': 'string', 'const': 'zq-src-7'},
          'mode': {
            'anyOf': [
              {'type': 'string', 'const': 'a'},
              {'type': 'string', 'const': 'b'},
            ],
          },
          'level': {
            'oneOf': [
              {'const': 1},
              {'const': 2},
            ],
          },
          'flag': {'type': 'boolean', 'const': true},
          'list': {
            'type': 'array',
            'items': {'const': 'x'},
          },
          'const': {'type': 'string'},
        },
        'required': ['origin'],
      };
      expect(schema(parameters), {
        'type': 'object',
        'properties': {
          'origin': {
            'type': 'string',
            'enum': ['zq-src-7'],
          },
          'mode': {
            'anyOf': [
              {
                'type': 'string',
                'enum': ['a'],
              },
              {
                'type': 'string',
                'enum': ['b'],
              },
            ],
          },
          'level': {
            'oneOf': [
              {
                'enum': [1],
              },
              {
                'enum': [2],
              },
            ],
          },
          'flag': {'type': 'boolean', 'const': true},
          'list': {
            'type': 'array',
            'items': {
              'enum': ['x'],
            },
          },
          'const': {'type': 'string'},
        },
        'required': ['origin'],
      });
      expect((parameters['properties'] as Json)['origin'], {'type': 'string', 'const': 'zq-src-7'});
    });
  });

  test('streamSimple throws at once without an API key', () {
    expect(
      () => streamSimpleGoogle(model('test-model', provider: 'test-provider', reasoning: false), const []),
      throwsA(isA<ProviderException>().having((e) => e.message, 'message', 'No API key for provider: test-provider')),
    );
  });
}
