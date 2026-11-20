// Strict tools. Ported from pi's constrained-sampling.test.ts (the JSON-schema tests), anthropic-strict-tool-schema
// .test.ts and anthropic-strict-tool-fallback.test.ts (without the inline tool additions, which come with native tool
// changes, and pi's gateway API).

import 'dart:convert';

import 'package:anthropic_sdk_lite_dart/anthropic_sdk_lite_dart.dart';
import 'package:chorus/chorus.dart';
import 'package:chorus/src/ai/constrained.dart';
import 'package:chorus/src/ai/providers/anthropic.dart' show buildParams;
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:test/test.dart';

import 'anthropic_fake.dart';

const grammarError =
    'The compiled grammar is too large, which would cause performance issues. Simplify your tool schemas or reduce '
    'the number of strict tools.';

Json object(Json properties, {List<String>? required, Json extra = const {}}) {
  return {'type': 'object', 'properties': properties, 'required': required ?? properties.keys.toList(), ...extra};
}

/// pi's tool: a required string `value` and an optional number `optional`, with [strict] sampling when given.
Tool tool(String name, [Strictness? strict]) {
  return Tool(
    name,
    'Look up a value',
    object(
      {
        'value': {'type': 'string'},
        'optional': {'type': 'number'},
      },
      required: ['value'],
    ),
    constrainedSampling: strict == null ? null : JsonSchemaSampling(strict),
  );
}

List<Message> transcript(List<Tool> tools) {
  return [SystemMessage(const [], added: tools, timestamp: 0), user('Hello')];
}

String errorBody(String message, [String type = 'invalid_request_error']) {
  return jsonEncode({
    'type': 'error',
    'error': {'type': type, 'message': message},
  });
}

String success() => sse(minimalEvents());

final model = claude(id: 'claude-haiku-4-5', reasoning: false, compat: const {'supportsStrictTools': true});

/// The `strict` of each tool of each request [server] received.
List<List<Object?>> stricts(FakeServer server) {
  return [
    for (final r in server.requests)
      [for (final t in (jsonDecode(r.body) as Json)['tools'] as List<Object?>) (t as Json)['strict']],
  ];
}

Future<AssistantMessage> run(FakeServer server, List<Tool> tools, {String? sessionId, AbortSignal? signal}) {
  return streamAnthropic(
    model,
    transcript(tools),
    AnthropicOptions(client: server.client(), maxRetries: 0, sessionId: sessionId, signal: signal),
  ).result;
}

void main() {
  tearDown(cleanupSessionResources);

  group('makeStrictJsonSchema', () {
    test('derives strict provider schemas without changing tool definitions', () {
      final parameters = object(
        {
          'path': {'type': 'string'},
          'offset': {'type': 'number'},
          'metadata': object({
            'enabled': {'type': 'boolean'},
          }, required: []),
          'nullable': {
            'anyOf': [
              {'type': 'string'},
              {'type': 'null'},
            ],
          },
        },
        required: ['path', 'metadata'],
      );
      final before = jsonEncode(parameters);
      final strict = makeStrictJsonSchema(parameters);
      expect(jsonEncode(parameters), before);
      expect(strict['additionalProperties'], false);
      expect(strict['required'], ['path', 'offset', 'metadata', 'nullable']);
      final properties = strict['properties'] as Json;
      expect(properties['offset'], {
        'anyOf': [
          {'type': 'number'},
          {'type': 'null'},
        ],
      });
      expect(properties['metadata'], {
        'type': 'object',
        'properties': {
          'enabled': {
            'anyOf': [
              {'type': 'boolean'},
              {'type': 'null'},
            ],
          },
        },
        'required': ['enabled'],
        'additionalProperties': false,
      });
      expect(properties['nullable'], {
        'anyOf': [
          {'type': 'string'},
          {'type': 'null'},
        ],
      });
    });

    test('keeps strict mode for enum-only schemas', () {
      final t = Tool(
        'sample_tool',
        'Sample tool',
        object({
          'level': {
            'enum': ['low', 'high'],
          },
        }),
        constrainedSampling: const JsonSchemaSampling(Strictness.prefer),
      );
      expect(resolveStrictSampling(t, true), isTrue);
    });

    test('falls back or rejects schemas that cannot be safely converted', () {
      final cases = <(Json, String)>[
        (
          object({
            'metadata': object(
              {},
              extra: const {
                'additionalProperties': {'type': 'string'},
              },
            ),
          }),
          'additionalProperties is unsupported',
        ),
        (
          {
            'allOf': [
              object({
                'a': {'type': 'string'},
              }),
              object({
                'b': {'type': 'number'},
              }),
            ],
          },
          'allOf schemas are unsupported',
        ),
        (
          object({
            'value': {
              'anyOf': [
                object({
                  'nested': {'type': 'string'},
                }),
                {'type': 'null'},
              ],
            },
          }),
          'object and array unions are unsupported',
        ),
        (
          object({
            'child': {r'$ref': 'https://example.com/child.json'},
          }),
          r'$ref schemas are unsupported',
        ),
        (
          object({
            'commandArguments': {'type': 'array', 'items': <String, Object?>{}},
          }, required: []),
          'schemas without a type are unsupported',
        ),
        (
          object({
            'value': {'description': 'Any JSON value'},
          }),
          'schemas without a type are unsupported',
        ),
        (
          object({
            'mode': {'const': 'fast'},
          }),
          'schemas without a type are unsupported',
        ),
      ];
      for (final (parameters, error) in cases) {
        expect(
          () => makeStrictJsonSchema(parameters),
          throwsA(isA<UnsupportedStrictSchema>().having((e) => e.message, 'message', contains(error))),
        );
        final prefer = Tool('t', 'd', parameters, constrainedSampling: const JsonSchemaSampling(Strictness.prefer));
        expect(resolveStrictSampling(prefer, true), isNull);
        final require = Tool('t', 'd', parameters, constrainedSampling: const JsonSchemaSampling(Strictness.require));
        expect(() => resolveStrictSampling(require, true), throwsA(isA<StateError>()));
      }
    });

    test('rejects a required strict tool when strict tools are unsupported', () {
      expect(
        () => resolveStrictSampling(tool('lookup', Strictness.require), false),
        throwsA(
          isA<StateError>().having((e) => e.message, 'message', contains('requires JSON-schema constrained sampling')),
        ),
      );
      expect(resolveStrictSampling(tool('lookup', Strictness.prefer), false), isNull);
      expect(resolveStrictSampling(tool('lookup'), true), isNull);
    });
  });

  group('Anthropic strict tool schemas', () {
    Json first(Tool t) {
      final strictModel = claude(
        id: 'claude-opus-4-8',
        compat: const {'forceAdaptiveThinking': true, 'supportsStrictTools': true},
      );
      final params = buildParams(
        strictModel,
        transcript([t]),
        const AnthropicOptions(cacheRetention: CacheRetention.none),
      );
      return (params['tools'] as List<Object?>).first as Json;
    }

    test('only sends the full input schema for strict JSON-schema tools', () {
      final legacy = Tool(
        'lookup',
        'Look up a value',
        object(
          {
            'value': {'type': 'string'},
          },
          extra: const {'additionalProperties': false, 'title': 'LookupInput'},
        ),
      );
      final legacyTool = first(legacy);
      expect(legacyTool.containsKey('strict'), isFalse);
      expect(legacyTool['input_schema'], {
        'type': 'object',
        'properties': {
          'value': {'type': 'string'},
        },
        'required': ['value'],
        'additionalProperties': false,
      });

      final strictTool = first(
        Tool(
          'lookup',
          'Look up a value',
          object(
            {
              'value': {'type': 'string'},
              'optional': {'type': 'number'},
            },
            required: ['value'],
            extra: const {'title': 'StrictLookupInput'},
          ),
          constrainedSampling: const JsonSchemaSampling(Strictness.prefer),
        ),
      );
      expect(strictTool['strict'], isTrue);
      final schema = strictTool['input_schema'] as Json;
      expect(schema['additionalProperties'], false);
      expect(schema['required'], ['value', 'optional']);
      expect(schema['title'], 'StrictLookupInput');
      expect((schema['properties'] as Json)['optional'], {
        'anyOf': [
          {'type': 'number'},
          {'type': 'null'},
        ],
      });
    });

    // pi #9557
    test('keeps root definitions in non-strict schemas so property refs resolve', () {
      final settings = object({
        'verbose': {'type': 'boolean'},
      }, required: []);
      for (final (ref, definitions) in [
        (
          r'#/$defs/Settings',
          {
            r'$defs': {'Settings': settings},
          },
        ),
        (
          '#/definitions/Settings',
          {
            'definitions': {'Settings': settings},
          },
        ),
      ]) {
        final parameters = object({
          'settings': {r'$ref': ref},
        }, extra: definitions);
        final t = first(Tool('lookup', 'Look up a value', parameters));
        expect(t.containsKey('strict'), isFalse);
        expect(t['input_schema'], parameters);
      }
    });

    // pi #9953
    test('sends prefer tools non-strict when they use keywords Anthropic strict mode rejects', () {
      Tool prefer(Json parameters) {
        return Tool('lookup', 'd', parameters, constrainedSampling: const JsonSchemaSampling(Strictness.prefer));
      }

      for (final parameters in [
        object({
          'timeoutMs': {'type': 'integer', 'minimum': 1, 'maximum': 300000},
        }, required: []),
        object({
          'options': object({
            'tags': {
              'type': 'array',
              'items': {'type': 'string'},
              'minItems': 2,
            },
          }),
        }),
        object({
          'expression': {'type': 'string', 'format': 'regex'},
        }),
      ]) {
        expect(first(prefer(parameters)).containsKey('strict'), isFalse, reason: jsonEncode(parameters));
      }

      final supported = first(
        prefer(
          object({
            'code': {'type': 'string', 'minLength': 1, 'maxLength': 1000, 'pattern': r'^[a-z]+$'},
            'url': {'type': 'string', 'format': 'uri'},
            'tags': {
              'type': 'array',
              'items': {'type': 'string'},
              'minItems': 1,
            },
          }),
        ),
      );
      expect(supported['strict'], isTrue);
    });
  });

  // pi #10409
  group('Anthropic preferred strict tool fallback', () {
    for (final message in [
      grammarError,
      'Too many strict tools (32). The maximum number of strict tools supported is 20. Try reducing the number of '
          'tools marked as strict.',
      'Schemas contains too many parameters with union types (480 parameters with type arrays or anyOf). This causes '
          'exponential compilation cost. Reduce the number of nullable or union-typed parameters (limit: 16 '
          'parameters with unions).',
    ]) {
      test('retries "${message.substring(0, 30)}..." with the original non-strict schemas', () async {
        final server = FakeServer.sequence([(400, errorBody(message)), (200, success())]);
        final tools = [tool('preferred', Strictness.prefer), tool('required', Strictness.require), tool('ordinary')];
        final messages = transcript(tools);
        final original = jsonEncode([for (final m in messages) m.toJson()]);
        var payloads = 0;
        var responses = 0;
        final stream = streamAnthropic(
          model,
          messages,
          AnthropicOptions(
            client: server.client(),
            maxRetries: 0,
            cacheRetention: CacheRetention.long,
            onPayload: (payload, _) {
              payloads++;
              return {...payload, 'inference_geo': 'us', 'stream': false};
            },
            onResponse: (_, _) => responses++,
          ),
        );
        final events = await stream.map((e) => e.runtimeType).toList();
        final result = await stream.result;

        expect(result.stopReason, StopReason.stop);
        expect(events.first, StartEvent);
        expect(events.last, DoneEvent);
        expect((server.requests.length, payloads, responses), (2, 2, 1));
        final bodies = [for (final r in server.requests) jsonDecode(r.body) as Json];
        final first = bodies[0]['tools'] as List<Object?>;
        final second = bodies[1]['tools'] as List<Object?>;
        expect((first[0] as Json)['strict'], isTrue);
        expect(((first[0] as Json)['input_schema'] as Json)['required'], ['value', 'optional']);
        expect((second[0] as Json).containsKey('strict'), isFalse);
        expect((second[0] as Json)['input_schema'], {
          'type': 'object',
          'properties': {
            'value': {'type': 'string'},
            'optional': {'type': 'number'},
          },
          'required': ['value'],
        });
        expect(jsonEncode(second[1]), jsonEncode(first[1]));
        expect((second[1] as Json)['strict'], isTrue);
        expect(jsonEncode(second[2]), jsonEncode(first[2]));
        expect((second[2] as Json)['cache_control'], {'type': 'ephemeral', 'ttl': '1h'});
        expect(bodies.map((b) => b['stream']), [true, true]);
        expect(bodies[1]['inference_geo'], 'us');
        expect(jsonEncode([for (final m in messages) m.toJson()]), original);
        expect(result.diagnostics?.first.type, 'anthropic_strict_tools_fallback');
        expect(result.diagnostics?.first.details, {'reason': 'strict_tool_limits_exceeded'});
      });
    }

    for (final (name, tools) in [
      ('required', [tool('lookup', Strictness.require)]),
      ('ordinary', [tool('lookup')]),
      (
        'locally unsupported prefer',
        [
          Tool(
            'lookup',
            'd',
            object({
              'value': {'type': 'number', 'minimum': 1},
            }),
            constrainedSampling: const JsonSchemaSampling(Strictness.prefer),
          ),
        ],
      ),
    ]) {
      test('does not retry a $name tool set', () async {
        final server = FakeServer(errorBody(grammarError), status: 400);
        final result = await run(server, tools);
        expect(result.stopReason, StopReason.error);
        expect(result.errorMessage, contains(grammarError));
        expect(server.requests, hasLength(1));
      });
    }

    for (final (name, message, status, type) in [
      ('unrelated 400', 'Invalid tool schema', 400, 'invalid_request_error'),
      ('wrong error type', grammarError, 400, 'authentication_error'),
      ('wrong HTTP status', grammarError, 500, 'invalid_request_error'),
    ]) {
      test('does not downgrade on $name', () async {
        final server = FakeServer(errorBody(message, type), status: status);
        final result = await run(server, [tool('lookup', Strictness.prefer)]);
        expect(result.stopReason, StopReason.error);
        expect(server.requests, hasLength(1));
      });
    }

    test('surfaces a second rejection rather than retrying again', () async {
      final server = FakeServer.sequence([
        (400, errorBody(grammarError)),
        (400, errorBody('Required grammar is still too large')),
      ]);
      final result = await run(server, [tool('lookup', Strictness.prefer), tool('required', Strictness.require)]);
      expect(result.stopReason, StopReason.error);
      expect(result.errorMessage, contains('Required grammar is still too large'));
      expect(stricts(server), [
        [true, true],
        [null, true],
      ]);
    });

    test('does not downgrade a hook exception, even if it looks like the provider rejection', () async {
      final server = FakeServer(success());
      final result = await streamAnthropic(
        model,
        transcript([tool('lookup', Strictness.prefer)]),
        AnthropicOptions(
          client: server.client(),
          maxRetries: 0,
          onPayload: (_, _) => throw APIError(400, {
            'error': {'type': 'invalid_request_error', 'message': grammarError},
          }, const {}),
        ),
      ).result;
      expect(result.stopReason, StopReason.error);
      expect(server.requests, isEmpty);
      expect(result.diagnostics, isNull);
    });

    test('honors cancellation before the fallback attempt', () async {
      final signal = AbortSignal();
      var requests = 0;
      final client = Anthropic(
        apiKey: 'test-key',
        baseURL: 'https://api.anthropic.com',
        httpClient: MockClient((request) async {
          requests++;
          signal.abort();
          return http.Response(errorBody(grammarError), 400);
        }),
      );
      final result = await streamAnthropic(
        model,
        transcript([tool('lookup', Strictness.prefer)]),
        AnthropicOptions(client: client, maxRetries: 0, signal: signal),
      ).result;
      expect(result.stopReason, StopReason.aborted);
      expect(requests, 1);
    });

    test('does not retry an SSE error after response headers', () async {
      final server = FakeServer(
        sse([
          {
            'type': 'error',
            'error': {'type': 'invalid_request_error', 'message': grammarError},
          },
        ]),
      );
      final result = await run(server, [tool('lookup', Strictness.prefer)]);
      expect(result.stopReason, StopReason.error);
      expect(server.requests, hasLength(1));
    });

    test('does not disable strict sampling on later requests without a session', () async {
      final server = FakeServer.sequence([(400, errorBody(grammarError)), (200, success())]);
      await run(server, [tool('lookup', Strictness.prefer)]);
      await run(server, [tool('lookup', Strictness.prefer)]);
      expect(stricts(server), [
        [true],
        [null],
        [true],
      ]);
    });

    // pi #11626
    test('sends preferred tools non-strict right away on later requests of a rejected session', () async {
      final server = FakeServer.sequence([(400, errorBody(grammarError)), (200, success())]);
      final tools = [tool('lookup', Strictness.prefer), tool('required', Strictness.require)];
      final first = await run(server, tools, sessionId: 'rejected');
      final second = await run(server, tools, sessionId: 'rejected');
      await run(server, tools, sessionId: 'other');
      expect(first.diagnostics?.first.type, 'anthropic_strict_tools_fallback');
      expect(second.stopReason, StopReason.stop);
      expect(second.diagnostics, isNull);
      expect(stricts(server), [
        [true, true],
        [null, true],
        [null, true],
        [true, true],
      ]);
    });

    test('tries strict sampling again after the session\'s resources are cleaned up', () async {
      final server = FakeServer.sequence([(400, errorBody(grammarError)), (200, success())]);
      await run(server, [tool('lookup', Strictness.prefer)], sessionId: 'cleaned');
      cleanupSessionResources('cleaned');
      await run(server, [tool('lookup', Strictness.prefer)], sessionId: 'cleaned');
      expect(stricts(server), [
        [true],
        [null],
        [true],
      ]);
    });

    test('surfaces a rejection of a remembered session\'s required tools without retrying', () async {
      final server = FakeServer.sequence([
        (400, errorBody(grammarError)),
        (200, success()),
        (400, errorBody(grammarError)),
      ]);
      final tools = [tool('lookup', Strictness.prefer), tool('required', Strictness.require)];
      await run(server, tools, sessionId: 'required');
      final result = await run(server, tools, sessionId: 'required');
      expect(result.stopReason, StopReason.error);
      expect(result.errorMessage, contains(grammarError));
      expect(server.requests, hasLength(3));
    });
  });

  test('reads and writes constrained sampling and diagnostics as pi JSON', () {
    final t = Tool.fromJson({
      'name': 'n',
      'description': 'd',
      'parameters': {'type': 'object'},
      'constrainedSampling': {'type': 'json_schema', 'strict': 'require'},
    });
    expect((t.constrainedSampling as JsonSchemaSampling).strict, Strictness.require);
    expect(t.toJson()['constrainedSampling'], {'type': 'json_schema', 'strict': 'require'});
    final grammar = Tool.fromJson({
      'name': 'n',
      'description': 'd',
      'parameters': {'type': 'object'},
      'constrainedSampling': {
        'type': 'grammar',
        'variants': {'openai_lark': 'start: /a/'},
      },
    });
    expect(grammar.toJson()['constrainedSampling'], {
      'type': 'grammar',
      'variants': {'openai_lark': 'start: /a/'},
    });
    expect(
      Tool.fromJson({'name': 'n', 'description': 'd', 'parameters': <String, Object?>{}, 'constrainedSampling': false})
          .constrainedSampling,
      isNull,
    );

    final message = AssistantMessage(api: 'a', provider: 'p', model: 'm', timestamp: 1)
      ..diagnostics = [
        const Diagnostic(type: 't', timestamp: 2, error: {'message': 'x'}, details: {'reason': 'r'}),
      ];
    final copy = AssistantMessage.fromJson(message.toJson());
    expect(copy.diagnostics?.single.toJson(), {
      'type': 't',
      'timestamp': 2,
      'error': {'message': 'x'},
      'details': {'reason': 'r'},
    });
  });
}
