// Ported from pi's packages/ai/test/validation.test.ts and validation-probes.test.ts. The probes are recorded in
// fixtures/pi-validation-probes.json from pi's test file, with TypeBox building the schemas: each has the schema as
// JSON, the arguments and pi's outcome, "ok <coerced arguments>", "REJECT <first error line>" or "THROW <message>".

import 'dart:convert';
import 'dart:io';

import 'package:chorus/chorus.dart';
import 'package:pluto/pluto.dart' show SchemaException, deepEqual;
import 'package:test/test.dart';

/// The outcome of one call, in the probes' format.
String outcome(String name, Json schema, Json args) {
  try {
    return 'ok ${jsonEncode(validateToolArguments(Tool(name, '', schema), ToolCall('1', name, args)))}';
  } on StateError catch (e) {
    final lines = e.message.split('\n');
    return e.message.startsWith('Validation failed') ? 'REJECT ${lines[1].trim()}' : 'THROW ${lines.first}';
  } on SchemaException catch (e) {
    return 'THROW SchemaException: ${e.message}';
  }
}

/// Whether two outcomes agree: the same text, or both "ok" with equal JSON (JSON texts differ in escapes).
bool agree(String actual, String expected) {
  if (actual == expected) {
    return true;
  }

  if (!actual.startsWith('ok ') || !expected.startsWith('ok ')) {
    return false;
  }

  return deepEqual(jsonDecode(actual.substring(3)), jsonDecode(expected.substring(3)));
}

/// Where chorus differs from pi, by probe (its test name and arguments): the outcome here.
const differences = {
  'reviewer probes > bigint {"n":1} {"n":1}': 'REJECT - n: expected bigint, got integer',
  'reviewer probes > bigint {"n":"12"} {"n":"12"}': 'REJECT - n: expected bigint, got string',
  'reviewer probes (Sol) > BigInt from a JSON string {"n":"42"}': 'REJECT - n: expected bigint, got string',
  'reviewer probes (Sol) > Undefined from JSON null {"n":null}': 'REJECT - n: expected undefined, got null',
  'reviewer probes > format-url {"u":"not a url"} {"u":"not a url"}': 'ok {"u":"not a url"}',
  'reviewer probes > mcp-invalid-regex {"s":"y"} {"s":"y"}':
      'THROW SchemaException: Invalid pattern "[": Unterminated character class',
};

void main() {
  final probes = (jsonDecode(File('test/fixtures/pi-validation-probes.json').readAsStringSync()) as List)
      .cast<Map<String, Object?>>();
  group('pi probes', () {
    // pluto checks standard JSON Schema only: no bigint or undefined types (JavaScript values without a JSON form),
    // and no url format. An invalid pattern fails with Dart's own message. The probe that changes its schema after
    // the first call cannot be replayed from JSON; it is ported below.
    for (final p in probes.where((p) => !(p['test'] as String).contains('keeps the validator'))) {
      final name = '${p['test']} ${jsonEncode(p['args'])}';
      test(name, () {
        final actual = outcome(p['name'] as String, p['schema'] as Json, p['args'] as Json);
        expect(agree(actual, differences[name] ?? p['expected'] as String), isTrue, reason: 'got $actual');
      });
    }

    test('lists only differences that name a probe', () {
      final names = {for (final p in probes) '${p['test']} ${jsonEncode(p['args'])}'};
      expect(differences.keys.where((k) => !names.contains(k)), isEmpty);
    });

    test('keeps the validator of a schema object, as before, also when the schema is changed later', () {
      final n = <String, Object?>{'type': 'number', 'minimum': 0};
      final parameters = {
        'type': 'object',
        'properties': {'n': n},
        'required': ['n'],
      };
      expect(outcome('late', parameters, {'n': '4'}), 'ok {"n":4}');
      n['minimum'] = 10;
      expect(outcome('late', parameters, {'n': '4'}), 'ok {"n":4}');
    });
  });

  group('validateToolArguments', () {
    // pi's createToolCallWithPlainSchema: `{value}` under one required property.
    (Tool, ToolCall) plain(Object schema, Object? value) {
      final tool = Tool('echo', 'Echo tool', {
        'type': 'object',
        'properties': {'value': schema},
        'required': ['value'],
      });
      return (tool, ToolCall('tool-1', 'echo', {'value': value}));
    }

    test('coerces serialized plain JSON schemas', () {
      final cases = <(Object, Object?, Object?)>[
        ({'type': 'number'}, '42', 42),
        ({'type': 'number'}, true, 1),
        ({'type': 'number'}, null, 0),
        ({'type': 'integer'}, '42', 42),
        ({'type': 'boolean'}, 'true', true),
        ({'type': 'boolean'}, 'false', false),
        ({'type': 'boolean'}, 1, true),
        ({'type': 'boolean'}, 0, false),

        // TypeBox's conversion rules come first: null is "null" for a required string.
        ({'type': 'string'}, null, 'null'),
        ({'type': 'string'}, true, 'true'),
        ({'type': 'null'}, '', null),
        ({'type': 'null'}, 0, null),
        ({'type': 'null'}, false, null),
        (
          {
            'type': ['number', 'string'],
          },
          '1',
          '1',
        ),
        (
          {
            'type': ['boolean', 'number'],
          },
          '1',
          1,
        ),
        ({'type': 'boolean'}, '1', true),
        ({'type': 'boolean'}, '0', false),
        ({'type': 'null'}, 'null', null),
        ({'type': 'integer'}, '42.1', 42),
        (
          {
            'type': 'array',
            'items': {'type': 'number'},
          },
          '7',
          [7],
        ),
      ];
      for (final (schema, input, expected) in cases) {
        final (tool, call) = plain(schema, input);
        expect(validateToolArguments(tool, call), {'value': expected}, reason: '$schema: $input');
      }
    });

    test('treats null as omission for optional non-nullable properties', () {
      final tool = Tool('echo', 'Echo tool', {
        'type': 'object',
        'properties': {
          'path': {'type': 'string'},
          'offset': {'type': 'number'},
          'nullable': {
            'anyOf': [
              {'type': 'string'},
              {'type': 'null'},
            ],
          },
          'metadata': {
            'type': 'object',
            'properties': {
              'enabled': {'type': 'boolean'},
            },
          },
        },
        'required': ['path', 'metadata'],
      });
      final call = ToolCall('tool-1', 'echo', {
        'path': 'file.txt',
        'offset': null,
        'nullable': null,
        'metadata': {'enabled': null},
      });
      expect(validateToolArguments(tool, call), {'path': 'file.txt', 'nullable': null, 'metadata': {}});
    });

    test('preserves optional nulls whose referenced schema is nullable', () {
      final tool = Tool('echo', 'Echo tool', {
        'type': 'object',
        'properties': {
          'value': {r'$ref': r'#/$defs/value'},
        },
        r'$defs': {
          'value': {
            'anyOf': [
              {'type': 'number'},
              {'type': 'null'},
            ],
          },
        },
      });
      expect(validateToolArguments(tool, ToolCall('tool-1', 'echo', {'value': null})), {'value': null});
    });

    test('preserves a value that already matches a nullable union arm', () {
      for (final keyword in ['anyOf', 'oneOf']) {
        final (tool, call) = plain({
          keyword: [
            {'type': 'number'},
            {'type': 'null'},
          ],
        }, null);
        expect(validateToolArguments(tool, call), {'value': null});
      }
    });

    test('still coerces nullable unions when the original value does not match any arm', () {
      final (tool, call) = plain({
        'anyOf': [
          {'type': 'number'},
          {'type': 'null'},
        ],
      }, '42');
      expect(validateToolArguments(tool, call), {'value': 42});
    });

    test('accepts null for nullable array schemas with items', () {
      final (tool, call) = plain({
        'type': ['array', 'null'],
        'items': {'type': 'string'},
      }, null);
      expect(validateToolArguments(tool, call), {'value': null});
    });

    test('rejects invalid coercions for serialized plain JSON schemas', () {
      for (final (schema, input) in [
        ({'type': 'boolean'}, 'yes'),
        ({'type': 'number'}, 'abc'),
        ({'type': 'null'}, 'nil'),
        ({'type': 'integer'}, '4x'),
      ]) {
        final (tool, call) = plain(schema, input);
        expect(() => validateToolArguments(tool, call), throwsA(_message(startsWith('Validation failed'))));
      }
    });

    test("lists pluto's errors, then the arguments it received", () {
      final (tool, call) = plain({'type': 'number'}, 'many');
      expect(
        () => validateToolArguments(tool, call),
        throwsA(
          _message(
            'Validation failed for tool "echo":\n  - value: expected number, got string\n\n'
            'Received arguments:\n{\n  "value": "many"\n}',
          ),
        ),
      );
    });

    test('reports a missing required property at the property', () {
      final (tool, _) = plain({'type': 'string'}, 'x');
      expect(
        () => validateToolArguments(tool, ToolCall('tool-1', 'echo', {})),
        throwsA(_message(contains('  - value: missing'))),
      );
    });

    test('returns a copy of the arguments, also when nothing is coerced', () {
      final (tool, call) = plain({'type': 'string'}, 'x');
      final arguments = validateToolArguments(tool, call);
      expect(arguments, {'value': 'x'});
      expect(arguments, isNot(same(call.arguments)));
    });

    // pi #10897
    test('reports malformed arguments JSON instead of validating empty arguments', () {
      final tool = Tool('echo', 'Echo tool', {
        'type': 'object',
        'properties': {
          'value': {'type': 'string'},
        },
      });
      expect(
        () => validateToolArguments(tool, ToolCall('tool-1', 'echo', {}, argumentsError: 'Unexpected token')),
        throwsA(_message('Invalid arguments for tool "echo": Unexpected token')),
      );
    });

    test('validateToolCall finds the tool by name', () {
      final (tool, call) = plain({'type': 'number'}, '1');
      expect(validateToolCall([tool], call), {'value': 1});
      expect(() => validateToolCall([], call), throwsA(_message('Tool "echo" not found')));
    });
  });
}

Matcher _message(Object matcher) => isA<StateError>().having((e) => e.message, 'message', matcher);
