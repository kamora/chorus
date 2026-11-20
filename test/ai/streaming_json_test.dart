// Ported from pi's packages/ai/test/streaming-json.test.ts. Left out: the comparison with pi's earlier
// parseStreamingJson (chorus never had it) and the `__proto__` checks (a JavaScript prototype matter). pi's allocation
// measurements (V8 heap counters) are timing checks here, with one more for a long open string, which a Dart string
// makes costly where a JavaScript one does not.

import 'dart:convert';
import 'dart:io';

import 'package:chorus/src/ai/json.dart';
import 'package:chorus/src/ai/streaming_json.dart';
import 'package:test/test.dart';

/// Seeded random numbers, so every run checks the same documents: pi's generator, with JavaScript's arithmetic (the
/// product is a double, which loses its low bits, before the bitwise and), so the documents are pi's own.
double Function() random(int seed) {
  var state = seed;
  return () {
    state = (state.toDouble() * 1103515245 + 12345).toInt() & 0x7fffffff;
    return state / 0x7fffffff;
  };
}

typedef Random = double Function();

T pick<T>(Random rand, List<T> items) => items[(rand() * items.length).floor()];

const stringPieces = [
  'a',
  'b',
  'x y',
  ' ',
  '\n',
  '\t',
  r'\',
  '"',
  '/',
  'é',
  '😀',
  '\u0001',
  '\u001f',
  ' ',
  '{',
  ']',
  ',',
];

String randomString(Random rand) {
  final text = StringBuffer();
  final length = (rand() * 6).floor();
  for (var i = 0; i < length; i++) {
    text.write(pick(rand, stringPieces));
  }

  return text.toString();
}

Object? randomScalar(Random rand) {
  return pick<Object? Function()>(rand, [
    () => randomString(rand),
    () => (rand() * 2000).floor() - 1000,
    () => rand() * 100,
    () => 1e21,
    () => -5e-8,
    () => -0.0,
    () => 0,
    () => true,
    () => false,
    () => null,
  ])();
}

Object? randomValue(Random rand, int depth) {
  final roll = rand();
  if (depth > 3 || roll < 0.3) {
    return randomScalar(rand);
  }

  if (roll < 0.65) {
    final members = <String, Object?>{};
    final count = (rand() * 4).floor();
    for (var i = 0; i < count; i++) {
      members[randomString(rand)] = randomValue(rand, depth + 1);
    }

    return members;
  }

  return List.generate((rand() * 4).floor(), (_) => randomValue(rand, depth + 1));
}

/// JSON text of [value], with random insignificant whitespace when [spaced].
String render(Object? value, Random rand, bool spaced) {
  String space() => spaced ? pick(rand, ['', '', ' ', '\n  ', '\t', '\r\n']) : '';
  if (value is List) {
    // The items render before the separator's whitespace is drawn, as in JavaScript.
    return '[${space()}${value.map((v) => render(v, rand, spaced)).toList().join('${space()},${space()}')}${space()}]';
  }

  if (value is Map) {
    final members = [
      for (final MapEntry(:key, :value) in value.entries)
        '${jsonEncode(key)}${space()}:${space()}${render(value, rand, spaced)}',
    ];
    return '{${space()}${members.join('${space()},${space()}')}${space()}}';
  }

  return jsonEncode(value);
}

List<String> randomDocuments(int seed, int count, bool spaced) {
  final rand = random(seed);
  return [
    for (var i = 0; i < count; i++)
      render(
        rand() < 0.8 ? {'path': randomString(rand), 'args': randomValue(rand, 0)} : randomValue(rand, 0),
        rand,
        spaced,
      ),
  ];
}

// The reference: a plain recursive reading of the whole prefix of valid JSON (strings may also hold raw control
// characters and invalid escapes) that follows the documented rules, written independently of the incremental parser
// so the two check each other.

/// A value read, whether it ended, and where.
typedef Read = ({Object? value, bool complete, int end});

/// Stands for "no value yet", where null is a value.
const none = '<none>';

var pendingScalars = true;

Object? referenceSnapshot(String text, {bool pending = true}) {
  pendingScalars = pending;
  final read = readValue(text, skipSpace(text, 0));
  return read == null ? none : read.value;
}

int skipSpace(String text, int start) {
  var index = start;
  while (index < text.length && ' \t\n\r'.contains(text[index])) {
    index++;
  }

  return index;
}

Read? readValue(String text, int start) {
  if (start >= text.length) {
    return null;
  }

  return switch (text[start]) {
    '"' => readString(text, start),
    '{' => readObject(text, start),
    '[' => readArray(text, start),
    _ => readScalar(text, start),
  };
}

const simple = {'"': '"', r'\': r'\', '/': '/', 'b': '\b', 'f': '\f', 'n': '\n', 'r': '\r', 't': '\t'};

Read readString(String text, int start) {
  final value = StringBuffer();
  var index = start + 1;
  while (index < text.length) {
    final char = text[index];
    if (char == '"') {
      return (value: value.toString(), complete: true, end: index + 1);
    }

    if (char != r'\') {
      value.write(char);
      index++;
      continue;
    }

    if (index + 1 >= text.length) {
      return (value: value.toString(), complete: false, end: text.length);
    }

    final next = text[index + 1];
    if (simple[next] != null) {
      value.write(simple[next]);
      index += 2;
      continue;
    }

    if (next == 'u') {
      var end = index + 2;
      while (end < index + 6 && end < text.length && RegExp('[0-9a-fA-F]').hasMatch(text[end])) {
        end++;
      }

      if (end == index + 6) {
        value.writeCharCode(int.parse(text.substring(index + 2, end), radix: 16));
        index = end;
        continue;
      }

      if (end == text.length) {
        return (value: value.toString(), complete: false, end: end);
      }

      // Not four hex digits: the escape stays literal and the next character is read normally.
      value.write(text.substring(index, end));
      index = end;
      continue;
    }

    // An invalid escape: the backslash stays literal and the next character is read normally.
    value.write(r'\');
    index++;
  }

  return (value: value.toString(), complete: false, end: text.length);
}

Read readObject(String text, int start) {
  final members = <String, Object?>{};
  final partial = (value: members, complete: false, end: text.length);
  var index = skipSpace(text, start + 1);
  if (index >= text.length) {
    return partial;
  }

  if (text[index] == '}') {
    return (value: members, complete: true, end: index + 1);
  }

  for (;;) {
    final key = readString(text, index);
    if (!key.complete) {
      return partial;
    }

    index = skipSpace(text, key.end);
    if (index >= text.length) {
      return partial;
    }

    index = skipSpace(text, index + 1);
    final member = readValue(text, index);
    if (member == null) {
      return partial;
    }

    members[key.value as String] = member.value;
    if (!member.complete) {
      return partial;
    }

    index = skipSpace(text, member.end);
    if (index >= text.length) {
      return partial;
    }

    if (text[index] == '}') {
      return (value: members, complete: true, end: index + 1);
    }

    index = skipSpace(text, index + 1);
    if (index >= text.length) {
      return partial;
    }
  }
}

Read readArray(String text, int start) {
  final elements = <Object?>[];
  final partial = (value: elements, complete: false, end: text.length);
  var index = skipSpace(text, start + 1);
  if (index >= text.length) {
    return partial;
  }

  if (text[index] == ']') {
    return (value: elements, complete: true, end: index + 1);
  }

  for (;;) {
    final element = readValue(text, index);
    if (element == null) {
      return partial;
    }

    elements.add(element.value);
    if (!element.complete) {
      return partial;
    }

    index = skipSpace(text, element.end);
    if (index >= text.length) {
      return partial;
    }

    if (text[index] == ']') {
      return (value: elements, complete: true, end: index + 1);
    }

    index = skipSpace(text, index + 1);
    if (index >= text.length) {
      return partial;
    }
  }
}

Read? readScalar(String text, int start) {
  var end = start;
  while (end < text.length && !' \t\n\r,:{}[]"'.contains(text[end])) {
    end++;
  }

  final token = text.substring(start, end);
  if (end < text.length) {
    return (value: jsonDecode(token), complete: true, end: end);
  }

  if (!pendingScalars) {
    return null;
  }

  // At the end of the text: a literal prefix, a number, or a number whose exponent has no digits yet.
  for (final (literal, value) in [('true', true), ('false', false), ('null', null)]) {
    if (literal.startsWith(token)) {
      return (value: value, complete: false, end: end);
    }
  }

  if (RegExp(r'^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?$').hasMatch(token)) {
    return (value: num.parse(token), complete: false, end: end);
  }

  final mantissa = RegExp(r'^(-?(?:0|[1-9]\d*)(?:\.\d+)?)[eE][+-]?$').firstMatch(token);
  return mantissa == null ? null : (value: num.parse(mantissa[1]!), complete: false, end: end);
}

/// A snapshot as a comparable value: [none] before a value starts.
Object? shown(({Object? value})? snapshot) => snapshot == null ? none : snapshot.value;

/// A deep copy of a JSON value.
Object? copy(Object? value) {
  return switch (value) {
    Map() => {for (final MapEntry(:key, :value) in value.entries) key: copy(value)},
    List() => [for (final v in value) copy(v)],
    _ => value,
  };
}

typedef Step = ({String read, Object? snapshot, Object? copy});

/// Feeds [text] in pieces of [size] (or random sizes); returns each snapshot, a copy taken right then, and the text
/// read by then.
List<Step> stream(String text, Object size) {
  final parser = StreamingJsonParser();
  final steps = <Step>[];
  for (var index = 0; index < text.length;) {
    final length = size is int ? size : 1 + ((size as Random)() * 12).floor();
    final end = index + length < text.length ? index + length : text.length;
    parser.append(text.substring(index, end));
    index = end;
    final snapshot = shown(parser.snapshot());
    steps.add((read: text.substring(0, index), snapshot: snapshot, copy: copy(snapshot)));
  }

  return steps;
}

Object? finalSnapshot(String text, [int? size]) {
  final steps = stream(text, size ?? (text.isEmpty ? 1 : text.length));
  return steps.isEmpty ? none : steps.last.snapshot;
}

/// A stable seed from the text, so random chunk sizes differ per document.
int seedOf(String text) {
  var hash = 7;
  for (final c in text.codeUnits) {
    hash = (hash * 31 + c).toSigned(32);
  }

  return hash.abs() + 1;
}

const chunkSizes = <Object>[1, 2, 3, 7, 'random'];

Object sizeFor(Object size, String text) => size == 'random' ? random(seedOf(text)) : size;

/// Asserts [value] is plain JSON data.
void expectJson(Object? value, String label) {
  switch (value) {
    case null || String() || bool() || num():
      return;
    case Map():
      expect(value, isA<Map<String, Object?>>(), reason: label);
      for (final v in value.values) {
        expectJson(v, label);
      }
    case List():
      for (final v in value) {
        expectJson(v, label);
      }
    default:
      if (value != none) {
        fail('$label: not JSON: $value');
      }
  }
}

/// A copy of a value kept current by applying the parser's changes, as a writer keeps one.
class Mirror {
  Object? value = none;

  void apply(List<JsonChange> changes) {
    for (final change in changes) {
      final path = change.path;
      Object? parent = value;
      for (final segment in path.isEmpty ? const <Object>[] : path.sublist(0, path.length - 1)) {
        parent = segment is int ? (parent as List)[segment] : (parent as Map)[segment];
      }

      final key = path.lastOrNull;
      final Object? next;
      switch (change) {
        case JsonSet(:final value):
          next = copy(value);
        case JsonAppend(:final text):
          final old = key == null ? value : (key is int ? (parent as List)[key] : (parent as Map)[key]);
          next = '${old as String}$text';
      }

      if (key == null) {
        value = next;
      } else if (key is int) {
        final list = parent as List;
        if (key == list.length) {
          list.add(next);
        } else {
          list[key] = next;
        }
      } else {
        (parent as Map)[key] = next;
      }
    }
  }
}

/// Feeds [text] into a parser and a mirror of its changes; checks the mirror after each piece.
Mirror mirrorStream(String text, Object size) {
  final parser = StreamingJsonParser();
  final mirror = Mirror();
  for (var index = 0; index < text.length;) {
    final length = size is int ? size : 1 + ((size as Random)() * 12).floor();
    final end = index + length < text.length ? index + length : text.length;
    mirror.apply(parser.append(text.substring(index, end)));
    index = end;

    // Changes leave out unterminated numbers and literals, which snapshots show.
    expect(mirror.value, referenceSnapshot(text.substring(0, index), pending: false), reason: text.substring(0, index));
  }

  return mirror;
}

List<Json> changes(List<JsonChange> list) => [for (final c in list) c.toJson()];

typedef Json = Map<String, Object?>;

// Expected values of inputs, from other partial-JSON parsers' public test suites.

// partial-json (MIT, https://github.com/promplate/partial-json-parser-js/blob/main/tests/examples.test.js).
const partialJsonVectors = <(String, Object?)>[
  ('"', ''),
  ('["', ['']),
  ('{"": "', {'': ''}),
  ('n', null),
  ('t', true),
  ('f', false),
  ('0', 0),
  ('-1.25e+4', -1.25e4),
  ('-1.25e+', -1.25),
  ('-1.25e', -1.25),

  // Differences pi documents: an invalid escape stays visible, and `Infinity`/`NaN` words are dropped.
  ('" \\x12', ' \\x12'),
  ('I', none),
  ('-I', none),
  ('N', none),
];

// Vercel AI SDK fixJson (Apache-2.0, https://github.com/vercel/ai/blob/main/packages/ai/src/util/fix-json.test.ts):
// the expected values are JSON.parse() of its repaired texts.
const fixJsonVectors = <(String, Object?)>[
  ('', none),
  ('nul', null),
  ('t', true),
  ('fals', false),
  ('12.2', 12.2),
  ('-12', -12),
  ('-', none),
  ('2.5e', 2.5),
  ('2.5e-', 2.5),
  ('2.5e3', 2.5e3),
  ('-2.5e3', -2.5e3),
  ('2.5E', 2.5),
  ('2.5E-', 2.5),
  ('2.5E3', 2.5e3),
  ('-2.5E3', -2.5e3),
  ('12.34e', 12.34),
  ('5e', 5),
  ('"abc', 'abc'),
  ('"value with \\"quoted\\" text and \\\\ escape', 'value with "quoted" text and \\ escape'),
  ('"value with \\', 'value with '),
  ('"\\u', ''),
  ('"\\u12', ''),
  ('"text \\u00', 'text '),
  ('{"a":"\\u12', {'a': ''}),
  ('"value with unicode <"', 'value with unicode <'),
  ('[', <Object?>[]),
  (
    '[[1], [2',
    [
      [1],
      [2],
    ],
  ),
  (
    '[["1"], ["2',
    [
      ['1'],
      ['2'],
    ],
  ),
  (
    '[[false], [nu',
    [
      [false],
      [null],
    ],
  ),
  (
    '[[[]], [[]',
    [
      [<Object?>[]],
      [<Object?>[]],
    ],
  ),
  (
    '[[{}], [{',
    [
      [<String, Object?>{}],
      [<String, Object?>{}],
    ],
  ),
  ('[1, ', [1]),
  ('[[], 123', [<Object?>[], 123]),
  ('{"key":', <String, Object?>{}),
  (
    '{"a": {"b": 1}, "c": {"d": 2',
    {
      'a': {'b': 1},
      'c': {'d': 2},
    },
  ),
  (
    '{"a": {"b": "1"}, "c": {"d": 2',
    {
      'a': {'b': '1'},
      'c': {'d': 2},
    },
  ),
  (
    '{"a": {"b": false}, "c": {"d": 2',
    {
      'a': {'b': false},
      'c': {'d': 2},
    },
  ),
  (
    '{"a": {"b": []}, "c": {"d": 2',
    {
      'a': {'b': <Object?>[]},
      'c': {'d': 2},
    },
  ),
  (
    '{"a": {"b": {}}, "c": {"d": 2',
    {
      'a': {'b': <String, Object?>{}},
      'c': {'d': 2},
    },
  ),
  ('{"ke', <String, Object?>{}),
  ('{"k1": 1, "k2', {'k1': 1}),
  ('{"k1": 1, "k2":', {'k1': 1}),
  ('{"key": "value"  ', {'key': 'value'}),
  (
    '{"a": {"b": {}',
    {
      'a': {'b': <String, Object?>{}},
    },
  ),
  (
    '[1, [2, 3, [',
    [
      1,
      [2, 3, <Object?>[]],
    ],
  ),
  (
    '[false, [true, [',
    [
      false,
      [true, <Object?>[]],
    ],
  ),
  ('{"key": {"subKey":', {'key': <String, Object?>{}}),
  ('{"key": 123, "key2": {"subKey":', {'key': 123, 'key2': <String, Object?>{}}),
  ('{"key": null, "key2": {"subKey":', {'key': null, 'key2': <String, Object?>{}}),
  (
    '{"key": [1, 2, {',
    {
      'key': [1, 2, <String, Object?>{}],
    },
  ),
  (
    '[1, 2, {"key": "value",',
    [
      1,
      2,
      {'key': 'value'},
    ],
  ),
  (
    '{"a": {"b": ["c", {"d": "e",',
    {
      'a': {
        'b': [
          'c',
          {'d': 'e'},
        ],
      },
    },
  ),
  (
    '{"a": {"b": {"c": {"d":',
    {
      'a': {
        'b': {'c': <String, Object?>{}},
      },
    },
  ),
  ('{"a": 1, "b": [', {'a': 1, 'b': <Object?>[]}),
  ('{"a": 1, "b": {', {'a': 1, 'b': <String, Object?>{}}),
  ('{"a": 1, "b": "', {'a': 1, 'b': ''}),
  (
    '{\n  "a": [\n    {\n      "a1": "v1",\n      "a2": "v2",\n      "a3": "v3"\n    }\n  ],\n  "b": [\n    {\n      '
        '"b1": "n',
    {
      'a': [
        {'a1': 'v1', 'a2': 'v2', 'a3': 'v3'},
      ],
      'b': [
        {'b1': 'n'},
      ],
    },
  ),
  (
    '{"type":"div","children":[{"type":"Card","props":{}',
    {
      'type': 'div',
      'children': [
        {'type': 'Card', 'props': <String, Object?>{}},
      ],
    },
  ),

  // A difference pi documents: fixJson completes `12.` to 12; a number cut after its dot is omitted.
  ('12.', none),
  ('12.e', none),
];

const malformed = [
  '{"a":1 "b":2}',
  '{"a" 1, "b": 2}',
  "{a:1, 'b': 2}",
  '{"a":1,,"b":2,}',
  '[1 2, , 3,]',
  '{"a":[1,2}, "b":3]',
  ']]]}}}{',
  '{"a":NaN,"b":Infinity,"c":-Infinity,"d":undefined,"e":1}',
  '{"a":01,"b":1.,"c":.5,"d":+1,"e":0x10,"f":1e}',
  '{"a":"x"} trailing garbage {"b":2}',
  '{"a": tru, "b": nulll, "c": falsey}',
  '{"a":{"b":{"c":',
  '{"a\\": 1}',
  '"unterminated \\',
  '// comment\n{"a":1}',
  '\u0000{\u0001"a"\u0002:\u00031}',
  '{"a":"x"}}}]]]',
];

/// How long [run] takes, in microseconds, after one warm-up run.
int timed(void Function() run) {
  run();
  final clock = Stopwatch()..start();
  run();
  return clock.elapsedMicroseconds;
}

void main() {
  group('differential fuzzing', () {
    test('matches the reference after every piece, at every chunk size, with insignificant whitespace', () {
      for (final text in randomDocuments(1, 250, true)) {
        for (final size in chunkSizes) {
          for (final step in stream(text, sizeFor(size, text))) {
            expect(step.snapshot, referenceSnapshot(step.read), reason: '${jsonEncode(step.read)} (chunk size $size)');
          }
        }
      }
    });

    test('reads complete documents exactly like jsonDecode', () {
      for (final text in [...randomDocuments(3, 300, true), ...randomDocuments(4, 300, false)]) {
        final expected = jsonDecode(text);
        for (final size in chunkSizes) {
          expect(finalSnapshot(text, size == 'random' ? 5 : size as int), expected, reason: text);
        }
      }
    });

    test('never changes a snapshot once it is returned', () {
      for (final text in randomDocuments(5, 150, true)) {
        for (final step in stream(text, random(seedOf(text)))) {
          expect(step.snapshot, step.copy);
        }
      }
    });

    test('repairs raw control characters and invalid escapes, and shows them while streaming', () {
      final rand = random(6);
      const pieces = ['\n', '\t', '\u0000', r'\d', r'\x41', r'\u12', r'\uZZZZ', r'é', r'\\', r'\q', 'a', ' ', 'é'];
      for (var round = 0; round < 300; round++) {
        final content = [for (var i = 0; i < 6; i++) pick(rand, pieces)].join();
        final text = '{"path":"p","content":"$content","n":$round}';

        // Complete text: the value repairJson gives.
        expect(finalSnapshot(text, 1), decodeWithRepair(text), reason: jsonEncode(text));
        for (final step in stream(text, 1)) {
          expect(step.snapshot, referenceSnapshot(step.read), reason: jsonEncode(step.read));
        }
      }
    });
  });

  group('strings', () {
    test('holds an escape back until it is complete, at every split position', () {
      for (final (text, value) in [
        (r'{"a":"xéy"}', 'xéy'),
        (r'{"a":"x\ny"}', 'x\ny'),
        (r'{"a":"x\\y"}', r'x\y'),
        (r'{"a":"x\"y"}', 'x"y'),
        (r'{"a":"x\/y"}', 'x/y'),
      ]) {
        for (var split = 0; split <= text.length; split++) {
          final parser = StreamingJsonParser()..append(text.substring(0, split));
          expect(
            shown(parser.snapshot()),
            referenceSnapshot(text.substring(0, split)),
            reason: text.substring(0, split),
          );
          parser.append(text.substring(split));
          expect(shown(parser.snapshot()), {'a': value});
        }
      }
    });

    test('joins surrogate pairs split across pieces, escaped or raw', () {
      for (final text in [r'{"a":"😀!"}', '{"a":"😀!"}']) {
        for (var split = 0; split <= text.length; split++) {
          final parser = StreamingJsonParser()..append(text.substring(0, split));
          parser
            ..snapshot()
            ..append(text.substring(split));
          expect(shown(parser.snapshot()), {'a': '😀!'});
        }
      }

      // A high surrogate whose low half has not arrived shows alone.
      expect(shown((StreamingJsonParser()..append(r'{"a":"\ud83d')).snapshot()), {'a': '\ud83d'});
    });

    test('keeps trailing whitespace of an unterminated string', () {
      expect(shown((StreamingJsonParser()..append('{"code":"x\t\t  ')).snapshot()), {'code': 'x\t\t  '});
    });

    test('keeps raw control characters and invalid escapes visible while streaming', () {
      expect(shown((StreamingJsonParser()..append('{"a":"x\ny \\d \\u12"')).snapshot()), {'a': 'x\ny \\d \\u12'});
    });

    // pi #10868
    test('keeps invalid \\u escapes literal without dropping fields', () {
      expect(finalSnapshot(r'{"path":"a.tex","content":"\usepackage{x}","mode":"w"}', 1), {
        'path': 'a.tex',
        'content': r'\usepackage{x}',
        'mode': 'w',
      });
      expect(finalSnapshot(r'{"text":"café \url"}', 3), {'text': r'café \url'});
      expect(shown((StreamingJsonParser()..append(r'{"text":"ab\u00')).snapshot()), {'text': 'ab'});
    });

    test('keeps `__proto__` an ordinary member', () {
      const text = '{"__proto__":{"polluted":true},"b":{"__proto__":"x"}}';
      expect(finalSnapshot(text, 1), jsonDecode(text));
    });

    test('keeps the last value of a duplicate key in the first key\'s position, as jsonDecode does', () {
      const text = '{"a":1,"b":2,"a":"three"}';
      expect((finalSnapshot(text, 1) as Map).keys, (jsonDecode(text) as Map).keys);
      expect(finalSnapshot(text, 1), jsonDecode(text));
    });
  });

  group('ported test vectors', () {
    for (final (input, expected) in [...partialJsonVectors, ...fixJsonVectors]) {
      test('reads ${jsonEncode(input)}', () {
        for (final size in [input.isEmpty ? 1 : input.length, 1, 2]) {
          expect(finalSnapshot(input, size), expected, reason: 'chunk size $size');
        }
      });
    }

    test('reads every JSONTestSuite document jsonDecode accepts exactly like it, and never throws on the rest', () {
      final suite = jsonDecode(File('test/fixtures/json-test-suite.json').readAsStringSync()) as Json;
      for (final MapEntry(:key, :value) in (suite['files'] as Json).entries) {
        final text = utf8.decode(base64.decode(value as String), allowMalformed: true);
        Object? expected = none;
        try {
          expected = jsonDecode(text);
        } on FormatException {
          expected = none;
        }

        for (final size in [text.isEmpty ? 1 : text.length, 1, 3]) {
          final snapshot = finalSnapshot(text, size);
          if (expected != none) {
            // Some documents nest hundreds of levels; `equals` compares 100 by default.
            expect(snapshot, equals(expected, 1000), reason: key);
          } else {
            expectJson(snapshot, key);
          }
        }
      }
    });
  });

  group('malformed text', () {
    test('never throws, keeps snapshots plain JSON and unchanged, at every chunk size', () {
      for (final text in malformed) {
        for (final size in chunkSizes) {
          final steps = stream(text, sizeFor(size, text));
          for (final step in steps) {
            expectJson(step.snapshot, text);
            expect(step.snapshot, step.copy);
          }

          // Every chunk size reads malformed text the same way.
          expect(steps.last.snapshot, finalSnapshot(text), reason: text);
        }
      }
    });

    test('shows an unterminated number only up to 64 characters, and once it ends', () {
      final digits = '1' * 70;
      final parser = StreamingJsonParser()..append('[1,$digits');
      expect(shown(parser.snapshot()), [1]);
      parser.append(']');
      expect(shown(parser.snapshot()), [1, num.parse(digits)]);
      expect(shown((StreamingJsonParser()..append('[${'1' * 64}')).snapshot()), [num.parse('1' * 64)]);
    });

    test('tolerates missing commas and colons and drops values in invalid positions', () {
      expect(finalSnapshot('{"a":1 "b":2}'), {'a': 1, 'b': 2});
      expect(finalSnapshot('{"a" 1, "b": 2}'), {'a': 1, 'b': 2});
      expect(finalSnapshot('[1 2, , 3,]'), [1, 2, 3]);
      expect(finalSnapshot('{"a":NaN,"b":Infinity,"e":1}'), {'e': 1});
      expect(finalSnapshot('{"a":"x"} trailing garbage {"b":2}'), {'a': 'x'});
    });

    test('survives random corruption of valid documents', () {
      final rand = random(7);
      const noise = ['"', r'\', '{', '}', '[', ']', ',', ':', 'x', '1', '-', '\n', 't'];
      for (final text in randomDocuments(8, 300, true)) {
        final characters = text.runes.map(String.fromCharCode).toList();
        for (var edit = 0; edit < 3; edit++) {
          final at = (rand() * (characters.length + 1)).floor();
          if (rand() < 0.5) {
            if (at < characters.length) {
              characters.removeAt(at);
            }
          } else {
            characters.insert(at, pick(rand, noise));
          }
        }

        final corrupted = characters.join();
        for (final step in stream(corrupted, random(seedOf(corrupted)))) {
          expectJson(step.snapshot, corrupted);
        }

        // The changes keep a copy equal to the parser's own value, malformed or not.
        final parser = StreamingJsonParser();
        final mirror = Mirror();
        for (var i = 0; i < corrupted.length; i += 4) {
          mirror.apply(parser.append(corrupted.substring(i, i + 4 < corrupted.length ? i + 4 : corrupted.length)));
        }

        // A space ends a number or literal at the end, which snapshots show and changes do not report yet.
        mirror.apply(parser.append(' '));
        expect(mirror.value, shown(parser.snapshot()), reason: corrupted);
      }
    });
  });

  group('changes', () {
    test('keep a copy equal to the value read so far, at every chunk size, and to jsonDecode at the end', () {
      for (final text in randomDocuments(9, 200, true)) {
        for (final size in chunkSizes) {
          expect(mirrorStream(text, sizeFor(size, text)).value, jsonDecode(text), reason: text);
        }
      }
    });

    test('report a number that ends and a string that starts in one piece, in order', () {
      final parser = StreamingJsonParser();
      expect(changes(parser.append('{"n":1')), [
        {'kind': 'set', 'path': <Object>[], 'value': <String, Object?>{}},
      ]);
      expect(changes(parser.append('2,"s":"ab')), [
        {
          'kind': 'set',
          'path': ['n'],
          'value': 12,
        },
        {
          'kind': 'set',
          'path': ['s'],
          'value': 'ab',
        },
      ]);
      expect(changes(parser.append('c","t":tru')), [
        {
          'kind': 'append',
          'path': ['s'],
          'text': 'c',
        },
      ]);
      expect(changes(parser.append('e}')), [
        {
          'kind': 'set',
          'path': ['t'],
          'value': true,
        },
      ]);
      expect(parser.append(' {"ignored":1}'), isEmpty);
    });

    test('report nested objects and arrays as empty containers, then their members and items', () {
      final parser = StreamingJsonParser();
      expect(changes(parser.append('{"a":[1,{"b":["x"')), [
        {'kind': 'set', 'path': <Object>[], 'value': <String, Object?>{}},
        {
          'kind': 'set',
          'path': ['a'],
          'value': <Object?>[],
        },
        {
          'kind': 'set',
          'path': ['a', 0],
          'value': 1,
        },
        {
          'kind': 'set',
          'path': ['a', 1],
          'value': <String, Object?>{},
        },
        {
          'kind': 'set',
          'path': ['a', 1, 'b'],
          'value': <Object?>[],
        },
        {
          'kind': 'set',
          'path': ['a', 1, 'b', 0],
          'value': 'x',
        },
      ]);
      expect(changes(parser.append(',"y')), [
        {
          'kind': 'set',
          'path': ['a', 1, 'b', 1],
          'value': 'y',
        },
      ]);
      expect(changes(parser.append('z"]}],"a":"again"}')), [
        {
          'kind': 'append',
          'path': ['a', 1, 'b', 1],
          'text': 'z',
        },
        {
          'kind': 'set',
          'path': ['a'],
          'value': 'again',
        },
      ]);
    });

    test('hold split escapes back and report their decoded text once complete', () {
      const text = r'{"a":"xé\n\\\"😀\q"}';
      final expected = (jsonDecode(r'{"a":"xé\n\\\"😀\\q"}') as Map)['a'];
      for (var first = 1; first < text.length; first++) {
        for (var second = first; second < text.length; second++) {
          final parser = StreamingJsonParser();
          final mirror = Mirror();
          for (final piece in [text.substring(0, first), text.substring(first, second), text.substring(second)]) {
            mirror.apply(parser.append(piece));
          }

          expect(mirror.value, {'a': expected});
        }
      }

      final parser = StreamingJsonParser()..append('{"a":"x');
      expect(parser.append(r'\u00'), isEmpty);
      expect(changes(parser.append('e9')), [
        {
          'kind': 'append',
          'path': ['a'],
          'text': 'é',
        },
      ]);
    });

    test('report a container nested deeper than 32 levels once, whole, when it closes', () {
      const depth = 40;
      final text = '${'{"a":' * depth}"deep"${'}' * depth}';
      final parser = StreamingJsonParser();
      final mirror = Mirror();
      for (var i = 0; i < text.length; i += 3) {
        mirror.apply(parser.append(text.substring(i, i + 3 < text.length ? i + 3 : text.length)));
      }

      expect(mirror.value, jsonDecode(text));
      final all = StreamingJsonParser().append(text);

      // The containers down to path length 32, then the one at length 32 again, whole, as it closes.
      expect(all, hasLength(34));
      final last = all.last as JsonSet;
      expect(last.path, hasLength(32));
      expect(last.value, jsonDecode(text.substring(32 * 5, text.length - 32)));
    });

    test('report nothing for values in invalid positions and text after the root', () {
      final parser = StreamingJsonParser();
      expect(changes(parser.append('{"a":1 2, "b" "x"}')), [
        {'kind': 'set', 'path': <Object>[], 'value': <String, Object?>{}},
        {
          'kind': 'set',
          'path': ['a'],
          'value': 1,
        },
        {
          'kind': 'set',
          'path': ['b'],
          'value': 'x',
        },
      ]);
      expect(parser.append('{"c":3}'), isEmpty);
    });
  });

  group('cost', () {
    final content = [for (var line = 0; line < 20000; line++) 'const line$line = "value $line"; // \\ done\n'].join();

    /// `write`-style arguments with [size] characters of file content.
    String writeArguments(int size) => jsonEncode({'path': '/src/file.ts', 'content': content.substring(0, size)});

    void streamArguments(String text, int delta) {
      final args = ToolArgumentsStream();
      for (var i = 0; i < text.length; i += delta) {
        args.append(text.substring(i, i + delta < text.length ? i + delta : text.length));
      }

      args.finish();
    }

    test('streams a long open string in time linear in its length', () {
      final small = writeArguments(50000);
      final large = writeArguments(500000);
      final smallTime = timed(() => streamArguments(small, 32));
      final largeTime = timed(() => streamArguments(large, 32));

      // Ten times the text: linear growth is about 10x, quadratic about 100x.
      expect(largeTime / smallTime, lessThan(30), reason: '$smallTime us, $largeTime us');
    });

    test('lets a long open string\'s preview lag but only move forward, and catch up when it ends', () {
      final text = writeArguments(200000);
      final parser = StreamingJsonParser();
      var shown = 0;
      for (var i = 0; i < text.length; i += 32) {
        parser.append(text.substring(i, i + 32 < text.length ? i + 32 : text.length));
        final value = (parser.snapshot()?.value as Map?)?['content'] as String? ?? '';
        expect(value.length, greaterThanOrEqualTo(shown));
        shown = value.length;
      }

      expect(parser.snapshot()?.value, jsonDecode(text));
    });

    test('keeps very large open arrays linear by letting their preview lag', () {
      String elements(int count) => jsonEncode({'items': List.generate(count, (i) => i)});
      final small = elements(10000);
      final large = elements(100000);
      final smallTime = timed(() => streamArguments(small, 16));
      final largeTime = timed(() => streamArguments(large, 16));
      expect(largeTime / smallTime, lessThan(30), reason: '$smallTime us, $largeTime us');

      final parser = StreamingJsonParser();
      var shown = 0;
      for (var i = 0; i < large.length; i += 16) {
        parser.append(large.substring(i, i + 16 < large.length ? i + 16 : large.length));
        final items = (parser.snapshot()?.value as Map?)?['items'] as List? ?? const [];

        // The preview only moves forward.
        expect(items.length, greaterThanOrEqualTo(shown));
        shown = items.length;
      }

      expect(parser.snapshot()?.value, jsonDecode(large));
    });

    test('reads deep nesting without recursion and in linear time', () {
      const depth = 200000;
      final text = '${'[' * depth}1${']' * depth}';
      final clock = Stopwatch()..start();
      final parser = StreamingJsonParser();
      for (var i = 0; i < text.length; i += 7) {
        parser
          ..append(text.substring(i, i + 7 < text.length ? i + 7 : text.length))
          ..snapshot();
      }

      Object? value = parser.snapshot()?.value;
      for (var level = 0; level < depth; level++) {
        value = (value as List)[0];
      }

      expect(value, 1);
      expect(clock.elapsedMilliseconds, lessThan(5000));
    });

    test('reads one huge piece, such as a provider\'s complete arguments, quickly', () {
      final text = writeArguments(500000);
      final clock = Stopwatch()..start();
      final parser = StreamingJsonParser()..append(text);
      expect(parser.snapshot()?.value, jsonDecode(text));
      expect(clock.elapsedMilliseconds, lessThan(2000));
    });
  });

  group('ToolArgumentsStream', () {
    test('shows partial arguments as an object and finishes with one full parse', () {
      final args = ToolArgumentsStream();
      expect(args.append('  '), isEmpty);
      expect(args.append('{"path":"a.ts","content":"x'), {'path': 'a.ts', 'content': 'x'});
      final before = args.append('y');
      expect(args.append(''), same(before));
      expect(args.text, '  {"path":"a.ts","content":"xy');
      expect(args.finish().error, parseToolArguments('  {"path":"a.ts","content":"xy').error);
      args.append('"}');
      final done = args.finish();
      expect(done.arguments, {'path': 'a.ts', 'content': 'xy'});
      expect(done.error, isNull);
    });

    test('shows non-object roots as no arguments yet', () {
      final args = ToolArgumentsStream();
      expect(args.append('[1,2'), isEmpty);
      args.append(']');
      expect(args.finish().error, startsWith('Tool arguments must be a JSON object'));
    });

    test('replaces the text with the provider\'s complete text, returning the missing piece', () {
      final args = ToolArgumentsStream()..append('{"a":"x');
      final continued = args.replace('{"a":"xyz"}');
      expect(continued.delta, 'yz"}');
      expect(continued.arguments, {'a': 'xyz'});
      final restarted = args.replace('{"b":2}');
      expect(restarted.delta, isNull);
      expect(restarted.arguments, {'b': 2});
      expect(args.text, '{"b":2}');
    });
  });
}
