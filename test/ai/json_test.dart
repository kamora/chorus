import 'package:chorus/src/ai/json.dart';
import 'package:chorus/src/ai/streaming_json.dart';
import 'package:test/test.dart';

/// The streaming parser's value of [text], read in one piece.
Object? parsePartialJson(String text) => (StreamingJsonParser()..append(text)).snapshot()?.value;

// Ported from pi's test/json-parse.test.ts; the partial cases at the end are chorus's own.
void main() {
  group('partial values', () {
    test('keeps invalid \\u escapes as literal text without dropping fields', () {
      expect(parsePartialJson(r'{"text":"\url{x}"}'), {'text': r'\url{x}'});
      expect(parsePartialJson(r'{"path":"a.tex","content":"\usepackage{x}","mode":"w"}'), {
        'path': 'a.tex',
        'content': r'\usepackage{x}',
        'mode': 'w',
      });
    });

    test('decodes valid \\u escapes', () {
      expect(parsePartialJson(r'{"text":"café \url"}'), {'text': r'café \url'});
    });

    test('omits a \\u escape that is cut off mid-stream', () {
      expect(parsePartialJson(r'{"text":"ab\u00'), {'text': 'ab'});
    });

    test('shows open strings, containers and finished members', () {
      expect(parsePartialJson('{"path": "/tmp/a'), {'path': '/tmp/a'});
      expect(parsePartialJson('{"a": 1, "b": [1, 2'), {
        'a': 1,
        'b': [1, 2],
      });
      expect(parsePartialJson('{"a": 1, "b": '), {'a': 1});
      expect(parsePartialJson('{"a": 1, "b'), {'a': 1});
      expect(parsePartialJson(r'{"a": "x\\'), {'a': r'x\'});
      expect(parsePartialJson(r'{"a": "x\'), {'a': 'x'});
      expect(parsePartialJson('{"a": {"b": {"c": "d'), {
        'a': {
          'b': {'c': 'd'},
        },
      });
      expect(parsePartialJson(''), isNull);
      expect(parsePartialJson('{'), isEmpty);
    });
  });

  group('parseToolArguments', () {
    test('parses complete JSON and repairs invalid escapes', () {
      final result = parseToolArguments(r'{"kind":"seek","note":"\url"}');
      expect(result.arguments, {'kind': 'seek', 'note': r'\url'});
      expect(result.error, isNull);
    });

    test('uses empty arguments for empty text', () {
      expect(parseToolArguments('').arguments, isEmpty);
      expect(parseToolArguments('').error, isNull);
    });

    test('does not keep the valid prefix of malformed JSON', () {
      const raw = '{"action":"apply","actions":[{"kind":"seek","time":.5}]}';
      final result = parseToolArguments(raw);
      expect(result.arguments, isEmpty);
      expect(result.error, contains('\n\nReceived arguments:\n'));
      expect(result.error, endsWith(raw));
    });

    test('does not complete cut-off JSON', () {
      const raw = '{"kind":"seek","clipId":"clip-fl';
      final result = parseToolArguments(raw);
      expect(result.arguments, isEmpty);
      expect(result.error, endsWith(raw));
    });

    test('rejects JSON that is not an object', () {
      final result = parseToolArguments('[1,2]');
      expect(result.arguments, isEmpty);
      expect(result.error, 'Tool arguments must be a JSON object\n\nReceived arguments:\n[1,2]');
    });
  });
}
