import 'dart:convert';

import 'streaming_json.dart';
import 'types.dart';

const _escapes = {'"', r'\', '/', 'b', 'f', 'n', 'r', 't'};
final _hex4 = RegExp(r'^[0-9a-fA-F]{4}$');

/// Repairs malformed JSON string literals: escapes raw control characters and doubles a backslash before an invalid
/// escape character.
String repairJson(String json) {
  final out = StringBuffer();
  var inside = false;
  for (var i = 0; i < json.length; i++) {
    final char = json[i];
    if (!inside) {
      out.write(char);
      inside = char == '"';
      continue;
    }

    if (char == '"') {
      out.write(char);
      inside = false;
      continue;
    }

    if (char == r'\') {
      if (i + 1 == json.length) {
        out.write(r'\\');
        continue;
      }

      final next = json[i + 1];
      if (next == 'u' && i + 6 <= json.length && _hex4.hasMatch(json.substring(i + 2, i + 6))) {
        out.write(json.substring(i, i + 6));
        i += 5;
        continue;
      }

      if (_escapes.contains(next)) {
        out.write('\\$next');
        i++;
        continue;
      }

      out.write(r'\\');
      continue;
    }

    out.write(char.codeUnitAt(0) <= 0x1f ? _escapeControl(char) : char);
  }

  return out.toString();
}

String _escapeControl(String char) {
  return switch (char) {
    '\b' => r'\b',
    '\f' => r'\f',
    '\n' => r'\n',
    '\r' => r'\r',
    '\t' => r'\t',
    _ => '\\u${char.codeUnitAt(0).toRadixString(16).padLeft(4, '0')}',
  };
}

/// Decodes [json], repairing it once when it does not decode as it is.
Object? decodeWithRepair(String json) {
  try {
    return jsonDecode(json);
  } on FormatException {
    final repaired = repairJson(json);
    if (repaired == json) {
      rethrow;
    }

    return jsonDecode(repaired);
  }
}

/// The arguments of a finished tool call from their complete text. Text that is not a JSON object gives empty
/// arguments and an error naming the problem and quoting the text, so tool validation can report it to the model.
({Json arguments, String? error}) parseToolArguments(String text) {
  if (text.trim().isEmpty) {
    return (arguments: {}, error: null);
  }

  try {
    final value = decodeWithRepair(text);
    if (value is! Map) {
      return (arguments: {}, error: 'Tool arguments must be a JSON object\n\nReceived arguments:\n$text');
    }

    return (arguments: value.cast<String, Object?>(), error: null);
  } on FormatException catch (e) {
    return (arguments: {}, error: '${e.message}\n\nReceived arguments:\n$text');
  }
}

/// The arguments of one streaming tool call: the raw text, for the one full parse when the call ends, and a
/// [StreamingJsonParser] for the partial arguments in between. Providers keep one per call in their own stream state,
/// never on the content block, so the raw text never reaches partial messages or transcripts.
class ToolArgumentsStream {
  final _text = StringBuffer();
  var _parser = StreamingJsonParser();
  ({Object? value})? _snapshot;
  Json _arguments = {};

  /// The raw argument text received so far.
  String get text => _text.toString();

  /// Appends a piece of argument text; returns the partial arguments, a new map whenever they changed.
  Json append(String delta) {
    _text.write(delta);
    _parser.append(delta);
    return _partial();
  }

  /// Takes the provider's complete argument text. When it continues the text so far, returns the missing piece as
  /// `delta`; otherwise starts over with [complete]. Either way returns the partial arguments of [complete].
  ({String? delta, Json arguments}) replace(String complete) {
    final current = _text.toString();
    if (complete.startsWith(current)) {
      final delta = complete.substring(current.length);
      return (delta: delta, arguments: delta.isEmpty ? _partial() : append(delta));
    }

    _text.clear();
    _parser = StreamingJsonParser();
    return (delta: null, arguments: append(complete));
  }

  /// The final arguments: one full parse of the complete text, which reports malformed arguments.
  ({Json arguments, String? error}) finish() => parseToolArguments(_text.toString());

  Json _partial() {
    final snapshot = _parser.snapshot();
    if (identical(snapshot?.value, _snapshot?.value) && (snapshot == null) == (_snapshot == null)) {
      return _arguments;
    }

    _snapshot = snapshot;

    // Tool arguments are an object; any other partial value shows as no arguments yet.
    final value = snapshot?.value;
    _arguments = value is Map<String, Object?> ? value : {};
    return _arguments;
  }
}
