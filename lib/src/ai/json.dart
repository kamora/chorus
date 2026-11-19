import 'dart:convert';

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

/// The value of possibly incomplete JSON text, for previews while it streams; never throws.
///
/// Open strings and containers are closed. A member or element that cannot be completed is dropped, and so is an
/// unfinished literal (pi shows `true` for `tr`). Each call reads the whole text, so a stream costs time quadratic
/// in its length; pi's incremental parser is linear.
Object? parsePartialJson(String text) {
  if (text.trim().isEmpty) {
    return null;
  }

  try {
    return decodeWithRepair(text);
  } on FormatException {
    // Retry at each earlier point where a member or element begins, newest first.
    final cuts = <int>[text.length];
    var inside = false;
    var escaped = false;
    for (var i = 0; i < text.length; i++) {
      final char = text[i];
      if (inside) {
        if (escaped) {
          escaped = false;
        } else if (char == r'\') {
          escaped = true;
        } else if (char == '"') {
          inside = false;
        }

        continue;
      }

      if (char == '"') {
        inside = true;
      } else if (char == ',') {
        cuts.add(i);
      } else if (char == '{' || char == '[') {
        cuts.add(i + 1);
      }
    }

    for (final cut in cuts..sort((a, b) => b - a)) {
      try {
        return decodeWithRepair(_close(text.substring(0, cut)));
      } on FormatException {
        continue;
      }
    }

    return null;
  }
}

/// [text] with its open string and containers closed.
String _close(String text) {
  final stack = <String>[];
  var inside = false;

  // Where the escape being read starts, and how many characters it still needs.
  var escape = -1;
  var needs = 0;
  for (var i = 0; i < text.length; i++) {
    final char = text[i];
    if (inside) {
      if (needs > 0) {
        needs--;
        if (char == 'u' && i == escape + 1) {
          needs = 4;
        }
      } else if (char == r'\') {
        escape = i;
        needs = 1;
      } else if (char == '"') {
        inside = false;
      }

      continue;
    }

    switch (char) {
      case '"':
        inside = true;
      case '{':
        stack.add('}');
      case '[':
        stack.add(']');
      case '}' || ']':
        if (stack.isNotEmpty) {
          stack.removeLast();
        }
    }
  }

  var out = text;
  if (inside) {
    // An unfinished escape, such as a `\u` with fewer than four digits, is dropped.
    out = '${needs > 0 ? out.substring(0, escape) : out}"';
  }

  out = out.trimRight();
  if (out.endsWith(',')) {
    out = out.substring(0, out.length - 1);
  }

  return out + stack.reversed.join();
}

/// The arguments of one streaming tool call: the raw text for the final parse, and partial arguments in between.
class ToolArgumentsStream {
  final _text = StringBuffer();

  /// Appends a piece of argument text; returns the partial arguments so far.
  Json append(String delta) {
    _text.write(delta);
    final value = parsePartialJson(_text.toString());
    return value is Map ? value.cast<String, Object?>() : {};
  }

  /// The final arguments: one full parse of the complete text.
  ({Json arguments, String? error}) finish() => parseToolArguments(_text.toString());
}
