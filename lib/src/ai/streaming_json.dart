// Incremental, tolerant parsing of JSON text that arrives in pieces, such as streamed tool call arguments. Ported from
// pi's utils/streaming-json.ts.
//
// [StreamingJsonParser.append] reads each piece once, so a stream costs time and memory linear in its length, and
// [StreamingJsonParser.snapshot] returns the value of the text read so far. Snapshots are plain JSON values that never
// change once returned: the containers still open (the "open path") are copied into each new snapshot, and completed
// containers and strings are shared. The rules for incomplete text:
//
// - An unterminated string shows the characters read so far; an unterminated escape is held back until complete.
// - An unterminated container shows the members and elements read so far.
// - A member whose key or value has not started is omitted.
// - An unterminated literal shows its value once it is a prefix of `true`, `false` or `null`; an unterminated number
//   shows its value when the digits read so far form a number (an exponent without digits is ignored).
//
// Malformed text is read leniently, the way repairJson treats it: raw control characters in strings are kept, an
// invalid escape such as `\d` stays literal text, missing commas and colons are tolerated, values in invalid positions
// and unknown words are dropped, and text after the root value is ignored. The parser never throws.
//
// [StreamingJsonParser.append] also returns the changes a piece made, for a writer that keeps its own copy of the
// value: a [JsonSet] when a key, an array item or a finished value appears (a new string is set once with its first
// text, a container empty), a [JsonAppend] when the piece extended an open string. Unterminated numbers and literals
// are not reported until they end. A container nested more than 32 levels deep is reported as one set of its whole
// value when it closes, so paths stay short.
//
// Cost: a fresh snapshot copies the open path (its member and element count) and the open string's text. Each piece
// earns copy credit proportional to its length plus a constant; a snapshot spends it, and returns the previous snapshot
// while a fresh one costs more than the credit. So the preview of a very large open container or a very long open
// string lags a little behind the text, and no input makes the total work grow faster than the text. Counting the open
// string is chorus's addition: a JavaScript string grows and is shared at no cost, a Dart string is copied whole.

import 'types.dart';

/// A path from the root value: member keys ([String]) and array indices ([int]).
typedef JsonPath = List<Object>;

/// One change a piece of text made to the value.
sealed class JsonChange {
  JsonChange(this.path);

  final JsonPath path;

  Json toJson();

  @override
  String toString() => '${toJson()}';
}

/// A value set at [path]: a key, an array item, a finished value, a new string with its first text, or an empty
/// container.
class JsonSet extends JsonChange {
  JsonSet(super.path, Object? value) : _value = value;

  Object? _value;

  // The text of a new string, while the piece being read extends it.
  StringBuffer? _text;

  Object? get value => _value;

  @override
  Json toJson() => {'kind': 'set', 'path': path, 'value': value};
}

/// Text appended to the open string at [path].
class JsonAppend extends JsonChange {
  JsonAppend(super.path, String text) : _text = StringBuffer(text);

  final StringBuffer _text;
  String? _done;

  String get text => _done ?? _text.toString();

  @override
  Json toJson() => {'kind': 'append', 'path': path, 'text': text};
}

/// Where a value sits in its parent; null for a value in an invalid position, which is read but dropped.
sealed class _Slot {
  const _Slot();
}

class _Root extends _Slot {
  const _Root();
}

class _Member extends _Slot {
  const _Member(this.key);

  final String key;
}

class _Element extends _Slot {
  const _Element(this.index);

  final int index;
}

enum _Phase { key, colon, value, comma }

/// An open object or array. Its container is never handed out while open; snapshots get copies.
class _Frame {
  _Frame(this.container, this.slot, this.path, this.reportsInside)
    : phase = container is List ? _Phase.value : _Phase.key;

  final Object container;
  final _Slot? slot;

  /// Where the container sits, when its changes are reported.
  final JsonPath? path;

  /// Whether changes inside it are reported; when not, its whole value is reported as it closes.
  final bool reportsInside;

  /// Members or elements: what a copy of the container costs.
  var size = 0;

  /// Objects: the key read last, where the member value goes.
  String? key;
  _Phase phase;

  bool get isArray => container is List;
}

/// An unterminated string.
class _OpenString {
  _OpenString(this.isKey, this.slot, this.path);

  final bool isKey;
  final _Slot? slot;
  final text = StringBuffer();

  /// The escape read so far: empty for none, otherwise a backslash and up to five more characters.
  var escape = '';

  /// Where a value string sits, when its changes are reported.
  final JsonPath? path;

  /// The change this string extends, while it is the newest change of the piece.
  JsonChange? change;
}

/// Stands in a container for the open string at its slot; snapshots put the string's text there.
class _OpenText {
  const _OpenText(this.open);

  final _OpenString open;
}

// Changes are reported at most this deep, so a path costs O(1) to build.
const _maxReportedDepth = 32;

// Copy credit per piece and per character. The floor keeps small documents fresh at every snapshot.
const _creditPerAppend = 256;
const _creditPerCharacter = 8;

// An unterminated number longer than this is not shown until it ends, so a snapshot stays cheap.
const _maxPendingScalar = 64;
final _number = RegExp(r'^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?$');
final _numberWithoutExponentDigits = RegExp(r'^(-?(?:0|[1-9]\d*)(?:\.\d+)?)[eE][+-]?$');
const _literals = <(String, Object?)>[('true', true), ('false', false), ('null', null)];

const _quote = 0x22;
const _backslash = 0x5c;

const _simpleEscapes = {'"': '"', r'\': r'\', '/': '/', 'b': '\b', 'f': '\f', 'n': '\n', 'r': '\r', 't': '\t'};

/// Incremental, tolerant JSON parser; see the library comment for its rules and costs.
class StreamingJsonParser {
  Object? _root;
  var _hasRoot = false;
  var _done = false;
  final _stack = <_Frame>[];
  _OpenString? _string;
  var _scalar = '';
  _Slot? _scalarSlot;

  /// The sum of `size + 1` over the open path: what a fresh snapshot copies, besides the open string.
  var _pathCost = 0;
  var _credit = 0;
  var _dirty = false;
  Object? _snapshot;
  var _hasSnapshot = false;

  /// The changes of the piece being read.
  var _changes = <JsonChange>[];

  /// Reads the next piece of text; returns the changes it made, in order. Costs time linear in the piece's length.
  List<JsonChange> append(String text) {
    _credit += _creditPerAppend + _creditPerCharacter * text.length;
    _changes = [];
    _string?.change = null;
    var index = 0;
    while (index < text.length && !_done) {
      final open = _string;
      if (open != null) {
        index = _readString(open, text, index);
      } else if (_scalar.isNotEmpty) {
        index = _readScalar(text, index);
      } else {
        _structural(text[index++]);
      }
    }

    // A string set with its first text holds that text in a buffer while the piece extends it.
    for (final change in _changes) {
      switch (change) {
        case JsonSet(_text: final StringBuffer text):
          change
            .._value = text.toString()
            .._text = null;
        case JsonAppend():
          change._done = change._text.toString();
        case JsonSet():
          break;
      }
    }

    return _changes;
  }

  /// The value of the text read so far: null before a value starts, otherwise a record of a plain JSON value that never
  /// changes afterwards. Returns the previous snapshot while the text read since did not change the value, or while a
  /// fresh copy of the open path is not paid for yet (see the library comment).
  ({Object? value})? snapshot() {
    _refresh();
    return _hasSnapshot ? (value: _snapshot) : null;
  }

  void _refresh() {
    if (!_dirty) {
      return;
    }

    final pending = _pendingScalar();
    final cost = _pathCost + (pending == null ? 0 : 1) + _openStringCost();
    if (cost > _credit) {
      return;
    }

    _credit -= cost;
    _dirty = false;
    final copy = _copyOpenPath(pending);
    _snapshot = copy.value;
    _hasSnapshot = copy.has;
  }

  /// What showing the open string costs: its length, when it is a value in a valid position.
  int _openStringCost() {
    final open = _string;
    if (open == null || open.isKey || open.slot == null) {
      return 0;
    }

    return open.text.length;
  }

  /// Reads string characters from [start] until the string ends or the text does; returns the next index.
  int _readString(_OpenString open, String text, int start) {
    var runStart = start;
    for (var index = start; index < text.length; index++) {
      final code = text.codeUnitAt(index);
      if (open.escape.isNotEmpty) {
        if (_continueEscape(open, text[index])) {
          runStart = index + 1;
          continue;
        }

        // Not part of the escape: read it as a normal string character below.
        runStart = index;
      }

      if (code == _quote) {
        if (index > runStart) {
          _appendString(open, text.substring(runStart, index));
        }

        _endString(open);
        return index + 1;
      }

      if (code == _backslash) {
        if (index > runStart) {
          _appendString(open, text.substring(runStart, index));
        }

        open.escape = r'\';
        runStart = index + 1;
      }
    }

    if (text.length > runStart) {
      _appendString(open, text.substring(runStart));
    }

    return text.length;
  }

  /// Reads number or literal characters from [start] up to the next delimiter; returns the delimiter's index.
  int _readScalar(String text, int start) {
    var end = start;
    while (end < text.length && !_isDelimiter(text[end])) {
      end++;
    }

    if (end > start) {
      _scalar += text.substring(start, end);
      if (_scalarSlot != null) {
        _dirty = true;
      }
    }

    if (end < text.length) {
      _endScalar();
    }

    return end;
  }

  /// Handles one character outside strings.
  void _structural(String char) {
    switch (char) {
      case ' ' || '\t' || '\n' || '\r':
        return;
      case '"':
        _startString();
      case '{':
        _openContainer(<String, Object?>{});
      case '[':
        _openContainer(<Object?>[]);
      case '}' || ']':
        _closeContainer();
      case ':':
        final frame = _stack.lastOrNull;
        if (frame != null && !frame.isArray && frame.phase == _Phase.colon) {
          frame.phase = _Phase.value;
        }
      case ',':
        final frame = _stack.lastOrNull;
        if (frame != null) {
          frame.phase = frame.isArray ? _Phase.value : _Phase.key;
        }
      default:
        _scalar = char;
        _scalarSlot = _valueSlot();
        if (_scalarSlot != null) {
          _dirty = true;
        }
    }
  }

  /// Where a value starting now goes, or null when this position takes no value.
  _Slot? _valueSlot() {
    final frame = _stack.lastOrNull;
    if (frame == null) {
      return _hasRoot ? null : const _Root();
    }

    if (frame.isArray) {
      return _Element((frame.container as List<Object?>).length);
    }

    // A missing colon is tolerated: `{"a" 1}` reads as `{"a": 1}`.
    final key = frame.key;
    if ((frame.phase == _Phase.value || frame.phase == _Phase.colon) && key != null) {
      return _Member(key);
    }

    return null;
  }

  void _startString() {
    // A missing comma is tolerated: `{"a": 1 "b": 2}` reads the second member.
    final frame = _stack.lastOrNull;
    if (frame != null && !frame.isArray && (frame.phase == _Phase.key || frame.phase == _Phase.comma)) {
      _string = _OpenString(true, null, null);
      return;
    }

    final slot = _valueSlot();
    final path = _reportedPath(slot);
    final open = _string = _OpenString(false, slot, path);
    _place(slot, _OpenText(open));
    if (path != null) {
      open.change = _report(JsonSet(path, '').._text = StringBuffer());
    }
  }

  /// The path of a value at [slot] in the innermost open container, when its changes are reported.
  JsonPath? _reportedPath(_Slot? slot) {
    switch (slot) {
      case null:
        return null;
      case _Root():
        return [];
      case _Member(:final key):
        final frame = _stack.last;
        return frame.reportsInside ? [...frame.path!, key] : null;
      case _Element(:final index):
        final frame = _stack.last;
        return frame.reportsInside ? [...frame.path!, index] : null;
    }
  }

  JsonChange _report(JsonChange change) {
    _changes.add(change);
    return change;
  }

  /// Continues an escape with [char]; returns false when [char] is not part of it and must be read normally.
  bool _continueEscape(_OpenString open, String char) {
    if (open.escape == r'\') {
      final decoded = _simpleEscapes[char];
      if (decoded != null) {
        open.escape = '';
        _appendString(open, decoded);
        return true;
      }

      if (char == 'u') {
        open.escape = r'\u';
        return true;
      }

      // An invalid escape: the backslash stays text and the character is read normally, as repairJson does.
      open.escape = '';
      _appendString(open, r'\');
      return false;
    }

    if (_isHexDigit(char)) {
      open.escape += char;
      if (open.escape.length < 6) {
        return true;
      }

      final decoded = String.fromCharCode(int.parse(open.escape.substring(2), radix: 16));
      open.escape = '';
      _appendString(open, decoded);
      return true;
    }

    // `\u` without four hex digits: it stays text and the character is read normally, as repairJson does.
    final literal = open.escape;
    open.escape = '';
    _appendString(open, literal);
    return false;
  }

  void _appendString(_OpenString open, String text) {
    open.text.write(text);
    if (open.isKey) {
      return;
    }

    if (open.slot != null) {
      _dirty = true;
    }

    final path = open.path;
    if (path == null) {
      return;
    }

    // Extend this string's newest change while nothing came after it; otherwise report an append.
    final change = open.change;
    if (change != null && identical(_changes.lastOrNull, change)) {
      switch (change) {
        case JsonSet():
          change._text!.write(text);
        case JsonAppend():
          change._text.write(text);
      }

      return;
    }

    open.change = _report(JsonAppend(path, text));
  }

  void _endString(_OpenString open) {
    _string = null;
    final frame = _stack.lastOrNull;
    if (open.isKey) {
      if (frame == null) {
        return;
      }

      frame
        ..key = open.text.toString()
        ..phase = _Phase.colon;
      return;
    }

    _place(open.slot, open.text.toString());
    _valueEnded(frame);
  }

  void _endScalar() {
    final token = _scalar;
    final slot = _scalarSlot;
    _scalar = '';
    _scalarSlot = null;
    final value = _scalarValue(token);
    if (value != _invalid) {
      _place(slot, value);
      final path = _reportedPath(slot);
      if (path != null) {
        _report(JsonSet(path, value));
      }
    }

    _valueEnded(_stack.lastOrNull);
  }

  /// The value of the unterminated scalar at the end of the text, if it shows.
  ({_Slot slot, Object? value})? _pendingScalar() {
    final slot = _scalarSlot;
    final token = _scalar;
    if (slot == null || token.isEmpty || token.length > _maxPendingScalar) {
      return null;
    }

    final value = _pendingScalarValue(token);
    return value == _invalid ? null : (slot: slot, value: value);
  }

  void _openContainer(Object container) {
    final slot = _valueSlot();
    final path = _reportedPath(slot);
    _place(slot, container);
    _stack.add(_Frame(container, slot, path, path != null && path.length < _maxReportedDepth));
    _pathCost += 1;
    _dirty = true;

    // A fresh container: the working one keeps changing, and a change never does.
    if (path != null) {
      _report(JsonSet(path, container is List ? <Object?>[] : <String, Object?>{}));
    }
  }

  void _closeContainer() {
    // A stray closer outside any container is ignored.
    if (_stack.isEmpty) {
      return;
    }

    final frame = _stack.removeLast();
    _pathCost -= frame.size + 1;
    _dirty = true;

    // Too deep to report inside: the whole value, final now that it is closed.
    if (frame.path != null && !frame.reportsInside) {
      _report(JsonSet(frame.path!, frame.container));
    }

    _valueEnded(_stack.lastOrNull);
  }

  /// A value in [frame], or the root value when there is no frame, has ended.
  void _valueEnded(_Frame? frame) {
    if (frame == null) {
      // Text after the root value is ignored, but only once a root value exists.
      if (_hasRoot) {
        _done = true;
      }

      return;
    }

    frame.phase = _Phase.comma;
  }

  /// Sets the value at [slot] in the open path; nothing for a value in an invalid position.
  void _place(_Slot? slot, Object? value) {
    if (slot == null) {
      return;
    }

    _dirty = true;
    switch (slot) {
      case _Root():
        _root = value;
        _hasRoot = true;
      case _Element(:final index):
        final frame = _stack.last;
        final elements = frame.container as List<Object?>;
        if (index == elements.length) {
          frame.size++;
          _pathCost++;
          elements.add(value);
        } else {
          elements[index] = value;
        }
      case _Member(:final key):
        final frame = _stack.last;
        final members = frame.container as Map<String, Object?>;
        if (!members.containsKey(key)) {
          frame.size++;
          _pathCost++;
        }

        members[key] = value;
    }
  }

  /// A fresh snapshot: copies of the open containers, innermost first, each holding the copy of its open child, with
  /// the open string's text and the pending scalar in place.
  ({Object? value, bool has}) _copyOpenPath(({_Slot slot, Object? value})? pending) {
    if (_stack.isEmpty) {
      if (pending != null && pending.slot is _Root) {
        return (value: pending.value, has: true);
      }

      return (value: _shown(_root), has: _hasRoot);
    }

    Object? child;
    _Slot? childSlot;
    for (var depth = _stack.length - 1; depth >= 0; depth--) {
      final frame = _stack[depth];
      final container = frame.container;
      final Object copy = container is List<Object?>
          ? List<Object?>.of(container)
          : Map<String, Object?>.of(container as Map<String, Object?>);
      if (depth == _stack.length - 1) {
        final open = _string;
        if (open != null && !open.isKey && open.slot != null) {
          _setSlot(copy, open.slot!, open.text.toString());
        }

        if (pending != null) {
          _setSlot(copy, pending.slot, pending.value);
        }
      }

      if (childSlot != null) {
        _setSlot(copy, childSlot, child);
      }

      child = copy;
      childSlot = frame.slot;
    }

    // The outermost open container is the root unless it sits in an invalid position.
    if (childSlot is _Root) {
      return (value: child, has: true);
    }

    return (value: _shown(_root), has: _hasRoot);
  }

  // A root that is still an open string shows its text.
  Object? _shown(Object? value) => value is _OpenText ? value.open.text.toString() : value;
}

// Marks a token that is not a value.
const _invalid = Object();

bool _isDelimiter(String char) {
  return switch (char) {
    ' ' || '\t' || '\n' || '\r' || ',' || ':' || '{' || '}' || '[' || ']' || '"' => true,
    _ => false,
  };
}

bool _isHexDigit(String char) {
  final c = char.codeUnitAt(0);
  return (c >= 0x30 && c <= 0x39) || (c >= 0x61 && c <= 0x66) || (c >= 0x41 && c <= 0x46);
}

/// The value of a complete number or literal token; [_invalid] for anything else, which is dropped.
Object? _scalarValue(String token) {
  for (final (literal, value) in _literals) {
    if (token == literal) {
      return value;
    }
  }

  return _number.hasMatch(token) ? num.parse(token) : _invalid;
}

/// The value an unterminated number or literal token shows; [_invalid] for none.
Object? _pendingScalarValue(String token) {
  for (final (literal, value) in _literals) {
    if (literal.startsWith(token)) {
      return value;
    }
  }

  if (_number.hasMatch(token)) {
    return num.parse(token);
  }

  final mantissa = _numberWithoutExponentDigits.firstMatch(token);
  return mantissa == null ? _invalid : num.parse(mantissa[1]!);
}

void _setSlot(Object container, _Slot slot, Object? value) {
  switch (slot) {
    case _Element(:final index):
      // A pending scalar is the element after the last one.
      final elements = container as List<Object?>;
      if (index == elements.length) {
        elements.add(value);
      } else {
        elements[index] = value;
      }
    case _Member(:final key):
      (container as Map<String, Object?>)[key] = value;
    case _Root():
      break;
  }
}
