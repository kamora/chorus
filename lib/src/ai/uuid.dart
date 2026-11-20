// Time-ordered UUIDv7 ids, ported from pi's utils/uuid.ts.

import 'dart:math';

const _maxTimestamp = 0xffffffffffff;
const _maxSequence = (1 << 41) - 1;

/// The clock in milliseconds since the epoch; tests replace it.
int Function() uuidClock = () => DateTime.now().millisecondsSinceEpoch;

/// Fills a list with random bytes; tests replace it.
void Function(List<int> bytes) uuidRandom = _secureRandom;

final _secure = Random.secure();

void _secureRandom(List<int> bytes) {
  for (var i = 0; i < bytes.length; i++) {
    bytes[i] = _secure.nextInt(256);
  }
}

var _lastTimestamp = -1;
int? _sequence;

/// A time-ordered UUIDv7. Ids made without [timestamp] never go back in time, even when the clock does; a given
/// [timestamp] is kept as it is, for ids that follow another. A 41-bit sequence, random at first and counting up,
/// orders ids of the same millisecond. Throws a RangeError for a timestamp outside 48 bits.
String uuidv7([int? timestamp]) {
  final requested = timestamp ?? uuidClock();
  if (requested < 0 || requested > _maxTimestamp) {
    throw RangeError.range(requested, 0, _maxTimestamp, 'timestamp', 'UUIDv7 timestamp must fit 48 bits');
  }

  final effective = timestamp ?? max(requested, _lastTimestamp);
  if (timestamp == null) {
    _lastTimestamp = effective;
  }

  final bytes = List<int>.filled(16, 0);
  uuidRandom(bytes);
  final sequence = _sequence;
  if (sequence == null) {
    _sequence = (bytes[1] << 32) | (bytes[2] << 24) | (bytes[3] << 16) | (bytes[4] << 8) | bytes[5];
  } else {
    if (sequence == _maxSequence) {
      throw RangeError('UUIDv7 generator sequence exhausted');
    }

    _sequence = sequence + 1;
  }

  final s = _sequence!;
  for (var i = 5; i >= 0; i--) {
    bytes[i] = (effective >> ((5 - i) * 8)) & 0xff;
  }

  bytes[6] = 0x70 | ((s >> 37) & 0x0f);
  bytes[7] = (s >> 29) & 0xff;
  bytes[8] = 0x80 | ((s >> 23) & 0x3f);
  bytes[9] = (s >> 15) & 0xff;
  bytes[10] = (s >> 7) & 0xff;
  bytes[11] = ((s & 0x7f) << 1) | (bytes[11] & 0x01);

  final hex = [for (final b in bytes) b.toRadixString(16).padLeft(2, '0')];
  return '${hex.sublist(0, 4).join()}-${hex.sublist(4, 6).join()}-${hex.sublist(6, 8).join()}-'
      '${hex.sublist(8, 10).join()}-${hex.sublist(10).join()}';
}
