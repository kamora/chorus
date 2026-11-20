// Ported from pi's packages/ai/test/uuid.test.ts. pi's checks for fractional, NaN and infinite timestamps have no Dart
// counterpart: the timestamp is an int.

import 'package:chorus/src/ai/uuid.dart';
import 'package:test/test.dart';

final pattern = RegExp(r'^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$');
const timestamp = 0x0123456789ab;

int parseTimestamp(String uuid) => int.parse(uuid.replaceAll('-', '').substring(0, 12), radix: 16);

void main() {
  final clock = uuidClock;
  final random = uuidRandom;
  tearDown(() {
    uuidClock = clock;
    uuidRandom = random;
  });

  test('generates ordered UUIDv7s while preserving follower timestamps', () {
    var now = timestamp;
    uuidClock = () => now;
    final first = uuidv7();
    final second = uuidv7();
    now = timestamp - 1;
    final afterRollback = uuidv7();
    now = timestamp + 1;
    final afterAdvance = uuidv7();
    final ordinary = [first, second, afterRollback, afterAdvance];
    const follower = timestamp - 1000;
    final followers = [uuidv7(follower), uuidv7(follower)];

    for (final id in [...ordinary, ...followers]) {
      expect(id, matches(pattern));
    }

    expect(ordinary, [...ordinary]..sort());
    expect(ordinary.toSet(), hasLength(ordinary.length));
    expect(ordinary.map(parseTimestamp), [timestamp, timestamp, timestamp, timestamp + 1]);
    expect(followers.map(parseTimestamp), [follower, follower]);
    expect(followers.toSet(), hasLength(2));
  });

  test('uses fresh randomness for every UUID tail', () {
    var byte = 0;
    uuidRandom = (bytes) => bytes.fillRange(0, bytes.length, ++byte);
    expect([uuidv7(timestamp).substring(28), uuidv7(timestamp).substring(28)], ['01010101', '02020202']);
  });

  for (final t in [0, (1 << 48) - 1]) {
    test('accepts timestamp boundary $t', () {
      expect(parseTimestamp(uuidv7(t)), t);
    });
  }

  for (final t in [-1, 1 << 48]) {
    test('rejects invalid timestamp $t', () {
      expect(() => uuidv7(t), throwsRangeError);
    });
  }
}
