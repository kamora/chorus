/// A fast deterministic hash (pi's `shortHash`): two 32-bit lanes over the UTF-16 units.
String shortHash(String text) {
  int mul(int a, int b) => (a * b) & 0xffffffff;
  var h1 = 0xdeadbeef;
  var h2 = 0x41c6ce57;
  for (final c in text.codeUnits) {
    h1 = mul(h1 ^ c, 2654435761);
    h2 = mul(h2 ^ c, 1597334677);
  }

  h1 = mul(h1 ^ (h1 >> 16), 2246822507) ^ mul(h2 ^ (h2 >> 13), 3266489909);
  h2 = mul(h2 ^ (h2 >> 16), 2246822507) ^ mul(h1 ^ (h1 >> 13), 3266489909);
  return h2.toRadixString(36) + h1.toRadixString(36);
}
