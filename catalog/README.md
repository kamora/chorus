# Model catalog

pi's built-in model catalog for the providers chorus serves: `anthropic.json`, `google.json` and `openai.json`, the
chat models of each, keyed by id. `lib/src/ai/models.generated.dart` is generated from them.

## Source

Generated on 2026-11-20 by pi's catalog generator (`packages/ai/catalog/generate.ts`, pi at commit `1e34c3b`), with
`--json-only --json-output`: the catalog pi.dev serves. The generator builds it from [models.dev](https://models.dev)
(MIT, see `LICENSE-models.dev`) with pi's corrections and per-model settings; pi is MIT (see `../LICENSE`).

## Updating

In a checkout of pi, with Node 22:

```sh
cd packages/ai
node --conditions=source --experimental-strip-types catalog/generate.ts --json-only --json-output /tmp/catalog
```

Then copy `providers/anthropic.json`, `providers/google.json` and `providers/openai.json` from the output here,
prettified with tabs, and in chorus:

```sh
dart run tool/models.dart
dart test test/ai/catalog_test.dart
```
