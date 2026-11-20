# chorus

A Dart port of [pi](https://github.com/earendil-works/pi)'s agent toolkit: one API over LLM providers (pi-ai) and
the agent loop on top of it (pi-agent-core). The CLI, the terminal UI and pi's other packages are not part of it.

Work in progress. So far:

- pi-ai's message, content and event types, which read and write pi's JSON, and the event stream of one response.
- The Anthropic provider (`streamAnthropic`, `streamSimpleAnthropic`), built on
  [anthropic_sdk_lite_dart](https://pub.dev/packages/anthropic_sdk_lite_dart): prompt caching, thinking (budget and
  adaptive), tools, images, retries, aborts. Not yet: strict tools, message threads, mid-conversation effort, system
  messages and tool changes, and server-side fallback models.
- pi's model catalog for Anthropic, OpenAI and Google (`getBuiltinModel`, `getBuiltinModels`), vendored in `catalog/`
  from pi's generator, which builds it from [models.dev](https://models.dev); see `catalog/README.md`.
- Providers and the collection that routes requests to them (`Provider`, `Models`, `builtinModels()`): `stream`,
  `complete`, `streamSimple` and `completeSimple` pick the provider from the model. Without pi's auth layer: keys are
  request options. With `getSupportedThinkingLevels` and `clampThinkingLevel`.
- pi-agent-core's agent loop (`agentLoop`, `agentLoopContinue`, `runToolCall`): turns, tool calls in sequence or in
  parallel, before and after hooks, steering and follow-up messages, tool changes announced to the model. Tool
  arguments are coerced and validated against the tool's JSON Schema (`validateToolArguments`) with
  [pluto](https://pub.dev/packages/pluto), the Dart port of pi-schema.
- pi-agent-core's `Agent`: the transcript and tools, runs with subscribers, abort, and steering and follow-up queues.
  pi's `continue()` is `proceed()`, since `continue` is a reserved word in Dart.

## Tests

`dart test`. Most tests are ported from pi's own; `test/fixtures/pi-messages.json` holds messages taken from pi's
session fixtures, with long texts trimmed, and `test/fixtures/pi-validation-probes.json` pi's tool argument probes,
recorded with their outcomes. Nothing calls a real API.

`tool/live.dart` runs live checks against the Anthropic API (a plain answer, a tool round trip through the Agent,
adaptive thinking, and thinking sent back in a second turn); it costs a few cents' worth of tokens:

```sh
ANTHROPIC_API_KEY=... dart run tool/live.dart [model]
```

## License

MIT, like pi. Ported from pi, Copyright (c) 2025 Mario Zechner. See `LICENSE`.
