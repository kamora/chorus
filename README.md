# chorus

A Dart port of [pi](https://github.com/earendil-works/pi)'s agent toolkit: one API over LLM providers (pi-ai) and
the agent loop on top of it (pi-agent-core). The CLI, the terminal UI and pi's other packages are not part of it.

Work in progress. So far:

- pi-ai's message, content and event types, which read and write pi's JSON, and the event stream of one response.
- The Anthropic provider (`streamAnthropic`, `streamSimpleAnthropic`), built on
  [anthropic_sdk_lite_dart](https://pub.dev/packages/anthropic_sdk_lite_dart): prompt caching, thinking (budget and
  adaptive), tools and strict tools, images, retries, aborts, message threads, mid-conversation effort, system
  messages and tool changes in place, and server-side fallback models.
- The OpenAI Responses provider (`streamOpenAIResponses`, `streamSimpleOpenAIResponses`), built on
  [openai_sdk_lite_dart](https://github.com/kamora/openai-sdk-lite-dart): reasoning with its encrypted items replayed
  across turns, tools and strict tools, images, prompt caching with session headers, service tier pricing,
  mid-conversation effort and system messages, tool changes as `additional_tools` or client tool search, retries,
  aborts. Not ported: grammar tools, Sign in with ChatGPT, sampling parameters.
- The Google provider for the Gemini API (`streamGoogle`, `streamSimpleGoogle`), built on google_sdk_lite_dart: thinking
  as Gemini levels or token budgets, thought signatures replayed across turns, tools with strict (validated) calling,
  images in function responses, retries, aborts. It sends the REST body the Gen AI SDK would send for pi's request,
  so `onPayload` sees that body rather than the SDK's parameters. Not ported: Vertex AI and Cloud Code Assist.
- A scripted provider for tests (`FauxProvider`), and pi's utilities: context overflow detection, the cache warming
  policy, uuidv7.
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
adaptive thinking, thinking sent back in a second turn, budget-based thinking, prompt caching, an image, a real API
error, and the beta features: strict tools, mid-conversation effort, tool and system changes, message threads and
server-side fallbacks); it costs a few cents' worth of tokens:

```sh
ANTHROPIC_API_KEY=... dart run tool/live.dart [model]
```

`tool/live_openai.dart` does the same against the OpenAI Responses API (a plain answer, a strict tool round trip,
reasoning and its encrypted item sent back, prompt caching, an image, a real API error, and effort, tool and system
changes mid-conversation with the session headers), on gpt-6-luna by default:

```sh
OPENAI_API_KEY=... dart run tool/live_openai.dart [model]
```

`tool/live_google.dart` checks the Gemini API (a plain answer, a strict tool round trip, thinking at a Gemini level
and its thought signature sent back, budget-based thinking, implicit prompt caching, an image, an image a tool
returns, and a real API error), on gemini-3.1-flash-lite by default:

```sh
GEMINI_API_KEY=... dart run tool/live_google.dart [model]
```

## License

MIT, like pi. Ported from pi, Copyright (c) 2025 Mario Zechner. See `LICENSE`.
