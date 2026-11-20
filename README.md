# chorus

A Dart port of [pi](https://github.com/earendil-works/pi)'s agent toolkit: one API over LLM providers (pi-ai) and
the agent loop on top of it (pi-agent-core). The CLI, the terminal UI and pi's other packages are not part of it.

Work in progress. So far:

- pi-ai's message, content and event types, which read and write pi's JSON, and the event stream of one response.
- The Anthropic provider (`streamAnthropic`, `streamSimpleAnthropic`), built on
  [anthropic_sdk_lite_dart](https://pub.dev/packages/anthropic_sdk_lite_dart): prompt caching, thinking (budget and
  adaptive), tools, images, retries, aborts. Not yet: strict tools, message threads, mid-conversation effort, system
  messages and tool changes, server-side fallback models and the model catalog.
- pi-agent-core's agent loop (`agentLoop`, `agentLoopContinue`, `runToolCall`): turns, tool calls in sequence or in
  parallel, before and after hooks, steering and follow-up messages, tool changes announced to the model. Tool
  arguments are checked only by a tool's own `validate`; pi's JSON Schema validation (pi-schema) comes later as its
  own package.
- pi-agent-core's `Agent`: the transcript and tools, runs with subscribers, abort, and steering and follow-up queues.
  pi's `continue()` is `proceed()`, since `continue` is a reserved word in Dart.

## Tests

`dart test`. Most tests are ported from pi's own; `test/fixtures/pi-messages.json` holds messages taken from pi's
session fixtures, with long texts trimmed. Nothing calls a real API.

## License

MIT, like pi. Ported from pi, Copyright (c) 2025 Mario Zechner. See `LICENSE`.
