# chorus

A Dart port of [pi](https://github.com/earendil-works/pi)'s agent toolkit: one API over LLM providers (pi-ai) and
the agent loop on top of it (pi-agent-core). The CLI, the terminal UI and pi's other packages are not part of it.

Work in progress. So far it has pi-ai's message, content and event types, which read and write pi's JSON, and the
event stream of one assistant response. The Anthropic provider comes next, built on
[anthropic_sdk_lite_dart](https://pub.dev/packages/anthropic_sdk_lite_dart).

## Tests

`dart test`. `test/fixtures/pi-messages.json` holds messages taken from pi's session fixtures, with long texts
trimmed.

## License

MIT, like pi. Ported from pi, Copyright (c) 2025 Mario Zechner. See `LICENSE`.
