## 0.1.0

- First release: pi-ai's provider layer and pi-agent-core's agent loop, ported to Dart.
- Providers for Anthropic (Messages), OpenAI (Responses) and Google (Gemini), on their own lite clients, with
  streaming, tools and strict tools, thinking carried across turns, images, prompt caching, retries and aborts.
- pi's model catalog for the three providers, and `builtinModels()` to route requests by model.
- `Agent` and the agent loop: tool calls in sequence or in parallel, hooks around tool calls, steering and follow-up
  messages, tool changes between turns. Tool arguments are validated with pluto.
- `FauxProvider` for offline tests, and pi's utilities: context overflow detection, the cache warming policy, uuidv7.
