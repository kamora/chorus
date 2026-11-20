# chorus

Build AI agents in Dart with one API for Anthropic, OpenAI and Google models: stream replies, run tools and keep
conversations going without a separate integration for each provider.

chorus is a port of the provider layer and agent loop of [pi](https://github.com/earendil-works/pi)'s agent toolkit
(pi-ai and pi-agent-core). It does not include pi's coding-agent CLI or terminal UI. Messages read and write pi's JSON
format.

It needs Dart 3.13 or later. It is tested on the Dart VM; it also builds for the web, but has not been run in a
browser, and Flutter is untested.

## Install

```sh
dart pub add chorus
```

## Use

Choose a model from the built-in catalog and pass your API key with the request; chorus reads no environment
variables. `builtinModels()` sends each request to the model's provider, and the reply's usage includes a cost
estimated from the catalog's prices.

```dart
import 'package:chorus/chorus.dart';

final models = builtinModels();
final model = getBuiltinModel('openai', 'gpt-6-luna')!;
final reply = await models.completeSimple(model, [
  UserMessage([TextContent('Say hello in French.')], timestamp: DateTime.now().millisecondsSinceEpoch),
], SimpleStreamOptions(apiKey: key));

print(reply.content.whereType<TextContent>().map((c) => c.text).join());
print('${reply.usage.input} in, ${reply.usage.output} out, \$${reply.usage.cost.total}');
```

`streamSimple` gives the same reply as events, as they arrive:

```dart
await for (final event in models.streamSimple(model, messages, SimpleStreamOptions(apiKey: key))) {
  switch (event) {
    case TextDeltaEvent(:final delta):
      stdout.write(delta);
    case ErrorEvent(:final error):
      stderr.writeln(error.errorMessage);
    default:
      break;
  }
}
```

To switch providers, change the model and the key; what is available depends on the model. A `reasoning` level turns
thinking on, mapped to what each model takes (Anthropic's adaptive thinking or budgets, OpenAI's reasoning effort,
Gemini's thinking levels or budgets):

```dart
SimpleStreamOptions(apiKey: key, reasoning: ThinkingLevel.medium)
```

### An agent with tools

An `Agent` keeps the transcript and runs the loop: it sends the conversation, runs the tool calls the model makes,
sends the results back, and repeats until the model answers. Tool arguments are checked against the tool's JSON
Schema first. From [`example/example.dart`](example/example.dart):

```dart
final agent = Agent(
  model: getBuiltinModel('anthropic', 'claude-haiku-5-5')!,
  systemPrompt: 'Be brief. Use the add tool for arithmetic.',
  tools: [
    AgentTool(
      name: 'add',
      description: 'Adds two numbers.',
      parameters: const {
        'type': 'object',
        'properties': {
          'a': {'type': 'number'},
          'b': {'type': 'number'},
        },
        'required': ['a', 'b'],
      },
      execute: (id, args, signal, onUpdate) async {
        final json = args as Json;
        return AgentToolResult([TextContent('${(json['a'] as num) + (json['b'] as num)}')]);
      },
    ),
  ],
  options: SimpleStreamOptions(apiKey: key),
  stream: models.streamSimple,
);

agent.subscribe((event, signal) {
  switch (event) {
    case MessageUpdate(event: TextDeltaEvent(:final delta)):
      stdout.write(delta);
    case ToolExecutionStart(:final call):
      stdout.writeln('[${call.name} ${call.arguments}]');
    default:
      break;
  }
});
await agent.promptText('What is 1234 + 4321?');
```

The agent also takes hooks before and after each tool call, messages to steer it or follow up while it runs, tools
that change between turns, and `abort()`.

### Testing without a provider

`FauxProvider` answers from a script, streams like a real provider and estimates usage, so agents can be tested
offline:

```dart
final faux = FauxProvider();
final models = Models()..setProvider(faux.provider);
faux.setResponses([
  fauxAssistantMessage([fauxToolCall('add', {'a': 2, 'b': 3})], stopReason: StopReason.toolUse),
  fauxAssistantMessage('5'),
]);
final agent = Agent(model: faux.getModel()!, stream: models.streamSimple);
```

## Providers

Each provider sits on a small client of its own, cut down from the official TypeScript SDK to the calls chorus needs.

| Provider | API | Client |
|---|---|---|
| Anthropic | Messages | [anthropic_sdk_lite_dart](https://pub.dev/packages/anthropic_sdk_lite_dart) |
| OpenAI | Responses | [openai_sdk_lite_dart](https://pub.dev/packages/openai_sdk_lite_dart) |
| Google | Gemini (`generateContent`) | [google_sdk_lite_dart](https://pub.dev/packages/google_sdk_lite_dart) |

All three stream text, thinking and tool calls, take images, support strict tool schemas, keep a model's reasoning
across turns so it can continue from it, use prompt caching where the API offers it, retry failed requests when asked
(`maxRetries`), and can be aborted. Each also has its own features, such as Anthropic's message threads and fallback
models, OpenAI's service tiers, and changing the effort or the tools in the middle of a conversation without losing
the cached prompt.

## Scope and limitations

- No session storage, authentication or proxy: keys and HTTP clients are passed with each request.
- Not ported: sampling parameters such as `top_p`, OpenAI's grammar tools and Sign in with ChatGPT, Vertex AI and
  Cloud Code Assist, telemetry.
- Tool arguments are validated with [pluto](https://pub.dev/packages/pluto), a Dart port of pi-schema that supports
  standard JSON Schema only.

### For pi users

- pi's `continue()` is `proceed()` and its `continue` turn decision is `TurnDecision.proceed`, since `continue` is
  reserved in Dart; `onProviderStreamEvent` is `StreamOptions.onEvent`.
- The Google provider builds the REST body the Gen AI SDK would send for pi's request, so `onPayload` sees that body
  rather than the SDK's parameters.
- pi's identity headers and its OAuth "stealth mode" are not ported.

## Tests

`dart test` runs offline. Most tests are ported from pi's own, with fixtures taken from pi's sessions; chorus adds
its own where pi's leave gaps, and mutation testing (bugs planted in each provider) checked that they catch errors.

The live checks call the real APIs and cost a few cents. Each takes its key from the environment and a model from
the catalog:

```sh
ANTHROPIC_API_KEY=... dart run tool/live.dart [model]
OPENAI_API_KEY=... dart run tool/live_openai.dart [model]
GEMINI_API_KEY=... dart run tool/live_google.dart [model]
```

They cover a plain answer, a tool round trip through the Agent, thinking carried into a second turn, prompt caching,
images, a real API error, and each provider's own features, reading the requests actually sent. All three passed on
21 November 2026; Gemini's budget thinking was skipped, as the Gemini 2.5 models that use budgets are no longer
offered to new API keys.

## License

MIT, like pi. Ported from pi, Copyright (c) 2025 Mario Zechner. See `LICENSE`.
