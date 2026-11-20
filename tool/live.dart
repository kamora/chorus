/// Live checks against the Anthropic API: a plain answer, a tool round trip through the Agent, adaptive thinking, and
/// signed thinking sent back in a second turn.
///
/// Run with a key in the environment; it is never printed:
///
///     ANTHROPIC_API_KEY=... dart run tool/live.dart [model]
///
/// The model defaults to claude-haiku-5-5. Exits with 1 when a check fails.
library;

import 'dart:io';

import 'package:chorus/chorus.dart';

Model _model(String id) {
  return Model(
    id: id,
    name: id,
    api: 'anthropic-messages',
    provider: 'anthropic',
    baseUrl: 'https://api.anthropic.com',
    input: const ['text', 'image'],
    cost: const ModelCost(input: 1, output: 5, cacheRead: 0.1, cacheWrite: 1.25),
    reasoning: true,
    contextWindow: 200000,
    maxTokens: 8000,
    compat: const {'forceAdaptiveThinking': true},
  );
}

String _text(AssistantMessage message) => message.content.whereType<TextContent>().map((c) => c.text).join().trim();

var _failed = false;

void _check(String name, bool passed, AssistantMessage message) {
  _failed = _failed || !passed;
  stdout.writeln('${passed ? 'ok  ' : 'FAIL'} $name: ${message.stopReason.name}, "${_text(message)}"');
  if (message.errorMessage != null) {
    stdout.writeln('     ${message.errorMessage}');
  }
}

Future<void> main(List<String> args) async {
  final key = Platform.environment['ANTHROPIC_API_KEY'];
  if (key == null || key.isEmpty) {
    stderr.writeln('Set ANTHROPIC_API_KEY.');
    exit(2);
  }

  final model = _model(args.firstOrNull ?? 'claude-haiku-5-5');
  final options = SimpleStreamOptions(apiKey: key, maxTokens: 4000);
  final plain = await streamSimpleAnthropic(model, [
    UserMessage([TextContent('Reply with exactly: pong')], timestamp: 1),
  ], options).result;
  _check('plain answer', _text(plain).toLowerCase().contains('pong'), plain);

  final add = Agent(
    model: model,
    systemPrompt: 'Use the add tool for arithmetic, then answer with the number only.',
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
    options: options,
    stream: streamSimpleAnthropic,
  );
  await add.promptText('What is 1234 + 4321?');
  final answer = add.messages.last as AssistantMessage;
  _check(
    'tool round trip',
    add.messages.whereType<ToolResultMessage>().isNotEmpty && _text(answer).contains('5555'),
    answer,
  );

  final thinking = Agent(model: model, level: ThinkingLevel.low, options: options, stream: streamSimpleAnthropic);
  await thinking.promptText('Is 391 prime? Answer yes or no.');
  final first = thinking.messages.last as AssistantMessage;
  _check(
    'adaptive thinking',
    first.content.whereType<ThinkingContent>().any((c) => c.signature?.isNotEmpty ?? false) &&
        _text(first).toLowerCase().startsWith('no'),
    first,
  );

  await thinking.promptText('And 397?');
  final second = thinking.messages.last as AssistantMessage;
  _check(
    'thinking sent back in a 2nd turn',
    second.stopReason == StopReason.stop && _text(second).toLowerCase().startsWith('yes'),
    second,
  );

  exit(_failed ? 1 : 0);
}
