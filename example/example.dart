// An agent with one tool, streaming its answer. Run with a key in the environment:
//
//     ANTHROPIC_API_KEY=... dart run example/example.dart

import 'dart:io';

import 'package:chorus/chorus.dart';

Future<void> main() async {
  final models = builtinModels();
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
    options: SimpleStreamOptions(apiKey: Platform.environment['ANTHROPIC_API_KEY']),
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
  stdout.writeln();
}
