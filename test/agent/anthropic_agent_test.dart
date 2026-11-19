import 'dart:convert';

import 'package:chorus/chorus.dart';
import 'package:test/test.dart';

import '../ai/anthropic_fake.dart';

// chorus's own: the agent loop on the Anthropic provider, through a fake server, for one tool round trip.

void main() {
  test('runs a tool the model calls and sends its result back', () async {
    final server = FakeServer.sequence([
      (
        200,
        sse([
          {
            'type': 'message_start',
            'message': {
              'id': 'msg_1',
              'usage': {'input_tokens': 10, 'output_tokens': 0},
            },
          },
          {
            'type': 'content_block_start',
            'index': 0,
            'content_block': {'type': 'tool_use', 'id': 'toolu_1', 'name': 'add', 'input': <String, Object?>{}},
          },
          {
            'type': 'content_block_delta',
            'index': 0,
            'delta': {'type': 'input_json_delta', 'partial_json': '{"a": 2, '},
          },
          {
            'type': 'content_block_delta',
            'index': 0,
            'delta': {'type': 'input_json_delta', 'partial_json': '"b": 3}'},
          },
          {'type': 'content_block_stop', 'index': 0},
          {
            'type': 'message_delta',
            'delta': {'stop_reason': 'tool_use'},
            'usage': {'output_tokens': 7},
          },
          {'type': 'message_stop'},
        ]),
      ),
      (200, sse(minimalEvents())),
    ]);
    final add = AgentTool(
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
    );

    final messages = await agentLoop(
      [
        UserMessage([TextContent('What is 2 + 3?')], timestamp: 1),
      ],
      AgentContext(
        [
          SystemMessage([TextContent('You add numbers.')], timestamp: 0),
        ],
        tools: [add],
      ),
      AgentLoopConfig(model: claude(), convertToLlm: (m) => m.whereType<Message>().toList()),
      (model, messages, options) => streamAnthropic(model, messages, AnthropicOptions(client: server.client())),
    ).result;

    expect((messages.whereType<ToolResultMessage>().single.content.single as TextContent).text, '5');
    expect(((messages.last as AssistantMessage).content.single as TextContent).text, 'Hello');

    final first = jsonDecode(server.requests.first.body) as Json;
    expect((first['tools'] as List).single['name'], 'add');
    expect((first['system'] as List).single['text'], 'You add numbers.');

    final second = jsonDecode(server.requests.last.body) as Json;
    expect((second['messages'] as List).map((m) => (m as Json)['role']), ['user', 'assistant', 'user']);
    expect(((second['messages'] as List).last['content'] as List).single, {
      'type': 'tool_result',
      'tool_use_id': 'toolu_1',
      'content': '5',
      'is_error': false,
      'cache_control': {'type': 'ephemeral'},
    });
  });
}
