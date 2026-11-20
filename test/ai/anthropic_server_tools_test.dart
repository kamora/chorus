import 'package:chorus/chorus.dart';
import 'package:chorus/src/ai/transform.dart';
import 'package:test/test.dart';

import 'anthropic_fake.dart';

// chorus's own: Anthropic's server tools, which pi does not have. The blocks follow a real web search reply of Claude
// Opus 5.5 (2026-11-21), with the encrypted payloads shortened.

const search = {'type': 'web_search_20260209', 'name': 'web_search', 'max_uses': 2};

// A code execution call whose input streams in, the web search it makes, both results, then the answer.
List<Json> searchEvents() {
  return [
    {
      'type': 'message_start',
      'message': {
        'id': 'msg_search',
        'usage': {'input_tokens': 19928, 'output_tokens': 0},
      },
    },
    {
      'type': 'content_block_start',
      'index': 0,
      'content_block': {'type': 'server_tool_use', 'id': 'srvtoolu_code', 'name': 'code_execution', 'input': {}},
    },
    {
      'type': 'content_block_delta',
      'index': 0,
      'delta': {'type': 'input_json_delta', 'partial_json': '{"code": "print(1'},
    },
    {
      'type': 'content_block_delta',
      'index': 0,
      'delta': {'type': 'input_json_delta', 'partial_json': ')"}'},
    },
    {'type': 'content_block_stop', 'index': 0},
    {
      'type': 'content_block_start',
      'index': 1,
      'content_block': {
        'type': 'server_tool_use',
        'id': 'srvtoolu_search',
        'name': 'web_search',
        'input': {'query': 'latest Formula 1 winner'},
        'caller': {'type': 'code_execution_20260120', 'tool_id': 'srvtoolu_code'},
      },
    },
    {'type': 'content_block_stop', 'index': 1},
    {
      'type': 'content_block_start',
      'index': 2,
      'content_block': {
        'type': 'web_search_tool_result',
        'tool_use_id': 'srvtoolu_search',
        'content': [
          {
            'type': 'web_search_result',
            'title': 'Grand Prix',
            'url': 'https://example.com/gp',
            'encrypted_content': 'Eu',
          },
        ],
      },
    },
    {'type': 'content_block_stop', 'index': 2},
    {
      'type': 'content_block_start',
      'index': 3,
      'content_block': {
        'type': 'code_execution_tool_result',
        'tool_use_id': 'srvtoolu_code',
        'content': {'type': 'encrypted_code_execution_result', 'encrypted_stdout': 'Eo'},
      },
    },
    {'type': 'content_block_stop', 'index': 3},
    {
      'type': 'content_block_start',
      'index': 4,
      'content_block': {'type': 'text', 'text': ''},
    },
    {
      'type': 'content_block_delta',
      'index': 4,
      'delta': {'type': 'text_delta', 'text': 'Verstappen won.'},
    },
    {'type': 'content_block_stop', 'index': 4},
    {
      'type': 'message_delta',
      'delta': {'stop_reason': 'end_turn'},
      'usage': {'output_tokens': 428},
    },
    {'type': 'message_stop'},
  ];
}

void main() {
  test('server blocks are kept as they came, a streamed input included, with events only for the answer', () async {
    final events = <AssistantMessageEvent>[];
    final stream = streamAnthropic(claude(), [
      user('Who won?'),
    ], AnthropicOptions(client: FakeServer(sse(searchEvents())).client()));
    stream.listen(events.add);
    final reply = await stream.result;

    expect(
      [for (final c in reply.content) c.runtimeType],
      [ServerToolContent, ServerToolContent, ServerToolContent, ServerToolContent, TextContent],
    );
    final blocks = [for (final c in reply.content.whereType<ServerToolContent>()) c.block];
    expect(blocks[0]['input'], {'code': 'print(1)'});
    expect(blocks[1], searchEvents()[5]['content_block']);
    expect(blocks[2], searchEvents()[7]['content_block']);
    expect(blocks[3], searchEvents()[9]['content_block']);
    expect((reply.content.last as TextContent).text, 'Verstappen won.');
    expect(reply.stopReason, StopReason.stop);
    expect(events.whereType<TextStartEvent>().single.index, 4);
    expect(events.whereType<ToolCallStartEvent>(), isEmpty);
    expect(events.whereType<ThinkingStartEvent>(), isEmpty);
  });

  test('server tools are declared first, and the next request sends the blocks back unchanged', () async {
    final model = claude();
    final server = FakeServer(sse(searchEvents()));
    final options = AnthropicOptions(client: server.client(), serverTools: const [search]);
    final first = await streamAnthropic(model, [user('Who won?')], options).result;
    expect(server.body['tools'], [search]);

    final messages = <Message>[
      SystemMessage(
        const [],
        added: [
          Tool('remember', 'Keeps a fact.', const {'type': 'object', 'properties': <String, Object?>{}}),
        ],
        timestamp: 0,
      ),
      user('Who won?'),
      first,
      user('And before that?', timestamp: 2),
    ];
    await streamAnthropic(model, messages, options).result;

    final tools = server.body['tools'] as List<Object?>;
    expect(tools.first, search);
    expect((tools[1] as Json)['name'], 'remember');
    final sent = (server.body['messages'] as List<Object?>)[1] as Json;
    expect(sent['role'], 'assistant');
    expect(sent['content'], [
      for (final c in first.content.whereType<ServerToolContent>()) c.block,
      {'type': 'text', 'text': 'Verstappen won.'},
    ]);
  });

  test('a server block goes back only to its own API and provider, and survives JSON', () {
    final block = ServerToolContent({
      'type': 'web_search_tool_result',
      'tool_use_id': 'srvtoolu_search',
      'content': [],
    });
    final reply = AssistantMessage(
      content: [block, TextContent('Verstappen won.')],
      api: 'anthropic-messages',
      provider: 'anthropic',
      model: 'claude-test',
      stopReason: StopReason.stop,
      timestamp: 1,
    );

    expect(Content.fromJson(block.toJson()).toJson(), block.toJson());
    expect(block.toJson()['type'], 'serverTool');

    final same = transformMessages([user('Who won?'), reply], claude(id: 'claude-other'));
    expect((same[1] as AssistantMessage).content.first, isA<ServerToolContent>());

    final other = transformMessages([user('Who won?'), reply], claude(provider: 'proxy'));
    expect([for (final c in (other[1] as AssistantMessage).content) c.runtimeType], [TextContent]);
  });
}
