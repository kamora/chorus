// The OpenAI Responses cases of pi's packages/ai/test/transcript-tool-changes.test.ts, and its resolveTranscriptTools
// test.

import 'package:chorus/chorus.dart';
import 'package:chorus/src/ai/transcript.dart';
import 'package:test/test.dart';

Tool tool(String name, [String? description]) {
  return Tool(name, description ?? '$name tool', const {'type': 'object', 'properties': <String, Object?>{}});
}

final baseTool = tool('base_tool');
final lateTool = tool('late_tool');
final dropTool = tool('drop_tool');

Model model(Json compat) {
  return Model(
    id: 'gpt-5.4',
    name: 'GPT-5.4',
    api: 'openai-responses',
    provider: 'openai',
    baseUrl: 'http://127.0.0.1:9',
    reasoning: true,
    cost: const ModelCost(input: 0, output: 0, cacheRead: 0, cacheWrite: 0),
    contextWindow: 100000,
    maxTokens: 1000,
    compat: compat,
  );
}

final additionalTools = model({'supportsMidConvoSystemMessages': true, 'supportsAdditionalTools': true});

SystemMessage system(
  String text, {
  List<Tool>? added,
  List<String>? removed,
  Map<String, String?>? sections,
  int at = 0,
}) {
  return SystemMessage([TextContent(text)], added: added, removed: removed, sections: sections, timestamp: at);
}

UserMessage user(String text, [int at = 1]) => UserMessage([TextContent(text)], timestamp: at);

final changes = <Message>[
  system(
    'base prompt',
    sections: {'rules': '<rules>\nold rules\n</rules>', 'docs': '<docs>\nread docs\n</docs>'},
    added: [baseTool],
  ),
  user('before'),
  system(
    'updated guidance',
    sections: {'rules': '<rules>\nnew rules\n</rules>', 'docs': null},
    removed: ['base_tool'],
    added: [lateTool],
    at: 2,
  ),
];

final additions = <Message>[
  system('base prompt', added: [baseTool]),
  user('before'),
  system('updated guidance', added: [lateTool], at: 2),
];

class Captured implements Exception {}

/// The request body for [messages], captured through streamSimple.
Future<Json> payload(Model m, List<Message> messages) async {
  Json? captured;
  await streamSimpleOpenAIResponses(
    m,
    messages,
    SimpleStreamOptions(
      apiKey: 'test-key',
      onPayload: (body, _) {
        captured = body;
        throw Captured();
      },
    ),
  ).result;
  return captured!;
}

List<Json> input(Json payload) => (payload['input'] as List<Object?>).cast<Json>();

List<String>? names(Object? tools) => [
  for (final t in (tools as List<Object?>?) ?? const []) (t as Json)['name'] as String,
];

/// One request after each step: initial tools, the removal of drop_tool, then two loads.
Future<List<Json>> history(Model m) async {
  final messages = <Message>[
    system('base prompt', added: [baseTool, dropTool]),
  ];
  final payloads = <Json>[];
  for (final step in [
    [user('first')],
    [
      system('', removed: ['drop_tool'], at: 2),
      user('after removal', 3),
    ],
    [
      system('', added: [lateTool], at: 4),
      user('after first load', 5),
    ],
    [
      system('', added: [tool('later_tool')], at: 6),
      user('after second load', 7),
    ],
  ]) {
    messages.addAll(step);
    payloads.add(await payload(m, [...messages]));
  }

  return payloads;
}

/// Every request's input starts with the previous request's input.
void expectGrowingPrefix(List<List<Json>> inputs) {
  for (var i = 1; i < inputs.length; i++) {
    expect(inputs[i].take(inputs[i - 1].length), inputs[i - 1]);
  }
}

void main() {
  test('declares every OpenAI tool in additional_tools at its developer message', () async {
    final p = await payload(additionalTools, additions);

    // A later additional_tools item cannot replace a top-level tool, so no tool is declared there.
    expect(p.containsKey('tools'), isFalse);
    expect(
      [for (final i in input(p).where((i) => i['type'] == 'additional_tools')) names(i['tools'])!.first],
      ['base_tool', 'late_tool'],
    );

    // The incremental notice keeps models from reading a later item as the whole tool set.
    final shape = [for (final i in input(p)) i['type'] ?? i['content']];
    expect(shape, hasLength(6));
    expect(shape.take(2), ['additional_tools', 'base prompt']);
    expect(shape[2], isA<List<Object?>>());
    expect(shape[3], contains('incremental tools update'));
    expect(shape.skip(4), ['additional_tools', 'updated guidance']);
  });

  test('maps system-message additions into synthetic tool search', () async {
    final p = await payload(model({'supportsMidConvoSystemMessages': true, 'supportsToolSearch': true}), additions);
    expect(names(p['tools']), ['base_tool']);
    expect([for (final i in input(p)) i['type']], contains('tool_search_call'));
    expect(names(input(p).firstWhere((i) => i['type'] == 'tool_search_output')['tools']), ['late_tool']);
  });

  test('folds OpenAI updates into the leading developer message without native support', () async {
    final m = Model(
      id: 'gpt-4.1',
      name: 'GPT-4.1',
      api: 'openai-responses',
      provider: 'openai',
      baseUrl: 'http://127.0.0.1:9',
      reasoning: true,
      cost: const ModelCost(input: 0, output: 0, cacheRead: 0, cacheWrite: 0),
      contextWindow: 100000,
      maxTokens: 1000,
      compat: const {'supportsAdditionalTools': true},
    );
    final p = await payload(m, changes);
    expect(names(p['tools']), ['late_tool']);
    expect([for (final i in input(p)) i['type'] ?? i['role']], ['developer', 'user']);
    expect(input(p).first['content'], 'base prompt\n\nupdated guidance\n\n<rules>\nnew rules\n</rules>');
  });

  test('announces OpenAI removals as text', () async {
    final p = await payload(additionalTools, changes);
    expect(p.containsKey('tools'), isFalse);
    expect(
      [for (final i in input(p).where((i) => i['type'] == 'additional_tools')) names(i['tools'])!.first],
      ['base_tool', 'late_tool'],
    );
    final update = input(p).last['content'] as String;
    expect(update, contains('updated guidance'));
    expect(update, contains('The following tools are no longer available. Do not call them:\n- base_tool'));
  });

  test('redefines an OpenAI tool with additional_tools instead of announcing a removal', () async {
    final p = await payload(additionalTools, [
      system('base prompt', added: [baseTool]),
      user('before'),
      system('', removed: ['base_tool'], added: [tool('base_tool', 'changed')], at: 2),
    ]);
    expect(p.containsKey('tools'), isFalse);
    final items = input(p);
    expect(items, hasLength(5));
    expect(items[0]['type'], 'additional_tools');
    expect(names(items[0]['tools']), ['base_tool']);
    expect(((items[0]['tools'] as List<Object?>).single as Json)['description'], 'base_tool tool');
    expect((items[1]['role'], items[1]['content']), ('developer', 'base prompt'));
    expect(items[2]['role'], 'user');
    expect(items[3]['role'], 'developer');
    expect(items[3]['content'], contains('latest definition replaces the earlier one'));
    expect(((items[4]['tools'] as List<Object?>).single as Json)['description'], 'changed');
  });

  // Deferred loads after a removal must not rewrite the cached prefix.
  test('keeps the OpenAI request prefix across a removal and later loads', () async {
    final payloads = await history(additionalTools);
    for (final p in payloads) {
      expect(p.containsKey('tools'), isFalse);
    }

    expectGrowingPrefix([for (final p in payloads) input(p)]);
    final last = input(payloads.last);
    expect(
      [for (final i in last.where((i) => i['type'] == 'additional_tools')) names(i['tools'])],
      [
        ['base_tool', 'drop_tool'],
        ['late_tool'],
        ['later_tool'],
      ],
    );
    expect(last.any((i) => '${i['content']}'.contains('no longer available. Do not call them:\n- drop_tool')), isTrue);

    // Each later load restates the tools that are still unavailable.
    final notices = [
      for (final i in last)
        if ('${i['content']}'.contains('incremental tools update')) i['content'],
    ];
    expect(notices, [contains('\nStill unavailable: drop_tool.'), contains('\nStill unavailable: drop_tool.')]);
  });

  test('stops listing a tool as unavailable once it is declared again', () async {
    final p = await payload(additionalTools, [
      system('base prompt', added: [baseTool, dropTool]),
      system('', removed: ['drop_tool'], at: 1),
      system('', added: [dropTool], at: 2),
      user('go', 3),
    ]);
    final notice = input(p).firstWhere((i) => '${i['content']}'.contains('incremental tools update'))['content'];
    expect(notice, isNot(contains('Still unavailable')));
    final item = input(p)[input(p).length - 2];
    expect(item['type'], 'additional_tools');
    expect(names(item['tools']), ['drop_tool']);
  });

  test('does not list a tool as still unavailable before its removal is announced', () async {
    final p = await payload(additionalTools, [
      system('base prompt', added: [baseTool, dropTool]),
      system('', added: [lateTool], removed: ['drop_tool'], at: 1),
      user('go', 2),
    ]);

    // Like Codex: the notice, then the tools, then the removal notice.
    final tail = input(p).skip(input(p).length - 4).toList();
    expect(tail[0]['content'], allOf(contains('incremental tools update'), isNot(contains('Still unavailable'))));
    expect(tail[1]['type'], 'additional_tools');
    expect(names(tail[1]['tools']), ['late_tool']);
    expect(tail[2]['content'], contains('no longer available. Do not call them:\n- drop_tool'));
    expect(tail[3]['role'], 'user');
  });

  // Synthetic tool search cannot remove tools, so a removal rewrites the prefix once.
  test('keeps the OpenAI tool search prefix across loads after a removal', () async {
    final [before, ...after] = await history(
      model({'supportsMidConvoSystemMessages': true, 'supportsToolSearch': true}),
    );
    expect(names(before['tools']), ['base_tool', 'drop_tool']);
    for (final p in after) {
      expect(names(p['tools']), ['base_tool']);
    }

    expectGrowingPrefix([for (final p in after) input(p)]);
    expect(
      [for (final i in input(after.last).where((i) => i['type'] == 'tool_search_output')) names(i['tools'])!.first],
      ['late_tool', 'later_tool'],
    );
  });

  // Removals and redeclarations move the top-level boundary once; plain additions stay anchored.
  test('resolves the top-level tool boundary per transport', () {
    final late = [tool('late_tool', 'x'), tool('late_tool', 'y')];
    final messages = <Message>[
      system('base prompt', added: [baseTool, dropTool]),
      system('', removed: ['drop_tool'], at: 1),
      system('', added: [dropTool], at: 2),
      system('', added: [late[0]], at: 3),
      system('', added: [late[1]], at: 4),
    ];

    ({List<Tool> tools, int folded}) resolve(int count, ToolSupport support) {
      return resolveTranscriptTools(messages.take(count).toList(), support);
    }

    expect(resolve(2, ToolSupport.additions).tools, [baseTool]);
    expect(resolve(2, ToolSupport.additions).folded, 2);

    // Kimi K3 ignores a removed tool that a later tool message declares again, so the re-added drop_tool moves the
    // boundary; the new late_tool after it stays anchored.
    expect(resolve(4, ToolSupport.additions).tools, [baseTool, dropTool]);
    expect(resolve(4, ToolSupport.additions).folded, 3);

    // The redefinition of late_tool moves the boundary to its message.
    expect(resolve(5, ToolSupport.additions).tools, [baseTool, dropTool, late[1]]);
    expect(resolve(5, ToolSupport.additions).folded, 5);
    expect(resolve(5, ToolSupport.changes).tools, isEmpty);
    expect(resolve(5, ToolSupport.changes).folded, 0);
    expect(resolve(5, ToolSupport.none).tools, [baseTool, dropTool, late[1]]);
    expect(resolve(5, ToolSupport.none).folded, 5);
  });
}
