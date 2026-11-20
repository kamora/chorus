// Native system messages and tool changes. Ported from pi's transcript-tool-changes.test.ts (its Anthropic tests),
// system-message-replay.test.ts (the rendering of a later system message) and the inline-addition test of
// anthropic-strict-tool-fallback.test.ts.

import 'dart:convert';

import 'package:chorus/chorus.dart';
import 'package:chorus/src/ai/transcript.dart';
import 'package:test/test.dart';

import 'anthropic_fake.dart';

Tool tool(String name, [String? description]) {
  return Tool(name, description ?? '$name tool', const {'type': 'object', 'properties': <String, Object?>{}});
}

final baseTool = tool('base_tool');
final lateTool = tool('late_tool');

final native = claude(
  id: 'claude-opus-5',
  compat: const {'supportsMidConvoSystemMessages': true, 'supportsMidConvoToolChanges': true},
);

SystemMessage system(
  String text, {
  Map<String, String?>? sections,
  List<Tool>? added,
  List<String>? removed,
  int timestamp = 0,
}) {
  return SystemMessage([TextContent(text)], sections: sections, added: added, removed: removed, timestamp: timestamp);
}

final context = <Message>[
  system(
    'base prompt',
    sections: const {'rules': '<rules>\nold rules\n</rules>', 'docs': '<docs>\nread docs\n</docs>'},
    added: [baseTool],
  ),
  user('before'),
  system(
    'updated guidance',
    sections: const {'rules': '<rules>\nnew rules\n</rules>', 'docs': null},
    removed: ['base_tool'],
    added: [lateTool],
    timestamp: 2,
  ),
];

/// The request body of one request of [model] for [messages], stopped once its payload is captured.
Future<Json> capture(Model model, List<Message> messages) async {
  Json? payload;
  await streamAnthropic(
    model,
    messages,
    AnthropicOptions(
      apiKey: 'test-key',
      onPayload: (body, _) {
        payload = jsonDecode(jsonEncode(body)) as Json;
        throw StateError('payload captured');
      },
    ),
  ).result;
  return payload!;
}

List<Json> tools(Json payload) => (payload['tools'] as List<Object?>? ?? const []).cast<Json>();

List<Json> messages(Json payload) => (payload['messages'] as List<Object?>).cast<Json>();

List<Json> content(Json message) => (message['content'] as List<Object?>).cast<Json>();

List<Object?> betas(Json payload) => payload['betas'] as List<Object?>? ?? const [];

void main() {
  group('transcript system messages', () {
    test('sends Anthropic updates and tool changes in native system messages', () async {
      final payload = await capture(native, context);
      expect(betas(payload), contains('inline-tools-2026-09-15'));
      expect(betas(payload), isNot(contains('mid-conversation-tool-changes-2026-07-01')));
      expect(
        [for (final b in payload['system'] as List<Object?>) (b as Json)['text']],
        ['base prompt\n\n<rules>\nold rules\n</rules>\n\n<docs>\nread docs\n</docs>'],
      );

      // The top-level list holds only the initial tools (with the cache breakpoint) and the deferred placeholder.
      expect(tools(payload).map((t) => t['name']), ['base_tool', '__pi_deferred_placeholder__']);
      expect(tools(payload)[0]['cache_control'], {'type': 'ephemeral'});
      expect(tools(payload)[0].containsKey('defer_loading'), isFalse);
      expect(tools(payload)[1]['defer_loading'], isTrue);
      expect(tools(payload)[1].containsKey('cache_control'), isFalse);

      final update = messages(payload).last;
      expect(update['role'], 'system');
      final blocks = content(update);
      expect(blocks.map((b) => b['type']), ['text', 'tool_removal', 'tool_addition']);
      expect(blocks[1]['tool'], {'type': 'tool_reference', 'name': 'base_tool'});
      final addition = blocks[2]['tool'] as Json;
      expect(addition['type'], 'tool_definition');
      final definition = addition['definition'] as Json;
      expect((definition['name'], definition['description']), ('late_tool', 'late_tool tool'));
      expect(blocks[2]['cache_control'], {'type': 'ephemeral'});

      // Cache control goes on the block, never on the inline definition; nothing is deferred.
      expect(definition.containsKey('cache_control'), isFalse);
      expect(definition.containsKey('defer_loading'), isFalse);
      final text = blocks[0]['text'] as String;
      expect(text, contains('updated guidance'));
      expect(text, contains('<rules>\nnew rules\n</rules>'));
      expect(text, contains('Removed system prompt section "docs"'));

      // The placeholder is declared before any change so its scaffolding is cached from the first request.
      final initial = await capture(native, context.sublist(0, 2));
      expect(tools(initial).map((t) => t['name']), ['base_tool', '__pi_deferred_placeholder__']);
    });

    test('redefines an Anthropic tool by value under the same name', () async {
      final payload = await capture(native, [
        system('base prompt', added: [baseTool]),
        user('before'),
        system('', removed: ['base_tool'], added: [tool('base_tool', 'changed')], timestamp: 2),
      ]);
      expect(betas(payload), contains('inline-tools-2026-09-15'));

      // The top-level declaration keeps the original definition, so the cached prefix survives.
      expect(tools(payload).map((t) => (t['name'], t['description'])), [
        ('base_tool', 'base_tool tool'),
        ('__pi_deferred_placeholder__', 'Reserved placeholder. Never available. Never call this.'),
      ]);

      // The new definition replaces the old one in place; no removal is sent for it.
      final blocks = content(messages(payload).last);
      expect(blocks, hasLength(1));
      expect(blocks.single['type'], 'tool_addition');
      final definition = (blocks.single['tool'] as Json)['definition'] as Json;
      expect((definition['name'], definition['description']), ('base_tool', 'changed'));
    });

    test('sends the current Anthropic tool list without an initial tool', () async {
      // Anthropic rejects a tool list where every tool (here: the placeholder) is deferred.
      final payload = await capture(native, [
        system('base prompt'),
        user('before'),
        system('updated guidance', added: [lateTool], timestamp: 2),
      ]);
      expect(betas(payload), isNot(contains('inline-tools-2026-09-15')));
      expect(tools(payload).map((t) => t['name']), ['late_tool']);
      expect(tools(payload).single['cache_control'], {'type': 'ephemeral'});
      expect(tools(payload).single.containsKey('defer_loading'), isFalse);
      expect(content(messages(payload).last).map((b) => b['type']), ['text']);
    });

    // pi #11091
    test('holds an Anthropic update past an assistant message that converts to no blocks', () async {
      AssistantMessage reply(String text, String provider, String api, String model) {
        return AssistantMessage(
          content: [TextContent(text)],
          api: api,
          provider: provider,
          model: model,
          stopReason: StopReason.stop,
          timestamp: 3,
        );
      }

      final transcript = <Message>[
        system('base prompt'),
        user('hi'),
        system('updated guidance', timestamp: 2),

        // Another model's empty reply: Anthropic rejects empty text blocks, so it is dropped.
        reply('', 'openai-codex', 'openai-codex-responses', 'gpt-x'),
        user('again', timestamp: 4),
      ];
      List<Object?> roles(Json payload) => [for (final m in messages(payload)) m['role']];

      // No later reply: the update goes at the end.
      expect(roles(await capture(native, transcript)), ['user', 'user', 'system']);

      // A later reply that is sent: the update goes directly before it.
      final before = await capture(native, [
        ...transcript,
        reply('answer', native.provider, native.api, native.id),
        user('third', timestamp: 5),
      ]);
      expect(roles(before), ['user', 'user', 'system', 'assistant', 'user']);
    });

    test('folds Anthropic updates into the system prompt without native support', () async {
      final payload = await capture(claude(id: 'claude-sonnet-4-5'), context);
      expect(betas(payload), isNot(contains('inline-tools-2026-09-15')));
      expect(
        [for (final b in payload['system'] as List<Object?>) (b as Json)['text']],
        ['base prompt\n\nupdated guidance\n\n<rules>\nnew rules\n</rules>'],
      );
      expect(tools(payload).map((t) => t['name']), ['late_tool']);
      expect(messages(payload).map((m) => m['role']), ['user']);
    });

    test('requires both Anthropic capabilities for native tool changes', () async {
      final payload = await capture(
        claude(id: 'claude-opus-5', compat: const {'supportsMidConvoToolChanges': true}),
        context,
      );
      expect(betas(payload), isNot(contains('inline-tools-2026-09-15')));
      expect(tools(payload).map((t) => t['name']), ['late_tool']);
      expect(messages(payload).map((m) => m['role']), ['user']);
    });
  });

  test('sends system messages in place but tool changes in the tool list without native tool changes', () async {
    final payload = await capture(
      claude(id: 'claude-opus-5', compat: const {'supportsMidConvoSystemMessages': true}),
      context,
    );
    expect(betas(payload), isNot(contains('inline-tools-2026-09-15')));
    expect(tools(payload).map((t) => t['name']), ['late_tool']);
    expect(messages(payload).map((m) => m['role']), ['user', 'system']);
    expect(content(messages(payload).last).map((b) => b['type']), ['text']);
  });

  test('renders a later system message with its section changes framed by name', () {
    expect(
      renderSystemMessageUpdate(system('also do this', sections: const {'a': '<a>2</a>', 'b': null})),
      'also do this\n\nUpdated system prompt section "a":\n\n<a>2</a>\n\nRemoved system prompt section "b".',
    );
    expect(
      renderSystemMessageUpdate(system('', sections: const {'c': '<c>1</c>'})),
      'Updated system prompt section "c":\n\n<c>1</c>',
    );
  });

  test('downgrades inline additions and same-name redefinitions without downgrading require', () async {
    final grammarError = jsonEncode({
      'type': 'error',
      'error': {'type': 'invalid_request_error', 'message': 'The compiled grammar is too large'},
    });
    Tool strictTool(Strictness strict) {
      return Tool('lookup', 'Look up a value', const {
        'type': 'object',
        'properties': {
          'value': {'type': 'string'},
          'optional': {'type': 'number'},
        },
        'required': ['value'],
      }, constrainedSampling: JsonSchemaSampling(strict));
    }

    final server = FakeServer.sequence([(400, grammarError), (200, sse(minimalEvents()))]);
    final model = claude(
      id: 'claude-haiku-4-5',
      reasoning: false,
      compat: const {
        'supportsStrictTools': true,
        'supportsMidConvoSystemMessages': true,
        'supportsMidConvoToolChanges': true,
      },
    );
    final result = await streamAnthropic(model, [
      SystemMessage(const [], added: [strictTool(Strictness.require)], timestamp: 0),
      user('Hello'),
      SystemMessage(const [], added: [strictTool(Strictness.prefer)], timestamp: 2),
      SystemMessage(const [], added: [strictTool(Strictness.require)], timestamp: 3),
    ], AnthropicOptions(client: server.client(), maxRetries: 0)).result;
    expect(result.stopReason, StopReason.stop);
    expect(server.requests, hasLength(2));
    final bodies = [for (final r in server.requests) jsonDecode(r.body) as Json];
    expect(jsonEncode(bodies[1]['tools']), jsonEncode(bodies[0]['tools']));
    List<Json> additions(Json body) => [
      for (final m in (body['messages'] as List<Object?>).cast<Json>())
        for (final b in (m['content'] as List<Object?>).cast<Json>())
          if (b['type'] == 'tool_addition') (b['tool'] as Json)['definition'] as Json,
    ];
    expect(additions(bodies[0]).map((d) => d['strict']), [true, true]);
    expect(additions(bodies[1]).map((d) => d['strict']), [null, true]);
    expect((additions(bodies[1])[0]['input_schema'] as Json)['required'], ['value']);
    expect(jsonEncode(additions(bodies[1])[1]), jsonEncode(additions(bodies[0])[1]));
  });
}
