// Effort changes mid-conversation. Ported from pi's anthropic-mid-conversation-effort.test.ts; its OpenRouter checks
// are left out (chorus vendors the catalog of Anthropic, OpenAI and Google only).

import 'dart:convert';

import 'package:chorus/chorus.dart';
import 'package:test/test.dart';

import 'anthropic_fake.dart';

Model managed({
  Json compat = const {'forceAdaptiveThinking': true, 'supportsMidConvoEffort': true},
  Map<String, String?>? levels,
}) {
  return claude(
    id: 'claude-fable-5-1',
    levels:
        levels ?? const {'off': null, 'minimal': 'low', 'low': 'low', 'medium': 'medium', 'high': 'high', 'max': 'max'},
    compat: compat,
  );
}

AssistantMessage reply(Model model, [String? level]) {
  return AssistantMessage(
    content: [
      ThinkingContent('reasoning', signature: 'signature'),
      TextContent('answer'),
    ],
    api: 'anthropic-messages',
    provider: model.provider,
    model: model.id,
    providerThinkingLevel: level,
    stopReason: StopReason.stop,
    timestamp: 1,
  );
}

/// The request body and the response of one request, stopped once its payload is captured.
Future<({Json payload, AssistantMessage message})> capture(
  Model model,
  List<Message> messages, {
  String? effort,
  bool thinking = true,
}) async {
  Json? payload;
  final message = await streamAnthropic(
    model,
    messages,
    AnthropicOptions(
      apiKey: 'test-key',
      cacheRetention: CacheRetention.none,
      thinkingEnabled: thinking,
      effort: effort,
      onPayload: (body, _) {
        payload = jsonDecode(jsonEncode(body)) as Json;
        throw StateError('payload captured');
      },
    ),
  ).result;
  return (payload: payload!, message: message);
}

List<Object?> markers(Json payload) => [
  for (final m in payload['messages'] as List<Object?>)
    if ((m as Json)['role'] == 'system') m,
];

Json marker(String effort) {
  return {
    'role': 'system',
    'content': <Object?>[],
    'output_config': {'effort': effort},
  };
}

Json userWire(String text) {
  return {
    'role': 'user',
    'content': [
      {'type': 'text', 'text': text},
    ],
  };
}

void main() {
  group('Anthropic mid-conversation effort', () {
    test('reconstructs an exact historical marker prefix and appends the current marker', () async {
      final model = managed();
      final first = await capture(model, [user('one')], effort: 'low');
      final second = await capture(model, [
        user('one'),
        reply(model, 'low'),
        user('two', timestamp: 2),
      ], effort: 'high');

      expect(first.payload['messages'], [userWire('one'), marker('low')]);
      final messages = second.payload['messages'] as List<Object?>;
      expect(messages.sublist(0, 2), first.payload['messages']);
      expect(messages.last, marker('high'));
      expect(first.payload['output_config'], {'effort': 'low'});
      expect(second.payload['output_config'], {'effort': 'low'});
      expect(second.payload['thinking'], {
        'type': 'adaptive',
        'display': 'summarized',
        'block_binding': {'prefix_mismatch_behavior': 'drop_block'},
      });
      expect(first.message.providerThinkingLevel, 'low');
    });

    for (final effort in ['low', 'medium', 'high', 'xhigh', 'max']) {
      test('preserves native effort $effort', () async {
        final (:payload, :message) = await capture(managed(), [user('one')], effort: effort);
        expect(markers(payload), [marker(effort)]);
        expect(message.providerThinkingLevel, effort);
      });
    }

    test('defaults omitted effort to high and still enables drop_block', () async {
      final (:payload, :message) = await capture(managed(), [user('one')]);
      expect((payload['messages'] as List<Object?>).last, marker('high'));
      expect(((payload['thinking'] as Json)['block_binding'] as Json)['prefix_mismatch_behavior'], 'drop_block');
      expect(message.providerThinkingLevel, 'high');
    });

    test('does not invent markers for legacy or other-provider assistants', () async {
      final model = managed();
      final other = AssistantMessage(
        content: [TextContent('answer')],
        api: 'anthropic-messages',
        provider: 'other-provider',
        model: model.id,
        providerThinkingLevel: 'low',
        stopReason: StopReason.stop,
        timestamp: 1,
      );
      final (:payload, message: _) = await capture(model, [
        user('one'),
        reply(model),
        user('two', timestamp: 2),
        other,
        user('three', timestamp: 3),
      ], effort: 'medium');
      expect(markers(payload), [marker('medium')]);
    });

    test('sends plain disabled thinking without effort when off is supported', () async {
      // Anthropic rejects block_binding with disabled thinking, and effort markers that change level while thinking is
      // disabled.
      final model = managed(
        levels: const {'minimal': null, 'low': 'low', 'medium': 'medium', 'high': 'high', 'max': 'max'},
      );
      final (:payload, :message) = await capture(model, [
        user('one'),
        reply(model, 'low'),
        user('two', timestamp: 2),
        reply(model, 'medium'),
        user('three', timestamp: 3),
      ], thinking: false);
      expect(payload['thinking'], {'type': 'disabled'});
      expect(payload.containsKey('output_config'), isFalse);
      expect(markers(payload), isEmpty);
      expect(message.providerThinkingLevel, isNull);
    });

    test('keeps adaptive thinking when off is unsupported', () async {
      final (:payload, :message) = await capture(managed(), [user('one')], thinking: false);
      expect((payload['thinking'] as Json)['type'], 'adaptive');
      expect(markers(payload), [marker('high')]);
      expect(message.providerThinkingLevel, 'high');
    });

    test('leaves unsupported models on top-level effort', () async {
      final (:payload, :message) = await capture(managed(compat: const {'forceAdaptiveThinking': true}), [
        user('one'),
      ], effort: 'low');
      expect(payload['messages'], [userWire('one')]);
      expect(payload['output_config'], {'effort': 'low'});
      expect(payload['thinking'], {'type': 'adaptive', 'display': 'summarized'});
      expect(message.providerThinkingLevel, isNull);
    });

    test('sends the effort and binding beta headers', () async {
      final server = FakeServer(sse(minimalEvents()));
      final result = await streamAnthropic(managed(), [
        user('one'),
      ], AnthropicOptions(client: server.client(), cacheRetention: CacheRetention.none)).result;
      expect(result.stopReason, StopReason.stop);
      final beta = server.requests.single.headers['anthropic-beta'];
      expect(beta, contains('mid-conversation-output-config-2026-07-01'));
      expect(beta, contains('thinking-binding-controls-2026-08-01'));
    });

    test('generates exact model gates', () {
      final direct = getBuiltinModel('anthropic', 'claude-fable-5-1')!;
      expect(direct.compat['supportsMidConvoEffort'], isTrue);
      expect(direct.levels, containsPair('off', null));
      expect(getBuiltinModel('anthropic', 'claude-opus-4-8')!.compat.containsKey('supportsMidConvoEffort'), isFalse);
      expect(getBuiltinModel('anthropic', 'claude-opus-5')!.compat.containsKey('allowedFallbackModels'), isFalse);
    });

    test('supports off only on managed models that accept disabled thinking', () {
      for (final id in ['claude-opus-5', 'claude-haiku-5-5']) {
        final model = getBuiltinModel('anthropic', id)!;
        expect(model.compat['supportsMidConvoEffort'], isTrue, reason: id);
        expect(model.levels, isNot(containsPair('off', null)), reason: id);
      }

      for (final id in ['claude-opus-5-5', 'claude-sonnet-5-5', 'claude-fable-5-1', 'claude-fable-5-2']) {
        expect(getBuiltinModel('anthropic', id)!.levels, containsPair('off', null), reason: id);
      }
    });

    test('gates Fable 5.2 like Fable 5.1 and keeps only fallbacks that accept per-message effort', () {
      final direct = getBuiltinModel('anthropic', 'claude-fable-5-2')!;
      expect(direct.compat['supportsMidConvoEffort'], isTrue);
      expect(direct.compat['supportsMidConvoSystemMessages'], isTrue);
      expect(direct.levels, allOf(containsPair('xhigh', 'xhigh'), containsPair('max', 'max')));
      expect(
        [for (final f in direct.compat['allowedFallbackModels'] as List<Object?>) (f as Json)['model']],
        ['claude-opus-5'],
      );
    });
  });
}
