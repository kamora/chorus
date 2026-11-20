// Ported from pi's packages/ai/test: models-runtime.test.ts (the tests without auth and dynamic model lists),
// providers.test.ts (builtin providers, createProvider's dispatch), supports-xhigh.test.ts and max-thinking.test.ts
// (for the catalog's Anthropic and OpenAI models).

import 'package:chorus/chorus.dart';
import 'package:test/test.dart';

Model testModel(String provider, String id, {String api = 'test-api'}) {
  return Model(
    id: id,
    name: id,
    api: api,
    provider: provider,
    baseUrl: 'https://example.test/v1',
    cost: const ModelCost(input: 0, output: 0, cacheRead: 0, cacheWrite: 0),
    contextWindow: 10000,
    maxTokens: 1000,
  );
}

/// Streams that answer "ok" and record each call as `label:model`.
ProviderStreams recording(String label, List<String> calls) {
  AssistantMessageEventStream respond(Model model) {
    calls.add('$label:${model.id}');
    final message = AssistantMessage(
      content: [TextContent('ok')],
      api: model.api,
      provider: model.provider,
      model: model.id,
      stopReason: StopReason.stop,
      timestamp: 0,
    );
    return AssistantMessageEventStream()
      ..push(StartEvent(message))
      ..push(DoneEvent(StopReason.stop, message));
  }

  return ProviderStreams(stream: (model, _, _) => respond(model), streamSimple: (model, _, _) => respond(model));
}

Provider testProvider(String id, {List<Model>? models, List<String>? calls}) {
  return Provider(id: id, models: models ?? [testModel(id, 'model-a')], apis: {'test-api': recording(id, calls ?? [])});
}

final messages = <Message>[
  UserMessage([TextContent('hi')], timestamp: 0),
];

void main() {
  group('Models', () {
    test('registers, replaces, and deletes providers', () {
      final models = Models()
        ..setProvider(testProvider('p1'))
        ..setProvider(testProvider('p2'));
      expect(models.getProviders().map((p) => p.id), ['p1', 'p2']);

      final replacement = testProvider('p1');
      models.setProvider(replacement);
      expect(models.getProvider('p1'), same(replacement));
      expect(models.getProviders(), hasLength(2));

      models.deleteProvider('p1');
      expect(models.getProvider('p1'), isNull);

      models.clearProviders();
      expect(models.getProviders(), isEmpty);
    });

    test('lists and finds models per provider', () {
      final models = Models()
        ..setProvider(testProvider('p1', models: [testModel('p1', 'm1'), testModel('p1', 'm2')]))
        ..setProvider(testProvider('p2', models: [testModel('p2', 'm3')]));
      expect(models.getModels().map((m) => m.id), ['m1', 'm2', 'm3']);
      expect(models.getModels('p1').map((m) => m.id), ['m1', 'm2']);
      expect(models.getModels('nope'), isEmpty);
      expect(models.getModel('p2', 'm3')?.id, 'm3');
      expect(models.getModel('p2', 'missing'), isNull);
    });

    test('produces an error stream for unknown providers instead of throwing', () async {
      final result = await Models().completeSimple(testModel('ghost', 'model-a'), messages);
      expect(result.stopReason, StopReason.error);
      expect(result.errorMessage, contains('Unknown provider: ghost'));
    });

    test('ends an unknown provider\'s stream as aborted once the signal fired', () async {
      final signal = AbortSignal()..abort();
      final result = await Models().complete(testModel('ghost', 'model-a'), messages, StreamOptions(signal: signal));
      expect(result.stopReason, StopReason.aborted);
    });

    test('streams through the provider', () async {
      final calls = <String>[];
      final models = Models()..setProvider(testProvider('p1', calls: calls));
      final stream = models.streamSimple(testModel('p1', 'model-a'), messages);
      expect(await stream.map((e) => e.runtimeType).toList(), [StartEvent, DoneEvent]);
      expect((await stream.result).stopReason, StopReason.stop);
      expect(await models.complete(testModel('p1', 'model-a'), messages), isA<AssistantMessage>());
      expect(calls, ['p1:model-a', 'p1:model-a']);
    });
  });

  group('Provider', () {
    test('dispatches on model.api for mixed-API providers', () async {
      final calls = <String>[];
      final models = Models()
        ..setProvider(
          Provider(
            id: 'mixed',
            models: [
              testModel('mixed', 'model-a', api: 'api-a'),
              testModel('mixed', 'model-b', api: 'api-b'),
            ],
            apis: {'api-a': recording('a', calls), 'api-b': recording('b', calls)},
          ),
        );
      await models.completeSimple(testModel('mixed', 'model-a', api: 'api-a'), messages);
      await models.completeSimple(testModel('mixed', 'model-b', api: 'api-b'), messages);
      expect(calls, ['a:model-a', 'b:model-b']);
    });

    test('produces a stream error for a model whose api has no implementation', () async {
      final provider = Provider(
        id: 'mixed',
        models: [testModel('mixed', 'model-a', api: 'api-a')],
        apis: {'api-a': recording('a', [])},
      );
      final result = await provider.streamSimple(testModel('mixed', 'model-x', api: 'api-ghost'), messages).result;
      expect(result.stopReason, StopReason.error);
      expect(result.errorMessage, contains('no API implementation'));
    });
  });

  group('builtin providers', () {
    test('builtinModels registers every builtin provider with models', () {
      final models = builtinModels();
      expect(models.getProviders().map((p) => p.id), builtinProviders().map((p) => p.id));
      expect(models.getModel('anthropic', 'claude-haiku-4-5')?.api, 'anthropic-messages');
      for (final provider in models.getProviders()) {
        expect(provider.models, isNotEmpty);
        expect(provider.models.every((m) => m.provider == provider.id), isTrue);
      }
    });

    test('returns empty results for unknown provider ids', () {
      expect(getBuiltinModel('not-a-provider', 'x'), isNull);
      expect(getBuiltinModels('not-a-provider'), isEmpty);
      expect(getBuiltinProviders(), containsAll(['anthropic', 'openai', 'google']));
    });

    test('streams the Anthropic provider with its own options, built from neutral ones', () async {
      final model = getBuiltinModel('anthropic', 'claude-haiku-5-5')!;
      Json? payload;
      final options = StreamOptions(
        apiKey: 'test',
        onPayload: (body, _) {
          payload = body;
          throw StateError('payload captured');
        },
      );
      await builtinModels().complete(model, messages, options);
      expect(payload?['model'], 'claude-haiku-5-5');
    });

    test('compares models by id and provider', () {
      expect(modelsAreEqual(testModel('p', 'a'), testModel('p', 'a')), isTrue);
      expect(modelsAreEqual(testModel('p', 'a'), testModel('q', 'a')), isFalse);
      expect(modelsAreEqual(testModel('p', 'a'), null), isFalse);
    });
  });

  group('getSupportedThinkingLevels', () {
    List<String> levels(String provider, String id) {
      return getSupportedThinkingLevels(getBuiltinModel(provider, id)!).map((l) => l.name).toList();
    }

    test('includes max but not xhigh for Anthropic Opus 4.6 and Sonnet 4.6', () {
      for (final id in ['claude-opus-4-6', 'claude-sonnet-4-6']) {
        expect(levels('anthropic', id), allOf(contains('max'), isNot(contains('xhigh'))), reason: id);
      }
    });

    test('includes xhigh and max for Anthropic Opus 4.8, Opus 5 and Sonnet 5', () {
      for (final id in ['claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5']) {
        expect(levels('anthropic', id), containsAll(['xhigh', 'max']), reason: id);
      }
    });

    test('includes Claude Opus 5.5 with its always-on effort levels and official pricing', () {
      final model = getBuiltinModel('anthropic', 'claude-opus-5-5')!;
      expect((model.cost.input, model.cost.output, model.cost.cacheRead, model.cost.cacheWrite), (4, 20, 0.2, 5));
      expect((model.contextWindow, model.maxTokens), (1000000, 128000));
      expect(model.compat, containsPair('forceAdaptiveThinking', true));
      expect(model.compat, containsPair('supportsMidConvoEffort', true));
      expect(levels('anthropic', 'claude-opus-5-5'), ['low', 'medium', 'high', 'xhigh', 'max']);
    });

    test('includes Claude Sonnet 5.5 with managed effort levels and official pricing', () {
      final model = getBuiltinModel('anthropic', 'claude-sonnet-5-5')!;
      expect((model.cost.input, model.cost.output, model.cost.cacheRead, model.cost.cacheWrite), (2, 10, 0.1, 2.5));
      expect(model.compat, containsPair('supportsTemperature', false));
      expect(levels('anthropic', 'claude-sonnet-5-5'), ['low', 'medium', 'high', 'xhigh', 'max']);
    });

    test('includes xhigh and max but not off for Anthropic Claude Fable 5', () {
      expect(levels('anthropic', 'claude-fable-5'), allOf(containsAll(['xhigh', 'max']), isNot(contains('off'))));
    });

    test('does not include xhigh or max for Claude Sonnet 4.5', () {
      expect(levels('anthropic', 'claude-sonnet-4-5'), isNot(anyOf(contains('xhigh'), contains('max'))));
    });

    test('includes xhigh and max for OpenAI GPT-5.6 and GPT-6 models', () {
      for (final id in ['gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-5.6-luna', 'gpt-6-sol', 'gpt-6-luna', 'gpt-6.1-terra']) {
        expect(levels('openai', id), ['off', 'low', 'medium', 'high', 'xhigh', 'max'], reason: id);
      }
    });

    test('does not support off for GPT-6.1 Sol and GPT-6.2', () {
      for (final id in ['gpt-6.1-sol', 'gpt-6.2-astra', 'gpt-6.2-sol']) {
        expect(levels('openai', id), ['low', 'medium', 'high', 'xhigh', 'max'], reason: id);
        expect(getBuiltinModel('openai', id)!.levels, containsPair('off', null));
      }
    });

    test('supports off but not minimal for GPT-6.1 Terra', () {
      final model = getBuiltinModel('openai', 'gpt-6.1-terra')!;
      expect(model.levels, containsPair('off', 'none'));
      expect(model.levels, containsPair('minimal', null));
    });

    test('includes only medium/high/xhigh for OpenAI GPT-5.5 Pro', () {
      expect(levels('openai', 'gpt-5.5-pro'), ['medium', 'high', 'xhigh']);
    });

    test('includes only off for models without reasoning', () {
      expect(getSupportedThinkingLevels(testModel('p', 'a')), [ThinkingLevel.off]);
    });
  });

  group('max thinking level', () {
    Model reasoning(Map<String, String?>? levels) {
      return Model(
        id: 'm',
        name: 'm',
        api: 'openai-completions',
        provider: 'test',
        baseUrl: 'https://example.com/v1',
        reasoning: true,
        levels: levels,
        cost: const ModelCost(input: 0, output: 0, cacheRead: 0, cacheWrite: 0),
        contextWindow: 128000,
        maxTokens: 4096,
      );
    }

    test('is opt-in for ordinary reasoning models', () {
      final model = reasoning(null);
      expect(getSupportedThinkingLevels(model).map((l) => l.name), ['off', 'minimal', 'low', 'medium', 'high']);
      expect(clampThinkingLevel(model, ThinkingLevel.max), ThinkingLevel.high);
    });

    test('supports a hole between high and max', () {
      final model = reasoning({'xhigh': null, 'max': 'max'});
      expect(getSupportedThinkingLevels(model).map((l) => l.name), ['off', 'minimal', 'low', 'medium', 'high', 'max']);
      expect(clampThinkingLevel(model, ThinkingLevel.xhigh), ThinkingLevel.max);
    });

    test('clamps up first, then down', () {
      final model = reasoning({'off': null, 'minimal': null});
      expect(clampThinkingLevel(model, ThinkingLevel.off), ThinkingLevel.low);
      expect(clampThinkingLevel(model, ThinkingLevel.medium), ThinkingLevel.medium);
      expect(clampThinkingLevel(reasoning({'max': null}), ThinkingLevel.max), ThinkingLevel.high);
    });
  });
}
