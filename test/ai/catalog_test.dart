// The generated catalog (lib/src/ai/models.generated.dart) against the vendored JSON it is generated from
// (catalog/*.json, pi's model catalog): every entry, every field.

import 'dart:convert';
import 'dart:io';

import 'package:chorus/chorus.dart';
import 'package:test/test.dart';

/// A model's fields as JSON, to compare two models.
Json fields(Model m) {
  Json rates(Rates r) => {'input': r.input, 'output': r.output, 'cacheRead': r.cacheRead, 'cacheWrite': r.cacheWrite};

  return {
    'id': m.id,
    'name': m.name,
    'api': m.api,
    'provider': m.provider,
    'baseUrl': m.baseUrl,
    'input': m.input,
    'cost': {
      ...rates(m.cost),
      'tiers': [
        for (final t in m.cost.tiers) {'above': t.above, ...rates(t)},
      ],
    },
    'headers': m.headers,
    'reasoning': m.reasoning,
    'levels': m.levels,
    'contextWindow': m.contextWindow,
    'maxTokens': m.maxTokens,
    'compat': m.compat,
  };
}

void main() {
  for (final provider in ['anthropic', 'openai', 'google']) {
    test('$provider: the generated models are the catalog\'s', () {
      final catalog = jsonDecode(File('catalog/$provider.json').readAsStringSync()) as Json;
      final generated = builtinModels[provider]!;
      expect(generated.keys, catalog.keys);
      for (final MapEntry(:key, :value) in catalog.entries) {
        expect(jsonEncode(fields(generated[key]!)), jsonEncode(fields(Model.fromJson(value as Json))), reason: key);
      }
    });
  }

  test('every model belongs to its provider and is served on its API', () {
    const apis = {'anthropic': 'anthropic-messages', 'openai': 'openai-responses', 'google': 'google-generative-ai'};
    for (final MapEntry(:key, :value) in builtinModels.entries) {
      for (final model in value.values) {
        expect((model.provider, model.api), (key, apis[key]), reason: model.id);
      }
    }
  });
}
