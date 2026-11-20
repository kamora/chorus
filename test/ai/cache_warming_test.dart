// The cache warming policy. Ported from the policy assertions of pi's coding-agent cache-warmer.test.ts (pi's library
// has the policy, its CLI the scheduling), without the PI_CACHE_RETENTION environment variable, which chorus does not
// read.

import 'package:chorus/chorus.dart';
import 'package:test/test.dart';

/// [model] with other prompt cache lifetimes, or none.
Model withCache(Model model, PromptCache? cache) {
  return Model(
    id: model.id,
    name: model.name,
    api: model.api,
    provider: model.provider,
    baseUrl: model.baseUrl,
    input: model.input,
    cost: model.cost,
    reasoning: model.reasoning,
    levels: model.levels,
    contextWindow: model.contextWindow,
    maxTokens: model.maxTokens,
    compat: model.compat,
    promptCache: cache,
  );
}

final adaptive = withCache(getBuiltinModel('anthropic', 'claude-opus-4-6')!, const PromptCache(short: 300, long: 3600));
final budget = withCache(getBuiltinModel('anthropic', 'claude-sonnet-4-5')!, const PromptCache(short: 300, long: 3600));
final openai = withCache(getBuiltinModel('openai', 'gpt-5')!, const PromptCache(short: 300, long: 86400));
final unknown = withCache(adaptive, null);

void main() {
  test('derives eligibility and timing from retention and provider behavior', () {
    expect(
      [
        promptCacheTtl(adaptive),
        promptCacheTtl(adaptive, const StreamOptions(cacheRetention: CacheRetention.long)),
        promptCacheTtl(adaptive, const StreamOptions(cacheRetention: CacheRetention.none)),
        promptCacheTtl(openai, const StreamOptions(cacheRetention: CacheRetention.long)),
        promptCacheTtl(unknown),
      ],
      [const Duration(seconds: 300), const Duration(seconds: 3600), null, const Duration(seconds: 86400), null],
    );
    expect(
      [
        cacheWarmingDelay(const Duration(seconds: 300)),
        cacheWarmingDelay(const Duration(seconds: 60)),
        cacheWarmingDelay(const Duration(seconds: 10)),
      ],
      [const Duration(seconds: 270), const Duration(seconds: 50), null],
    );
    const medium = SimpleStreamOptions(reasoning: ThinkingLevel.medium);
    expect(
      [
        isCacheWarmingReplayable(budget, medium),
        isCacheWarmingReplayable(budget),
        isCacheWarmingReplayable(adaptive, medium),
        isCacheWarmingReplayable(openai, medium),
      ],
      [false, true, true, true],
    );
  });

  test('uses the catalog\'s lifetimes for direct Anthropic', () {
    expect(promptCacheTtl(getBuiltinModel('anthropic', 'claude-haiku-5-5')!), const Duration(seconds: 300));
  });

  test('warms a large idle prompt and stops for a smaller one or an unknown size', () {
    // Idle, a refresh pays only when 15% of the miss cost exceeds its own cost by $0.05: 100,000 tokens of Claude Opus
    // 4.6 save 0.15 * $0.575 - $0.050025, less than that; 400,000 tokens do.
    expect(evaluateCacheWarming(adaptive, 100000, CacheWarmingPhase.idle).action, CacheWarmingAction.stop);
    final large = evaluateCacheWarming(adaptive, 400000, CacheWarmingPhase.idle);
    expect(large.action, CacheWarmingAction.warm);
    expect(large.economicsAvailable, isTrue);
    expect(large.continuationProbability, cacheWarmingIdleContinuation);
    expect(large.expectedSavings, closeTo(large.continuationProbability * large.missCost - large.warmCost, 1e-12));

    final small = evaluateCacheWarming(adaptive, 5000, CacheWarmingPhase.idle);
    expect((small.action, small.economicsAvailable), (CacheWarmingAction.stop, true));

    final none = evaluateCacheWarming(adaptive, 0, CacheWarmingPhase.idle);
    expect((none.action, none.economicsAvailable), (CacheWarmingAction.stop, false));
  });

  test('prices a refresh as a cache read plus one output token, and a miss as a cache write', () {
    // Claude Opus 4.6: $5 input, $25 output, $0.5 cache read, $6.25 cache write per million tokens.
    final model = getBuiltinModel('anthropic', 'claude-opus-4-6')!;
    expect((model.cost.input, model.cost.output, model.cost.cacheRead, model.cost.cacheWrite), (5, 25, 0.5, 6.25));
    final decision = evaluateCacheWarming(model, 100000, CacheWarmingPhase.streaming);
    expect(decision.warmCost, closeTo(0.05 + 0.000025, 1e-12));
    expect(decision.missCost, closeTo(0.625 - 0.05, 1e-12));
    expect(decision.continuationProbability, 1);
    expect(decision.action, CacheWarmingAction.warm);
  });

  test('prices a miss as input on a model without cache write prices', () {
    final decision = evaluateCacheWarming(openai, 100000, CacheWarmingPhase.streaming);
    final input = openai.cost.input * 100000 / 1e6;
    final read = openai.cost.cacheRead * 100000 / 1e6;
    expect(openai.cost.cacheWrite, 0);
    expect(decision.missCost, closeTo(input - read, 1e-12));
  });

  test('replays with one output token, no retries and no continuation', () {
    final signal = AbortSignal();
    final options = cacheWarmingOptions(
      const SimpleStreamOptions(
        apiKey: 'k',
        maxTokens: 4000,
        maxRetries: 3,
        sessionId: 's',
        reasoning: ThinkingLevel.low,
      ),
      signal,
    );
    expect((options.maxTokens, options.maxRetries, options.disableContinuation), (1, 0, true));
    expect((options.apiKey, options.sessionId, options.reasoning), ('k', 's', ThinkingLevel.low));
    expect(options.signal, same(signal));
  });
}
