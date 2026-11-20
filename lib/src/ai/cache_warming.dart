// Prompt cache warming policy: when a refresh is due, whether a request can be replayed, and whether a refresh pays
// for itself. The host owns the scheduling; this owns the numbers. Ported from pi's cache-warming.ts.

import 'model.dart';
import 'options.dart';
import 'types.dart';

/// Whether a refresh is sent, or warming ends until the next real request.
enum CacheWarmingAction { warm, stop }

/// [streaming] while the agent run that sent the request is active; [idle] after it settled.
enum CacheWarmingPhase { streaming, idle }

/// Inputs and outcome of one warm-or-stop decision.
typedef CacheWarmingDecision = ({
  CacheWarmingPhase phase,

  /// Price of this refresh: a cache read of the prompt plus one output token.
  double warmCost,

  /// Extra price of the next real request if the cache entry is lost.
  double missCost,

  /// Estimated chance that a real request arrives before the entry expires.
  double continuationProbability,

  /// `continuationProbability * missCost - warmCost`.
  double expectedSavings,

  /// False when the prompt size or the model's prices are unknown.
  bool economicsAvailable,

  /// [CacheWarmingAction.warm] when [expectedSavings] reaches [cacheWarmingMinimumSavings].
  CacheWarmingAction action,
});

/// Streaming warming never continues past this long after the real request that started it.
const cacheWarmingMaxAge = Duration(minutes: 60);

/// Idle warming uses a shorter horizon: continuation estimates become less reliable with age.
const cacheWarmingMaxIdleAge = Duration(minutes: 30);

/// A refresh is sent only when it is expected to save at least this many dollars.
const cacheWarmingMinimumSavings = 0.05;

/// Chance that a real request arrives before the entry expires while the agent sits idle, measured by pi from its own
/// usage.
const cacheWarmingIdleContinuation = 0.15;

/// When to refresh an entry that lives [ttl]: at 90% of it, keeping at least ten seconds of margin; null for a
/// lifetime of ten seconds or less.
Duration? cacheWarmingDelay(Duration ttl) {
  final ms = ttl.inMilliseconds;
  if (ms <= 10000) {
    return null;
  }

  final delay = ms * 0.9 < ms - 10000 ? (ms * 0.9).floor() : ms - 10000;
  return Duration(milliseconds: delay < 1 ? 1 : delay);
}

/// The lifetime of the cache entry a request writes: the model's [Model.promptCache] for the retention it used
/// (short by default). Null when caching is off or the lifetime is unknown.
Duration? promptCacheTtl(Model model, [StreamOptions? options]) {
  final seconds = switch (options?.cacheRetention ?? CacheRetention.short) {
    CacheRetention.none => null,
    CacheRetention.short => model.promptCache?.short,
    CacheRetention.long => model.promptCache?.long,
  };
  return seconds == null ? null : Duration(seconds: seconds);
}

/// Whether replaying the request with a one-token output cap leaves its cache entry untouched. Anthropic's
/// budget-based thinking derives the thinking budget from the output cap; a replay would get another budget, which
/// Anthropic keys the message cache on, and could still think for thousands of tokens.
bool isCacheWarmingReplayable(Model model, [SimpleStreamOptions? options]) {
  final level = options?.reasoning;
  if (level == null || level == ThinkingLevel.off || model.api != 'anthropic-messages') {
    return true;
  }

  return model.compat['forceAdaptiveThinking'] == true;
}

/// The options of a refresh: [options] with [signal], capped at one output token, without retries, and without
/// continuing the conversation's provider-side state (the refresh repeats the last request).
SimpleStreamOptions cacheWarmingOptions(SimpleStreamOptions options, AbortSignal signal) {
  return SimpleStreamOptions(
    signal: signal,
    apiKey: options.apiKey,
    headers: options.headers,
    httpClient: options.httpClient,
    timeout: options.timeout,
    maxRetries: 0,
    maxRetryDelay: options.maxRetryDelay,
    temperature: options.temperature,
    maxTokens: 1,
    cacheRetention: options.cacheRetention,
    sessionId: options.sessionId,
    metadata: options.metadata,
    onPayload: options.onPayload,
    onResponse: options.onResponse,
    onEvent: options.onEvent,
    disableContinuation: true,
    reasoning: options.reasoning,
    budgets: options.budgets,
    toolChoice: options.toolChoice,
  );
}

/// Whether refreshing the cache entry of a request of [promptTokens] pays for itself in [phase].
CacheWarmingDecision evaluateCacheWarming(Model model, int promptTokens, CacheWarmingPhase phase) {
  double price(Usage usage) => calculateCost(model, usage).total;

  final hit = price(Usage(cacheRead: promptTokens));
  final miss = price(model.cost.cacheWrite > 0 ? Usage(cacheWrite: promptTokens) : Usage(input: promptTokens));
  final warmCost = price(Usage(cacheRead: promptTokens, output: 1));
  final missCost = miss - hit > 0 ? miss - hit : 0.0;
  final continuation = phase == CacheWarmingPhase.idle ? cacheWarmingIdleContinuation : 1.0;
  final savings = continuation * missCost - warmCost;
  return (
    phase: phase,
    warmCost: warmCost,
    missCost: missCost,
    continuationProbability: continuation,
    expectedSavings: savings,
    economicsAvailable: promptTokens > 0 && (hit > 0 || miss > 0),
    action: savings >= cacheWarmingMinimumSavings ? CacheWarmingAction.warm : CacheWarmingAction.stop,
  );
}
