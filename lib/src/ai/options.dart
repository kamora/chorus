import 'dart:async';
import 'dart:convert';
import 'dart:math';

import 'package:http/http.dart' as http;

import 'model.dart';
import 'transcript.dart';
import 'types.dart';

/// Cancels a request, like the web's `AbortController` and `AbortSignal` in one.
class AbortSignal {
  final _done = Completer<void>();

  bool get aborted => _done.isCompleted;

  /// Completes when [abort] is called.
  Future<void> get future => _done.future;

  void abort() {
    if (!_done.isCompleted) {
      _done.complete();
    }
  }
}

/// The status and headers of a provider's response, for [StreamOptions.onResponse].
class ProviderResponse {
  const ProviderResponse(this.status, this.headers);

  final int status;
  final Map<String, String> headers;
}

/// Options every provider understands, ported from pi's `StreamOptions`.
class StreamOptions {
  const StreamOptions({
    this.signal,
    this.apiKey,
    this.headers,
    this.httpClient,
    this.timeout,
    this.maxRetries,
    this.maxRetryDelay,
    this.temperature,
    this.maxTokens,
    this.cacheRetention,
    this.sessionId,
    this.metadata,
    this.onPayload,
    this.onResponse,
    this.onEvent,
  });

  final AbortSignal? signal;
  final String? apiKey;

  /// Merged over the provider's and model's headers; a `null` value removes a default header.
  final Map<String, String?>? headers;

  /// The HTTP client to send requests with; one is created per request when not set.
  final http.Client? httpClient;

  /// How long to wait for the response headers.
  final Duration? timeout;

  /// Retries of a failed request; none by default.
  final int? maxRetries;

  /// The longest server-requested retry delay to wait; a longer one fails at once. One minute by default, and
  /// [Duration.zero] waits for any delay.
  final Duration? maxRetryDelay;
  final double? temperature;
  final int? maxTokens;

  /// Prompt cache retention; `short` by default.
  final CacheRetention? cacheRetention;

  /// Identifies the conversation, for providers that route or cache by session.
  final String? sessionId;

  /// Request metadata; Anthropic reads `user_id`.
  final Json? metadata;

  /// Sees the request body before it is sent, and may return a replacement.
  final FutureOr<Json?> Function(Json payload, Model model)? onPayload;

  /// Called once the response headers arrive.
  final FutureOr<void> Function(ProviderResponse response, Model model)? onResponse;

  /// Sees each event of the provider's stream, as the provider sent it, before chorus reads it.
  final FutureOr<void> Function(Json event, Model model)? onEvent;
}

/// Token budgets for each thinking level, for models that think within a budget.
class ThinkingBudgets {
  const ThinkingBudgets({this.minimal, this.low, this.medium, this.high});

  final int? minimal;
  final int? low;
  final int? medium;
  final int? high;
}

/// Options for the provider-neutral `streamSimple`, ported from pi's `SimpleStreamOptions`.
class SimpleStreamOptions extends StreamOptions {
  const SimpleStreamOptions({
    super.signal,
    super.apiKey,
    super.headers,
    super.httpClient,
    super.timeout,
    super.maxRetries,
    super.maxRetryDelay,
    super.temperature,
    super.maxTokens,
    super.cacheRetention,
    super.sessionId,
    super.metadata,
    super.onPayload,
    super.onResponse,
    super.onEvent,
    this.reasoning,
    this.budgets,
    this.toolChoice,
  });

  /// The thinking level; `null` or [ThinkingLevel.off] for no thinking.
  final ThinkingLevel? reasoning;
  final ThinkingBudgets? budgets;

  /// `auto` or `none`.
  final String? toolChoice;
}

/// Tokens always left for the answer when a thinking budget shares the response ceiling.
const minAnswerTokens = 1024;

/// The thinking budget of [level], from [custom] or pi's defaults; `xhigh` and `max` use `high`'s.
int thinkingBudget(ThinkingLevel level, [ThinkingBudgets? custom]) {
  return switch (level) {
    ThinkingLevel.off => throw ArgumentError.value(level, 'level', 'No thinking, so no budget'),
    ThinkingLevel.minimal => custom?.minimal ?? 1024,
    ThinkingLevel.low => custom?.low ?? 2048,
    ThinkingLevel.medium => custom?.medium ?? 8192,
    ThinkingLevel.high || ThinkingLevel.xhigh || ThinkingLevel.max => custom?.high ?? 16384,
  };
}

/// The response ceiling and thinking budget for [level]. With no caller cap ([base] `null`) the model's [cap] is the
/// ceiling; otherwise the budget is added to [base], up to [cap]. A budget that fills the ceiling is cut to leave
/// [minAnswerTokens] for the answer.
({int maxTokens, int budget}) adjustMaxTokensForThinking(
  int? base,
  int cap,
  ThinkingLevel level, [
  ThinkingBudgets? custom,
]) {
  var budget = thinkingBudget(level, custom);
  final ceiling = base == null ? cap : min(base + budget, cap);
  if (ceiling <= budget) {
    budget = min(budget, max(0, ceiling - minAnswerTokens));
  }

  return (maxTokens: ceiling, budget: budget);
}

/// Fits [maxTokens] into [model]'s context window, leaving room for the input counted higher than estimated.
///
/// The margin is 4096 tokens, or 15% of the input when that is more, since OpenRouter can count the input that much
/// above the estimate. The 15% is kept only while it leaves 32768 tokens of output.
int clampMaxTokensToContext(Model model, List<Message> messages, int maxTokens) {
  if (model.contextWindow <= 0) {
    return max(1, maxTokens);
  }

  final input = estimateContextTokens(messages);
  int room = model.contextWindow - input - max(4096, (input * 0.15).ceil());
  if (room < 32768) {
    room = min(32768, model.contextWindow - input - 4096);
  }

  return min(maxTokens, max(1, room));
}

/// Estimated context tokens: the last usable assistant usage plus estimates for the messages after it.
int estimateContextTokens(List<Message> messages) {
  int? latest;
  int? last;
  for (var i = 0; i < messages.length; i++) {
    final m = messages[i];
    if (m is AssistantMessage) {
      // A message inserted after the reply (such as a compaction summary) means its usage no longer describes
      // the prefix.
      if (m.timestamp >= (latest ?? m.timestamp) &&
          m.stopReason != StopReason.aborted &&
          m.stopReason != StopReason.error) {
        if (_contextTokens(m.usage) > 0) {
          last = i;
        }
      }
    }

    latest = max(latest ?? m.timestamp, m.timestamp);
  }

  var tokens = last == null ? 0 : _contextTokens((messages[last] as AssistantMessage).usage);
  for (var i = (last ?? -1) + 1; i < messages.length; i++) {
    tokens += estimateMessageTokens(messages[i]);
  }

  return tokens;
}

int _contextTokens(Usage usage) {
  if (usage.totalTokens != 0) {
    return usage.totalTokens;
  }

  return usage.input + usage.output + usage.cacheRead + usage.cacheWrite;
}

/// About 3.5 characters per token, and 4800 characters per image.
int estimateMessageTokens(Message message) {
  int tokens(int chars) => (chars / 3.5).ceil();

  int content(List<Content> blocks) {
    return blocks.fold(0, (sum, b) => sum + (b is TextContent ? b.text.length : 4800));
  }

  return switch (message) {
    SystemMessage() =>
      tokens(systemText(message).length) +
          _tools(message.added?.map((t) => t.toJson()).toList()) +
          _tools(message.removed?.map((n) => {'name': n}).toList()),
    UserMessage() => tokens(content(message.content)),
    ToolResultMessage() => tokens(content(message.content)),
    AssistantMessage() => tokens(
      message.content.fold(0, (sum, b) {
        return sum +
            switch (b) {
              TextContent() => b.text.length,
              ThinkingContent() => b.thinking.length,
              ToolCall() => b.name.length + jsonEncode(b.arguments).length,
              ImageContent() => 0,
            };
      }),
    ),
  };
}

int _tools(List<Object?>? tools) {
  if (tools == null || tools.isEmpty) {
    return 0;
  }

  return (jsonEncode(tools).length / 3.5).ceil();
}
