// Context overflow: whether a response failed (or was cut) because its input exceeded the model's context window or
// the provider's request size limit. Ported from pi's utils/overflow.ts, which lists an example message per provider.

import 'types.dart';

RegExp _i(String pattern) => RegExp(pattern, caseSensitive: false);

/// Error messages of context and request-size overflow, by provider (pi's comments name each).
final overflowPatterns = [
  _i(r'prompt (?:is )?too long'), // Anthropic and z.ai token overflow
  _i(r'prompt exceeds max length'), // z.ai CN endpoint
  _i(r'request_too_large'), // Anthropic request size overflow (HTTP 413)
  _i(r'input is too long for requested model'), // Amazon Bedrock
  _i(r'exceeds the context window'), // OpenAI (Completions and Responses)
  _i(r"exceeds (?:the )?(?:model'?s )?maximum context length(?: of [\d,]+ tokens?|\s*\([\d,]+\))"), // LiteLLM
  _i(r'context window exceeded'), // LiteLLM pre-call check
  _i(r'input token count.*exceeds the maximum'), // Google (Gemini)
  _i(r'maximum prompt length is \d+'), // xAI (Grok)
  _i(r'reduce the length of the messages'), // Groq
  _i(r'maximum context length is \d+ tokens'), // OpenRouter (most backends)
  _i(r'exceeds (?:the )?maximum allowed input length of [\d,]+ tokens?'), // OpenRouter/Poolside
  _i(r"input \(\d+ tokens\) is longer than the model'?s context length \(\d+ tokens\)"), // Together AI
  _i(r'exceeds the limit of \d+'), // GitHub Copilot
  _i(r'exceeds the available context size'), // llama.cpp server
  _i(r'greater than the context length'), // LM Studio
  _i(r'context window exceeds limit'), // MiniMax
  _i(r'exceeded model token limit'), // Kimi For Coding
  _i(r'too large for model with \d+ maximum context length'), // Mistral
  _i(r'prompt has [\d,]+ tokens?, but the configured context size is [\d,]+ tokens?'), // DS4 server
  _i(r'model_context_window_exceeded'), // z.ai finish reason surfaced as error text
  _i(r'prompt too long; exceeded (?:max )?context length'), // Ollama
  _i(r'range of input length should be'), // DashScope / Qwen
  _i(r'context[_ ]length[_ ]exceeded'), // generic
  _i(r'too many tokens'), // generic
  _i(r'token limit exceeded'), // generic
  // HTTP 413 is request size overflow whatever the provider or body: a raw status, or a provider error's prefix, never
  // a status mentioned in the body.
  _i(r'^(?:[\w -]+ API error \(413\):|413(?:\s|:|$))'),
];

final _cerebrasBodyless = _i(r'^4(?:00|13)\s*(?:status code)?\s*\(no body\)');

/// Errors that are not overflow though they may match a pattern, such as Bedrock's throttling "Too many tokens".
final _notOverflow = [_i(r'^(Throttling error|Service unavailable):'), _i(r'rate limit'), _i(r'too many requests')];

/// Whether [message] failed for context overflow: an error matching an overflow pattern; or, given the model's
/// [contextWindow], a successful response whose input exceeds it (z.ai can overflow silently), or a length stop with
/// no output and the window filled (Xiaomi MiMo truncates the input to fit).
bool isContextOverflow(AssistantMessage message, [int? contextWindow]) {
  final error = message.errorMessage;
  if (message.stopReason == StopReason.error && error != null && !_notOverflow.any((p) => p.hasMatch(error))) {
    if (overflowPatterns.any((p) => p.hasMatch(error))) {
      return true;
    }

    if (message.provider == 'cerebras' && _cerebrasBodyless.hasMatch(error)) {
      return true;
    }
  }

  if (contextWindow == null || contextWindow == 0) {
    return false;
  }

  final input = message.usage.input + message.usage.cacheRead;
  if (message.stopReason == StopReason.stop) {
    return input > contextWindow;
  }

  return message.stopReason == StopReason.length && message.usage.output == 0 && input >= contextWindow * 0.99;
}

/// Whether a length stop ended below [desiredMaxOutput], the output limit before any clamping to the context: it may
/// come from context pressure or truncation, so one compact-and-retry attempt is worth making.
bool isRecoverableLength(AssistantMessage message, int desiredMaxOutput) {
  return message.stopReason == StopReason.length && desiredMaxOutput > 0 && message.usage.output < desiredMaxOutput;
}
