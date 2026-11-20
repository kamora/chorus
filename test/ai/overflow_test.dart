// Ported from pi's packages/ai/test/overflow.test.ts.

import 'package:chorus/chorus.dart';
import 'package:test/test.dart';

AssistantMessage error(String message, [String provider = 'ollama']) {
  return AssistantMessage(
    api: 'openai-completions',
    provider: provider,
    model: 'qwen3.5:35b',
    stopReason: StopReason.error,
    errorMessage: message,
    timestamp: 0,
  );
}

AssistantMessage lengthStop({required int input, required int cacheRead, required int output, int cacheWrite = 0}) {
  return AssistantMessage(
    api: 'openai-completions',
    provider: 'test-provider',
    model: 'test-model',
    usage: Usage(input: input, output: output, cacheRead: cacheRead, cacheWrite: cacheWrite),
    stopReason: StopReason.length,
    timestamp: 0,
  );
}

void main() {
  group('isContextOverflow', () {
    for (final message in [
      'github-copilot API error (413): {"message":"failed to parse request","code":""}',
      'custom-provider API error (413): Payload Too Large',
      '413 {"error":{"message":"failed to parse request","code":""}}',
      '413',
    ]) {
      test('detects provider-independent request-size overflow: $message', () {
        // The request can exceed the byte limit even with zero usage and a large token window.
        expect(isContextOverflow(error(message), 1000000), isTrue);
        expect(isContextOverflow(error(message)), isTrue);
      });
    }

    for (final message in [
      'github-copilot API error (400): {"message":"failed to parse request","code":""}',
      'custom-provider API error (429): Too many requests',
      'github-copilot API error (400): {"message":"failed to parse request: invalid value 413"}',
      'custom-provider API error (400): upstream mentioned OpenAI API error (413): invalid input',
      'custom-provider API error (4130): invalid input',
      '4130 failed to parse request',
    ]) {
      test('does not mistake unrelated errors for request-size overflow: $message', () {
        expect(isContextOverflow(error(message)), isFalse);
      });
    }

    test('limits HTTP-status matching to failed responses', () {
      for (final reason in [StopReason.aborted, StopReason.stop]) {
        expect(isContextOverflow(error('OpenAI API error (413): {"message":"failed"}')..stopReason = reason), isFalse);
      }
    });

    for (final (name, message, window) in [
      (
        'explicit Ollama prompt-too-long errors',
        '400 `prompt too long; exceeded max context length by 100918 tokens`',
        32768,
      ),
      ('z.ai prompt-too-long errors', '400 {"code":"1261","message":"Prompt too long"}', 1048576),
      (
        'z.ai CN endpoint prompt-exceeds-max-length errors',
        '400 {"code":"1261","message":"Prompt exceeds max length"}',
        1048576,
      ),
      (
        'Together AI context length errors',
        "400 The input (516368 tokens) is longer than the model's context length (262144 tokens).",
        262144,
      ),
      (
        'LiteLLM-wrapped OpenAI maximum context length errors',
        'Error: 503 litellm.ServiceUnavailableError: litellm.MidStreamFallbackError: litellm.APIConnectionError: '
            "APIConnectionError: OpenAIException - Requested token count exceeds the model's maximum context length of "
            '131072 tokens.',
        131072,
      ),
      (
        'LiteLLM pre-call ContextWindowExceededError',
        '400: {"message":"litellm.ContextWindowExceededError: litellm.BadRequestError: litellm._pre_call_checks: '
            'Context Window exceeded for given call."}',
        262144,
      ),
      (
        'OpenAI-compatible parenthesized maximum context length errors',
        "Error: 400 Input length (265330) exceeds model's maximum context length (262144).",
        262144,
      ),
      (
        'OpenRouter Poolside maximum allowed input length errors',
        'Provider returned error: Input length 131393 exceeds the maximum allowed input length of 131040 tokens.',
        131072,
      ),
      (
        'DS4 configured context size errors',
        '400 Prompt has 256468 tokens, but the configured context size is 256000 tokens',
        256000,
      ),
      (
        'DS4 errors with thousands separators',
        'Prompt has 5,958,968 tokens, but the configured context size is 256,000 tokens',
        256000,
      ),
      ('Anthropic prompt-too-long errors', 'prompt is too long: 213462 tokens > 200000 maximum', 200000),
      (
        'Anthropic request size errors',
        '413 {"error":{"type":"request_too_large","message":"Request exceeds the maximum size"}}',
        200000,
      ),
    ]) {
      test('detects $name', () {
        expect(isContextOverflow(error(message), window), isTrue);
      });
    }

    test('does not treat generic non-overflow Ollama errors as overflow', () {
      expect(isContextOverflow(error('500 `model runner crashed unexpectedly`'), 32768), isFalse);
    });

    // pi #9482
    test('only treats bodyless 400 errors as overflow for Cerebras', () {
      expect(isContextOverflow(error('400 status code (no body)', 'cerebras'), 131072), isTrue);
      expect(isContextOverflow(error('400 status code (no body)', 'opencode-go'), 1000000), isFalse);
    });

    for (final message in [
      'Throttling error: Too many tokens, please wait before trying again.',
      'Service unavailable: The service is temporarily unavailable.',
      'Rate limit exceeded, please retry after 30 seconds.',
      'Too many requests. Please slow down.',
    ]) {
      test('does not treat "$message" as overflow', () {
        expect(isContextOverflow(error(message), 200000), isFalse);
      });
    }

    test('detects silent overflow: a successful response whose input exceeds the window', () {
      final message = lengthStop(input: 1000, cacheRead: 200000, output: 10)..stopReason = StopReason.stop;
      expect(isContextOverflow(message, 200000), isTrue);
      expect(isContextOverflow(message), isFalse);
    });

    test('detects Xiaomi-style overflow (length stop with zero output and filled context)', () {
      expect(isContextOverflow(lengthStop(input: 58, cacheRead: 1048512, output: 0), 1048576), isTrue);
    });

    test('does not treat normal length stops with output as context overflow', () {
      expect(isContextOverflow(lengthStop(input: 1000, cacheRead: 0, output: 4096), 200000), isFalse);
    });

    test('needs the window 99% filled for a zero-output length stop', () {
      expect(isContextOverflow(lengthStop(input: 120000, cacheRead: 0, output: 0), 200000), isFalse);
      expect(isContextOverflow(lengthStop(input: 198000, cacheRead: 0, output: 0), 200000), isTrue);
    });

    test('does not treat zero-output length stops far below context as context overflow', () {
      expect(isContextOverflow(lengthStop(input: 100, cacheRead: 0, output: 0), 200000), isFalse);
    });
  });

  group('isRecoverableLength', () {
    test('treats a length stop below the desired output limit as recoverable', () {
      final message = lengthStop(input: 3, cacheRead: 253584, cacheWrite: 25554, output: 16);
      expect(isRecoverableLength(message, 128000), isTrue);
    });

    test('does not recover a length stop that reached the desired output limit', () {
      expect(isRecoverableLength(lengthStop(input: 4062, cacheRead: 0, output: 1024), 1024), isFalse);
    });

    test('treats zero-output length stops as recoverable without context metadata', () {
      expect(isRecoverableLength(lengthStop(input: 100, cacheRead: 0, output: 0), 128000), isTrue);
    });
  });
}
