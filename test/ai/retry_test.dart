import 'package:anthropic_sdk_lite_dart/anthropic_sdk_lite_dart.dart';
import 'package:chorus/chorus.dart';
import 'package:chorus/src/ai/retry.dart';
import 'package:test/test.dart';

// Ported from pi's test/provider-retry.test.ts, with short real delays instead of fake timers. Left out: pi's
// `noRetryStatuses`, which only other providers use.

APIError providerError(int status, Map<String, String> headers) => APIError(status, null, headers);

/// A request that fails with [errors] in turn, then answers `ok`; [calls] counts the attempts.
class Flaky {
  Flaky(this.errors);

  final List<Object> errors;
  var calls = 0;

  Future<String> call() async {
    calls++;
    if (calls <= errors.length) {
      throw errors[calls - 1];
    }

    return 'ok';
  }
}

void main() {
  test('retries retryable provider errors after the requested delay', () async {
    final request = Flaky([
      providerError(429, {'retry-after-ms': '50'}),
    ]);
    final clock = Stopwatch()..start();
    expect(await retryRequest(request.call, maxRetries: 1), 'ok');
    expect(request.calls, 2);
    expect(clock.elapsedMilliseconds, greaterThanOrEqualTo(45));
  });

  test('does not retry errors the provider marks as non-retryable', () async {
    final error = providerError(429, {'x-should-retry': 'false'});
    final request = Flaky([error, error, error]);
    await expectLater(retryRequest(request.call, maxRetries: 2), throwsA(same(error)));
    expect(request.calls, 1);
  });

  test('does not retry a client error', () async {
    final error = providerError(400, {});
    final request = Flaky([error]);
    await expectLater(retryRequest(request.call, maxRetries: 2), throwsA(same(error)));
    expect(request.calls, 1);
  });

  test('rejects a provider-requested retry delay above the limit', () async {
    final request = Flaky([
      providerError(429, {'retry-after': '277403'}),
    ]);
    await expectLater(
      retryRequest(request.call, maxRetries: 1, maxDelay: const Duration(seconds: 1)),
      throwsA(predicate((e) => '$e'.contains('Server requested 277403s retry delay (max: 1s)'))),
    );
    expect(request.calls, 1);
  });

  test('allows disabling the provider-requested retry delay cap', () async {
    final request = Flaky([
      providerError(429, {'retry-after': '0.05'}),
    ]);
    expect(await retryRequest(request.call, maxRetries: 1, maxDelay: Duration.zero), 'ok');
    expect(request.calls, 2);
  });

  test('aborts a provider-requested retry delay', () async {
    final signal = AbortSignal();
    final request = Flaky([
      providerError(429, {'retry-after': '277403'}),
      providerError(429, {'retry-after': '277403'}),
    ]);
    final result = retryRequest(request.call, maxRetries: 2, maxDelay: Duration.zero, signal: signal);
    await Future<void>.delayed(const Duration(milliseconds: 20));
    expect(request.calls, 1);
    signal.abort();
    await expectLater(result, throwsA(isA<AbortedException>()));
    expect(request.calls, 1);
  });
}
