import 'dart:async';
import 'dart:math';

import 'package:anthropic_sdk_lite_dart/anthropic_sdk_lite_dart.dart' as anthropic;
import 'package:http_parser/http_parser.dart';
import 'package:openai_sdk_lite_dart/openai_sdk_lite_dart.dart' as openai;

import 'options.dart';

/// Thrown when a request is aborted through its [AbortSignal].
class AbortedException implements Exception {
  const AbortedException();

  @override
  String toString() => 'Request was aborted';
}

final _random = Random();

/// Runs [request], retrying it like the Anthropic and OpenAI SDKs do, but with a backoff that [signal] can interrupt.
///
/// Retried: connection failures, 408, 409, 429 and 5xx, unless `x-should-retry` says otherwise. The delay is the
/// server's `retry-after-ms` or `retry-after`, else exponential from half a second up to 8 seconds, with jitter. A
/// server delay above [maxDelay] (one minute by default; [Duration.zero] for no limit) fails at once instead.
Future<T> retryRequest<T>(
  Future<T> Function() request, {
  int? maxRetries,
  Duration? maxDelay,
  AbortSignal? signal,
}) async {
  for (var attempt = 0; ; attempt++) {
    try {
      return await request();
    } catch (error) {
      final failure = _failure(error);
      if (failure == null) {
        rethrow;
      }

      if (signal?.aborted ?? false) {
        throw const AbortedException();
      }

      if (attempt >= (maxRetries ?? 0) || !_retryable(failure)) {
        rethrow;
      }

      await _sleep(_delay(failure, attempt, maxDelay), signal);
    }
  }
}

/// What a retry reads from an SDK's API error.
typedef _Failure = ({int? status, Map<String, String>? headers, String? message});

/// [error] as a failure that may be retried: an API error of either SDK, or null for anything else.
_Failure? _failure(Object error) {
  return switch (error) {
    anthropic.APIError() => (status: error.status, headers: error.headers, message: error.message),
    openai.APIError() => (status: error.status, headers: error.headers, message: error.message),
    _ => null,
  };
}

bool _retryable(_Failure error) {
  final should = error.headers?['x-should-retry'];
  if (should == 'true') {
    return true;
  }

  if (should == 'false') {
    return false;
  }

  final status = error.status;
  if (status == null) {
    return true;
  }

  return status == 408 || status == 409 || status == 429 || status >= 500;
}

Duration _delay(_Failure error, int attempt, Duration? maxDelay) {
  final ms = double.tryParse(error.headers?['retry-after-ms'] ?? '');
  if (ms != null && ms.isFinite) {
    return _allowed(Duration(microseconds: (ms * 1000).round()), maxDelay, error);
  }

  final after = error.headers?['retry-after'];
  if (after != null) {
    final seconds = double.tryParse(after);
    if (seconds != null && seconds.isFinite) {
      return _allowed(Duration(microseconds: (seconds * 1e6).round()), maxDelay, error);
    }

    final date = _httpDate(after);
    if (date != null) {
      return _allowed(date.difference(DateTime.now()), maxDelay, error);
    }
  }

  return Duration(microseconds: (min(0.5 * pow(2, attempt), 8) * 1e6 * (1 - _random.nextDouble() * 0.25)).round());
}

Duration _allowed(Duration delay, Duration? maxDelay, _Failure error) {
  final limit = maxDelay ?? const Duration(minutes: 1);
  if (limit > Duration.zero && delay > limit) {
    throw Exception(
      'Server requested ${(delay.inMilliseconds / 1000).ceil()}s retry delay '
      '(max: ${(limit.inMilliseconds / 1000).ceil()}s). ${error.message}',
    );
  }

  return delay;
}

DateTime? _httpDate(String text) {
  try {
    return parseHttpDate(text);
  } on FormatException {
    return null;
  }
}

Future<void> _sleep(Duration delay, AbortSignal? signal) async {
  if (signal?.aborted ?? false) {
    throw const AbortedException();
  }

  if (delay <= Duration.zero) {
    return;
  }

  final done = Completer<void>();
  final timer = Timer(delay, done.complete);
  signal?.future.then((_) {
    timer.cancel();
    if (!done.isCompleted) {
      done.completeError(const AbortedException());
    }
  });
  await done.future;
}
