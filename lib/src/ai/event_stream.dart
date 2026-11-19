import 'dart:async';

import 'types.dart';

/// A stream of events that ends with a result, ported from pi's `EventStream`.
///
/// Events are buffered until someone listens. The event [isLast] accepts completes [result] with what [resultOf]
/// takes from it and ends the stream; anything pushed after it is dropped.
class EventStream<T, R> extends Stream<T> {
  EventStream(this.isLast, this.resultOf);

  final bool Function(T event) isLast;
  final R Function(T event) resultOf;
  final _events = StreamController<T>();
  final _result = Completer<R>();

  /// The result: from the last event, or the one passed to [end].
  Future<R> get result => _result.future;

  /// Whether the stream has ended.
  bool get done => _events.isClosed;

  void push(T event) {
    if (_events.isClosed) {
      return;
    }

    _events.add(event);
    if (isLast(event)) {
      end(resultOf(event));
    }
  }

  /// Ends the stream, with [value] as the result when it has not been set yet.
  void end([R? value]) {
    if (value != null && !_result.isCompleted) {
      _result.complete(value);
    }

    _events.close();
  }

  @override
  StreamSubscription<T> listen(
    void Function(T event)? onData, {
    Function? onError,
    void Function()? onDone,
    bool? cancelOnError,
  }) {
    return _events.stream.listen(onData, onError: onError, onDone: onDone, cancelOnError: cancelOnError);
  }
}

/// The events of one assistant response, ported from pi's `AssistantMessageEventStream`.
///
/// A [DoneEvent] or [ErrorEvent] ends it. The final message also gets `durationMs`, measured from this stream's
/// creation, unless it already has one or its timestamp predates the stream (a response that started elsewhere).
class AssistantMessageEventStream extends EventStream<AssistantMessageEvent, AssistantMessage> {
  AssistantMessageEventStream() : super((e) => e is DoneEvent || e is ErrorEvent, _final);

  final _start = DateTime.now().millisecondsSinceEpoch;
  final _clock = Stopwatch()..start();

  static AssistantMessage _final(AssistantMessageEvent event) {
    return switch (event) {
      DoneEvent(:final message) => message,
      ErrorEvent(:final error) => error,
      _ => throw StateError('Not a final event: $event'),
    };
  }

  @override
  void push(AssistantMessageEvent event) {
    if (event is DoneEvent || event is ErrorEvent) {
      _time(_final(event));
    }

    super.push(event);
  }

  @override
  void end([AssistantMessage? value]) {
    if (value != null) {
      _time(value);
    }

    super.end(value);
  }

  void _time(AssistantMessage message) {
    if (done) {
      return;
    }

    if (message.durationMs != null) {
      return;
    }

    if (message.timestamp < _start) {
      return;
    }

    message.durationMs = _clock.elapsedMilliseconds;
  }
}
