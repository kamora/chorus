import 'dart:async';

import 'types.dart';

/// The events of one assistant response, ported from pi's `AssistantMessageEventStream`.
///
/// Events are buffered until someone listens. A [DoneEvent] or [ErrorEvent] completes [result] and ends the stream;
/// anything pushed after it is dropped. The final message also gets `durationMs`, measured from this stream's
/// creation, unless it already has one or its timestamp predates the stream (a response that started elsewhere).
class AssistantMessageEventStream extends Stream<AssistantMessageEvent> {
  final _events = StreamController<AssistantMessageEvent>();
  final _result = Completer<AssistantMessage>();
  final _start = DateTime.now().millisecondsSinceEpoch;
  final _clock = Stopwatch()..start();

  /// The final message: from the [DoneEvent] or [ErrorEvent], or the one passed to [end].
  Future<AssistantMessage> get result => _result.future;

  void push(AssistantMessageEvent event) {
    if (_events.isClosed) {
      return;
    }

    final last = switch (event) {
      DoneEvent(:final message) => message,
      ErrorEvent(:final error) => error,
      _ => null,
    };
    if (last == null) {
      _events.add(event);
      return;
    }

    _time(last);
    _events.add(event);
    end(last);
  }

  /// Ends the stream, with [message] as the result when it has not been set yet.
  void end([AssistantMessage? message]) {
    if (message != null && !_result.isCompleted) {
      _time(message);
      _result.complete(message);
    }

    _events.close();
  }

  void _time(AssistantMessage message) {
    if (_result.isCompleted) {
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

  @override
  StreamSubscription<AssistantMessageEvent> listen(
    void Function(AssistantMessageEvent event)? onData, {
    Function? onError,
    void Function()? onDone,
    bool? cancelOnError,
  }) {
    return _events.stream.listen(onData, onError: onError, onDone: onDone, cancelOnError: cancelOnError);
  }
}
