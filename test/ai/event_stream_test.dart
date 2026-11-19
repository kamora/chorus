import 'package:chorus/chorus.dart';
import 'package:test/test.dart';

AssistantMessage message(int timestamp) {
  return AssistantMessage(api: 'anthropic-messages', provider: 'anthropic', model: 'm', timestamp: timestamp);
}

void main() {
  test('done ends the stream and gives the result', () async {
    final stream = AssistantMessageEventStream();
    final partial = message(DateTime.now().millisecondsSinceEpoch);
    stream.push(StartEvent(partial));
    stream.push(TextStartEvent(0, partial));
    stream.push(DoneEvent(StopReason.stop, partial));
    stream.push(TextDeltaEvent(0, 'late', partial));

    expect(await stream.toList(), [isA<StartEvent>(), isA<TextStartEvent>(), isA<DoneEvent>()]);
    expect(await stream.result, same(partial));
    expect(partial.durationMs, isNotNull);
  });

  test('an error before start ends the stream with the error message', () async {
    final stream = AssistantMessageEventStream();
    final failed = message(DateTime.now().millisecondsSinceEpoch)
      ..stopReason = StopReason.error
      ..errorMessage = 'no auth';
    stream.push(ErrorEvent(StopReason.error, failed));

    expect(await stream.toList(), [isA<ErrorEvent>()]);
    expect((await stream.result).errorMessage, 'no auth');
  });

  test('a response that started before the stream is not timed', () async {
    final stream = AssistantMessageEventStream();
    final earlier = message(DateTime.now().millisecondsSinceEpoch - 60000);
    stream.end(earlier);

    expect(await stream.toList(), isEmpty);
    expect(await stream.result, same(earlier));
    expect(earlier.durationMs, isNull);
  });

  test('a duration already set is kept', () async {
    final stream = AssistantMessageEventStream();
    final timed = message(DateTime.now().millisecondsSinceEpoch)..durationMs = 1234;
    stream.push(DoneEvent(StopReason.stop, timed));

    await stream.drain<void>();
    expect(timed.durationMs, 1234);
  });
}
