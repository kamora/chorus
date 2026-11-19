import 'dart:convert';
import 'dart:io';

import 'package:chorus/chorus.dart';
import 'package:test/test.dart';

void main() {
  // Messages taken from pi's own session fixtures (packages/coding-agent/test/fixtures), one per shape, with long
  // texts trimmed. They must come back from a round trip exactly as pi wrote them.
  test('pi messages round-trip unchanged', () {
    final messages = jsonDecode(File('test/fixtures/pi-messages.json').readAsStringSync()) as List<Object?>;
    expect(messages, hasLength(14));
    for (final json in messages) {
      expect(jsonDecode(jsonEncode(Message.fromJson(json as Json).toJson())), equals(json));
    }
  });

  test('pi messages read into the right types', () {
    final messages = [
      for (final m in jsonDecode(File('test/fixtures/pi-messages.json').readAsStringSync()) as List<Object?>)
        Message.fromJson(m as Json),
    ];
    final call = messages.whereType<AssistantMessage>().expand((m) => m.content).whereType<ToolCall>().first;
    expect(call.id, startsWith('toolu_'));
    expect(call.arguments, isNotEmpty);

    final error = messages.whereType<AssistantMessage>().last;
    expect(error.stopReason, StopReason.error);
    expect(error.errorMessage, isNotEmpty);
    expect(messages.whereType<ToolResultMessage>().where((m) => m.isError), hasLength(1));
  });

  test('string content reads as one text block', () {
    expect(
      ((Message.fromJson({'role': 'user', 'content': 'Hello', 'timestamp': 1}) as UserMessage).content.single
              as TextContent)
          .text,
      'Hello',
    );

    expect(
      (Message.fromJson({
        'role': 'system',
        'content': 'Be brief.',
        'timestamp': 1,
      }) as SystemMessage).content.single.text,
      'Be brief.',
    );
  });

  test('a system message keeps sections and tool changes', () {
    final json = {
      'role': 'system',
      'content': [
        {'type': 'text', 'text': 'Base prompt.'},
      ],
      'sections': {'<rules>': 'Rule one.', '<old>': null},
      'toolsAdded': [
        {
          'name': 'read',
          'description': 'Read a file.',
          'parameters': {
            'type': 'object',
            'properties': {
              'path': {'type': 'string'},
            },
          },
        },
      ],
      'toolsRemoved': [
        {'name': 'write'},
      ],
      'timestamp': 5,
    };
    final system = Message.fromJson(json) as SystemMessage;
    expect(system.sections, {'<rules>': 'Rule one.', '<old>': null});
    expect(system.added!.single.name, 'read');
    expect(system.removed, ['write']);
    expect(system.toJson(), equals(json));
  });

  test('optional fields stay out of the JSON when unset', () {
    final message = AssistantMessage(api: 'anthropic-messages', provider: 'anthropic', model: 'm', timestamp: 1);
    message.content.add(ThinkingContent('hm'));
    expect(message.toJson().keys, ['role', 'content', 'api', 'provider', 'model', 'usage', 'stopReason', 'timestamp']);
    expect(message.toJson()['content'], [
      {'type': 'thinking', 'thinking': 'hm'},
    ]);
  });

  test('unknown roles and content types are rejected', () {
    expect(() => Message.fromJson({'role': 'bashExecution', 'timestamp': 1}), throwsFormatException);
    expect(() => Content.fromJson({'type': 'audio'}), throwsFormatException);
  });
}
