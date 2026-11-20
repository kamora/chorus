import 'dart:convert';

import 'types.dart';

/// The leading system message, if the transcript starts with one.
SystemMessage? initialSystemMessage(List<Message> messages) {
  if (messages.isEmpty) {
    return null;
  }

  final first = messages.first;
  return first is SystemMessage ? first : null;
}

/// The tools available after replaying every system message in order.
List<Tool> currentTools(List<Message> messages) {
  final tools = <String, Tool>{};
  for (final message in messages.whereType<SystemMessage>()) {
    for (final name in message.removed ?? const <String>[]) {
      tools.remove(name);
    }

    for (final tool in message.added ?? const <Tool>[]) {
      tools[tool.name] = tool;
    }
  }

  return tools.values.toList();
}

/// Every system message replayed into one: later content appended, sections patched by name, the current tools.
SystemMessage? currentSystemMessage(List<Message> messages) {
  final content = <String>[];
  final sections = <String, String>{};
  int? timestamp;
  for (final message in messages.whereType<SystemMessage>()) {
    timestamp ??= message.timestamp;
    final text = contentText(message.content);
    if (text.isNotEmpty) {
      content.add(text);
    }

    message.sections?.forEach((name, value) {
      if (value == null) {
        sections.remove(name);
      } else {
        sections[name] = value;
      }
    });
  }

  final tools = currentTools(messages);
  if (timestamp == null && tools.isEmpty) {
    return null;
  }

  return SystemMessage(
    [TextContent(content.join('\n\n'))],
    sections: sections.isEmpty ? null : sections,
    added: tools.isEmpty ? null : tools,
    timestamp: timestamp ?? 0,
  );
}

/// The transcript for APIs without mid-conversation system messages: the replayed system message leads, and every
/// later one is dropped.
List<Message> collapseSystemMessages(List<Message> messages) {
  return [?currentSystemMessage(messages), ...messages.where((m) => m is! SystemMessage)];
}

/// The text blocks of [content], joined.
String contentText(List<Content> content, [String separator = '\n']) {
  return content.whereType<TextContent>().map((c) => c.text).join(separator);
}

/// A system message as a complete prompt: its content followed by its sections.
String systemText(SystemMessage message) {
  return [
    contentText(message.content),
    ...?message.sections?.values.nonNulls,
  ].where((part) => part.isNotEmpty).join('\n\n');
}

/// A later system message, for APIs that accept system messages mid-conversation: its content, then each section
/// change framed by name so the model can relate it to the leading prompt.
String renderSystemMessageUpdate(SystemMessage message) {
  final text = contentText(message.content);
  return [
    if (text.isNotEmpty) text,
    for (final MapEntry(:key, :value) in (message.sections ?? const <String, String?>{}).entries)
      value == null ? 'Removed system prompt section "$key".' : 'Updated system prompt section "$key":\n\n$value',
  ].join('\n\n');
}

/// What tools must be added and removed to go from [previous] to [current]; a changed definition is both. Tools
/// compare by their declaration: name, description and parameters, as JSON. Added tools are plain declarations, copied
/// through JSON.
({List<Tool> added, List<String> removed}) toolStateChanges(List<Tool> previous, List<Tool> current) {
  final before = {for (final t in previous) t.name: jsonEncode(t.toJson())};
  final after = {for (final t in current) t.name: jsonEncode(t.toJson())};
  return (
    added: [
      for (final t in current)
        if (before[t.name] != after[t.name]) Tool.fromJson(jsonDecode(after[t.name]!) as Json),
    ],
    removed: [
      for (final t in previous)
        if (after[t.name] != before[t.name]) t.name,
    ],
  );
}
