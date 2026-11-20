import 'dart:convert';
import 'dart:math';

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

/// How an API can express tool changes after the first request.
enum ToolSupport {
  /// Only the top-level tool list exists; it always holds the current tools.
  none,

  /// Tools can be added at a later message, but not removed or redefined.
  additions,

  /// Every tool, the first ones included, is declared in place, and a later declaration under the same name replaces
  /// the earlier one. Removals cannot be sent, so they are announced as text and the tools stay declared.
  changes,
}

/// How tool declarations split between the request's top-level tools and changes in place, ported from pi's
/// `resolveTranscriptTools`: the top-level [tools], and the number of leading system messages whose changes they
/// already hold ([folded]); every later system message carries its own.
///
/// With [ToolSupport.changes] the top-level list is empty. With [ToolSupport.additions] it is the tool state after the
/// last removal or redefinition, which cannot be sent in place, so the cached prefix breaks once per removal rather
/// than on every later addition. With [ToolSupport.none] it is the current tool set.
({List<Tool> tools, int folded}) resolveTranscriptTools(List<Message> messages, ToolSupport support) {
  final system = messages.whereType<SystemMessage>().toList();
  switch (support) {
    case ToolSupport.none:
      return (tools: currentTools(system), folded: system.length);
    case ToolSupport.changes:
      return (tools: <Tool>[], folded: 0);
    case ToolSupport.additions:
      final folded = max(initialSystemMessage(messages) == null ? 0 : 1, _lastNonAdditiveChange(system) + 1);
      return (tools: currentTools(system.sublist(0, folded)), folded: folded);
  }
}

/// The index of the last system message that removes a tool or declares a name declared before (even after its
/// removal), or -1.
int _lastNonAdditiveChange(List<SystemMessage> messages) {
  final declared = <String>{};
  var last = -1;
  for (final (index, message) in messages.indexed) {
    if (message.removed?.isNotEmpty ?? false) {
      last = index;
    }

    for (final tool in message.added ?? const <Tool>[]) {
      if (!declared.add(tool.name)) {
        last = index;
      }
    }
  }

  return last;
}
