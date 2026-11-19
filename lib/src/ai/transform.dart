import 'model.dart';
import 'types.dart';

const _userImage = '(image omitted: model does not support images)';
const _toolImage = '(tool image omitted: model does not support images)';
const _errorImage = '(image attached after the tool results)';

/// An error tool result split for APIs that reject images in error results: [content] goes in the result, and
/// [caption] and [images] after all tool results of the turn.
class SplitResult {
  const SplitResult(this.content, this.caption, this.images);

  final List<TextContent> content;
  final String caption;
  final List<ImageContent> images;
}

/// Splits the images out of an error tool result; `null` when there is nothing to split.
SplitResult? splitErrorImages(ToolResultMessage message) {
  if (!message.isError) {
    return null;
  }

  final images = message.content.whereType<ImageContent>().toList();
  if (images.isEmpty) {
    return null;
  }

  return SplitResult(
    _replaceImages(message.content, _errorImage),
    'Images from the error result of tool call ${message.id} (${message.name}):',
    images,
  );
}

/// Replaces each run of images with one [placeholder] text.
List<TextContent> _replaceImages(List<Content> content, String placeholder) {
  final result = <TextContent>[];
  var placed = false;
  for (final block in content) {
    if (block is ImageContent) {
      if (!placed) {
        result.add(TextContent(placeholder));
      }

      placed = true;
      continue;
    }

    final text = block as TextContent;
    result.add(text);
    placed = text.text == placeholder;
  }

  return result;
}

List<Message> _downgradeImages(List<Message> messages, Model model) {
  if (model.input.contains('image')) {
    return messages;
  }

  return [
    for (final m in messages)
      switch (m) {
        UserMessage() => UserMessage(_replaceImages(m.content, _userImage), timestamp: m.timestamp),
        ToolResultMessage() => _withResult(m, content: _replaceImages(m.content, _toolImage)),
        _ => m,
      },
  ];
}

ToolResultMessage _withResult(ToolResultMessage m, {String? id, List<Content>? content}) {
  return ToolResultMessage(
    id: id ?? m.id,
    name: m.name,
    content: content ?? m.content,
    details: m.details,
    usage: m.usage,
    isError: m.isError,
    timestamp: m.timestamp,
    durationMs: m.durationMs,
  );
}

AssistantMessage _withContent(AssistantMessage m, List<Content> content) {
  return AssistantMessage(
    content: content,
    api: m.api,
    provider: m.provider,
    model: m.model,
    responseModel: m.responseModel,
    responseId: m.responseId,
    providerThinkingLevel: m.providerThinkingLevel,
    thinkingLevel: m.thinkingLevel,
    usage: m.usage,
    stopReason: m.stopReason,
    errorMessage: m.errorMessage,
    rawStopReason: m.rawStopReason,
    endTurn: m.endTurn,
    timestamp: m.timestamp,
    durationMs: m.durationMs,
  );
}

/// Prepares a transcript for [model], ported from pi's `transformMessages`.
///
/// Images become placeholders for models without image input. Thinking from another model is dropped, and so are
/// its tool calls' thought signatures; their ids go through [normalizeId], and tool results follow the new ids.
/// Errored and aborted assistant messages are skipped. A tool call with no result gets an error result saying none
/// was provided, before the next assistant or user message; system messages between a call and its results move
/// after them.
List<Message> transformMessages(List<Message> messages, Model model, [String Function(String id)? normalizeId]) {
  final ids = <String, String>{};
  final transformed = <Message>[];
  for (final m in _downgradeImages(messages, model)) {
    switch (m) {
      case ToolResultMessage():
        final id = ids[m.id];
        if (id == null) {
          transformed.add(m);
        } else {
          transformed.add(_withResult(m, id: id));
        }
      case AssistantMessage():
        final same = m.provider == model.provider && m.api == model.api && m.model == model.id;
        final content = <Content>[];
        for (final block in m.content) {
          switch (block) {
            case ThinkingContent():
              if (!same) {
                continue;
              }

              if (block.redacted || (block.signature?.isNotEmpty ?? false)) {
                content.add(block);
                continue;
              }

              if (block.thinking.trim().isNotEmpty) {
                content.add(block);
              }
            case TextContent():
              content.add(same ? block : TextContent(block.text));
            case ToolCall():
              var id = block.id;
              if (!same && normalizeId != null) {
                id = normalizeId(block.id);
              }

              if (id != block.id) {
                ids[block.id] = id;
              }

              if (same || (block.thoughtSignature == null && id == block.id)) {
                content.add(block);
                continue;
              }

              content.add(
                ToolCall(
                  id,
                  block.name,
                  block.arguments,
                  namespace: block.namespace,
                  argumentsError: block.argumentsError,
                ),
              );
            case ImageContent():
              content.add(block);
          }
        }

        transformed.add(_withContent(m, content));
      default:
        transformed.add(m);
    }
  }

  final result = <Message>[];
  var pending = <ToolCall>[];
  var answered = <String>{};
  final held = <Message>[];

  void close() {
    for (final call in pending) {
      if (!answered.contains(call.id)) {
        result.add(
          ToolResultMessage(
            id: call.id,
            name: call.name,
            content: [TextContent('No result provided')],
            isError: true,
            timestamp: DateTime.now().millisecondsSinceEpoch,
          ),
        );
      }
    }

    pending = [];
    answered = {};
    result.addAll(held);
    held.clear();
  }

  for (final m in transformed) {
    switch (m) {
      case AssistantMessage():
        close();
        if (m.stopReason == StopReason.error || m.stopReason == StopReason.aborted) {
          continue;
        }

        final calls = m.content.whereType<ToolCall>().toList();
        if (calls.isNotEmpty) {
          pending = calls;
          answered = {};
        }

        result.add(m);
      case ToolResultMessage():
        answered.add(m.id);
        result.add(m);
      case SystemMessage():
        if (pending.isNotEmpty) {
          held.add(m);
        } else {
          result.add(m);
        }
      case UserMessage():
        close();
        result.add(m);
    }
  }

  close();
  return result;
}
