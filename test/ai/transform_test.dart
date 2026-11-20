import 'package:chorus/chorus.dart';
import 'package:chorus/src/ai/transform.dart';
import 'package:test/test.dart';

// Ported from pi's test/transform-messages-cross-model-thinking.test.ts; the last group is chorus's own.

const kimi = Model(
  id: 'moonshotai/Kimi-K3',
  name: 'Kimi K3',
  api: 'openai-completions',
  provider: 'sference',
  baseUrl: 'https://api.sference.com/v1',
  reasoning: true,
  input: ['text', 'image'],
  cost: ModelCost(input: 0, output: 0, cacheRead: 0, cacheWrite: 0),
  contextWindow: 1048576,
  maxTokens: 32768,
);

AssistantMessage assistant(String model, List<Content> content, [StopReason stop = StopReason.toolUse]) {
  return AssistantMessage(
    content: content,
    api: 'openai-completions',
    provider: 'sference',
    model: model,
    stopReason: stop,
    timestamp: 1,
  );
}

UserMessage user(String text) => UserMessage([TextContent(text)], timestamp: 1);

ThinkingContent thinking(String text) => ThinkingContent(text, signature: 'reasoning_content');

List<Object?> contents(List<Message> messages) {
  return [
    for (final m in messages.whereType<AssistantMessage>()) [for (final c in m.content) c.toJson()],
  ];
}

void main() {
  group('thinking from other models', () {
    // pi's https://github.com/earendil-works/pi/issues/10995
    test("does not send another model's readable thinking as text", () {
      expect(
        contents(
          transformMessages([
            user('Why does curl work but node fails?'),
            assistant('zai-org/GLM-5.3', [
              thinking('The user asks about ECONNRESET. Candidates: proxy, IPv6, TLS.'),
              TextContent('Let me check the TLS versions.'),
              ToolCall('call_1', 'bash', {'command': 'curl -v'}),
            ]),
            ToolResultMessage(
              id: 'call_1',
              name: 'bash',
              content: [TextContent('TLSv1.2')],
              isError: false,
              timestamp: 1,
            ),
          ], kimi),
        ),
        [
          [
            {'type': 'text', 'text': 'Let me check the TLS versions.'},
            {
              'type': 'toolCall',
              'id': 'call_1',
              'name': 'bash',
              'arguments': {'command': 'curl -v'},
            },
          ],
        ],
      );
    });

    test('keeps thinking only from the target model when models alternate', () {
      Json t(String text) => {'type': 'thinking', 'thinking': text, 'thinkingSignature': 'reasoning_content'};
      expect(
        contents(
          transformMessages([
            user('1'),
            assistant('moonshotai/Kimi-K3', [thinking('kimi 1'), TextContent('a')], StopReason.stop),
            user('2'),
            assistant('zai-org/GLM-5.3', [thinking('glm 1'), TextContent('b')], StopReason.stop),
            user('3'),
            assistant('moonshotai/Kimi-K3', [thinking('kimi 2'), TextContent('c')], StopReason.stop),
            user('4'),
            assistant('zai-org/GLM-5.3', [thinking('glm 2'), TextContent('d')], StopReason.stop),
            user('hi'),
          ], kimi),
        ),
        [
          [
            t('kimi 1'),
            {'type': 'text', 'text': 'a'},
          ],
          [
            {'type': 'text', 'text': 'b'},
          ],
          [
            t('kimi 2'),
            {'type': 'text', 'text': 'c'},
          ],
          [
            {'type': 'text', 'text': 'd'},
          ],
        ],
      );
    });
  });

  group('tool calls and results', () {
    test('answers a tool call left without a result, before the next user message', () {
      final result = transformMessages([
        user('go'),
        assistant('moonshotai/Kimi-K3', [ToolCall('call_1', 'bash', {})]),
        user('stop'),
      ], kimi);

      expect(result.map((m) => m.runtimeType), [UserMessage, AssistantMessage, ToolResultMessage, UserMessage]);
      final answer = result[2] as ToolResultMessage;
      expect(answer.id, 'call_1');
      expect(answer.isError, isTrue);
      expect((answer.content.single as TextContent).text, 'No result provided');
    });

    test('skips errored and aborted assistant messages', () {
      expect(
        transformMessages([
          user('go'),
          assistant('moonshotai/Kimi-K3', [TextContent('half')], StopReason.aborted),
          user('again'),
        ], kimi).whereType<AssistantMessage>(),
        isEmpty,
      );
    });

    test("renames another model's tool call ids, and their results with them", () {
      final result = transformMessages(
        [
          user('go'),
          assistant('zai-org/GLM-5.3', [ToolCall('call|1', 'bash', {})]),
          ToolResultMessage(id: 'call|1', name: 'bash', content: [TextContent('ok')], isError: false, timestamp: 1),
        ],
        kimi,
        (id, _) => id.replaceAll('|', '_'),
      );

      expect((result[1] as AssistantMessage).content.whereType<ToolCall>().single.id, 'call_1');
      expect((result[2] as ToolResultMessage).id, 'call_1');
    });

    test('replaces images for models without image input', () {
      expect(
        (transformMessages(
                  [
                    UserMessage([
                      TextContent('look'),
                      const ImageContent('AAAA', 'image/png'),
                      const ImageContent('BBBB', 'image/png'),
                    ], timestamp: 1),
                  ],
                  Model(
                    id: 'text-only',
                    name: 'Text only',
                    api: kimi.api,
                    provider: kimi.provider,
                    baseUrl: kimi.baseUrl,
                    cost: kimi.cost,
                    contextWindow: 1000,
                    maxTokens: 100,
                  ),
                ).single
                as UserMessage)
            .content
            .map((c) => c.toJson()),
        [
          {'type': 'text', 'text': 'look'},
          {'type': 'text', 'text': '(image omitted: model does not support images)'},
        ],
      );
    });
  });
}
