// chorus's own tests of the OpenAI Responses request: message conversion (roles, system updates, message ids and
// phases, tool call ids, tool results with images) and the reasoning and token parameters.

import 'package:chorus/chorus.dart';
import 'package:chorus/src/ai/providers/openai_responses.dart';
import 'package:test/test.dart';

const _free = ModelCost(input: 0, output: 0, cacheRead: 0, cacheWrite: 0);

Model model({
  String id = 'gpt-test',
  String provider = 'openai',
  bool reasoning = true,
  List<String> input = const ['text'],
  Map<String, String?>? levels,
  Json compat = const {},
}) {
  return Model(
    id: id,
    name: id,
    api: 'openai-responses',
    provider: provider,
    baseUrl: 'https://api.openai.com/v1',
    reasoning: reasoning,
    input: input,
    levels: levels,
    cost: _free,
    contextWindow: 400000,
    maxTokens: 128000,
    compat: compat,
  );
}

UserMessage user(String text) => UserMessage([TextContent(text)], timestamp: 0);

AssistantMessage reply(
  List<Content> content, {
  String provider = 'openai',
  String api = 'openai-responses',
  String model = 'gpt-test',
}) {
  return AssistantMessage(
    content: content,
    api: api,
    provider: provider,
    model: model,
    stopReason: StopReason.stop,
    timestamp: 0,
  );
}

ToolResultMessage result(String id, List<Content> content) {
  return ToolResultMessage(id: id, name: 'lookup', content: content, isError: false, timestamp: 0);
}

Json params(Model m, OpenAIResponsesOptions options) => buildResponsesParams(m, [user('hi')], options);

void main() {
  group('roles and system messages', () {
    test('instructions go as developer messages for reasoning models, system ones otherwise', () {
      final system = SystemMessage([TextContent('sys')], timestamp: 0);
      expect(convertResponsesMessages(model(), [system, user('hi')]).first['role'], 'developer');
      expect(convertResponsesMessages(model(reasoning: false), [system, user('hi')]).first['role'], 'system');
      expect(
        convertResponsesMessages(model(compat: {'supportsDeveloperRole': false}), [system, user('hi')]).first['role'],
        'system',
      );
    });

    test('a later system message is sent as an update, with its sections framed', () {
      final input = convertResponsesMessages(model(compat: {'supportsMidConvoSystemMessages': true}), [
        SystemMessage([TextContent('base')], sections: {'rules': 'old'}, timestamp: 0),
        user('hi'),
        SystemMessage([TextContent('more')], sections: {'rules': 'new'}, timestamp: 1),
      ]);
      expect(
        [for (final i in input) i['content']],
        [
          'base\n\nold',
          [
            {'type': 'input_text', 'text': 'hi'},
          ],
          'more\n\nUpdated system prompt section "rules":\n\nnew',
        ],
      );
    });

    test('an empty user message is left out', () {
      final input = convertResponsesMessages(model(), [UserMessage(const [], timestamp: 0), user('hi')]);
      expect(input, hasLength(1));
    });
  });

  group('assistant messages', () {
    test('text without a signature gets an id from its position, the leading system message not counted', () {
      final input = convertResponsesMessages(model(), [
        SystemMessage([TextContent('sys')], timestamp: 0),
        UserMessage(const [], timestamp: 0),
        user('one'),
        reply([TextContent('a'), TextContent('b')]),
      ]);
      expect([for (final i in input.where((i) => i['type'] == 'message')) i['id']], ['msg_pi_1', 'msg_pi_1_1']);
    });

    test('a message id over 64 characters is hashed, and an unknown phase is dropped', () {
      final long = 'msg_${'x' * 70}';
      final input = convertResponsesMessages(model(), [
        user('hi'),
        reply([TextContent('a', signature: '{"v":1,"id":"$long","phase":"other"}')]),
      ]);
      final message = input.last;
      expect(message['id'], allOf(startsWith('msg_'), hasLength(lessThanOrEqualTo(64)), isNot(long)));
      expect(message.containsKey('phase'), isFalse);
    });

    test('a plain signature is the message id', () {
      final input = convertResponsesMessages(model(), [
        user('hi'),
        reply([TextContent('a', signature: 'msg_plain')]),
      ]);
      expect(input.last['id'], 'msg_plain');
    });

    test("another model's reply keeps no item ids, nor thinking", () {
      final input = convertResponsesMessages(model(), [
        user('hi'),
        reply([
          ThinkingContent('t', signature: '{"type":"reasoning","id":"rs_1"}'),
          ToolCall('call_1|fc_1', 'lookup', const {}),
        ], model: 'gpt-other'),
        result('call_1|fc_1', [TextContent('r')]),
      ]);
      expect([for (final i in input) i['type'] ?? i['role']], ['user', 'function_call', 'function_call_output']);
      expect(input[1].containsKey('id'), isFalse);
      expect((input[1]['call_id'], input[2]['call_id']), ('call_1', 'call_1'));
    });

    test("another provider's tool call gets a hashed fc_ item id", () {
      final input = convertResponsesMessages(model(), [
        user('hi'),
        reply([ToolCall('toolu_1|item_x', 'lookup', const {})], provider: 'anthropic', api: 'anthropic-messages'),
        result('toolu_1|item_x', [TextContent('r')]),
      ]);
      final call = input[1];
      expect(call['call_id'], 'toolu_1');
      expect(call['id'], allOf(startsWith('fc_'), isNot('fc_item_x')));
    });

    test("a provider outside OpenAI's keeps no item id: the whole id is cleaned", () {
      final input = convertResponsesMessages(model(provider: 'proxy'), [
        user('hi'),
        reply([ToolCall('call.1|weird id_', 'lookup', const {})], provider: 'anthropic', api: 'anthropic-messages'),
        result('call.1|weird id_', [TextContent('r')]),
      ]);
      expect((input[1]['call_id'], input[2]['call_id']), ('call_1_weird_id', 'call_1_weird_id'));
    });
  });

  group('tool results', () {
    List<Object?> outputs(Model m, List<Content> content) {
      return [
        for (final i in convertResponsesMessages(m, [
          user('hi'),
          reply([ToolCall('call_1|fc_1', 'lookup', const {})]),
          result('call_1|fc_1', content),
        ]).where((i) => i['type'] == 'function_call_output'))
          i['output'],
      ];
    }

    final image = ImageContent('abc', 'image/png');

    test('text is joined, and an empty result says so', () {
      expect(outputs(model(), [TextContent('a'), TextContent('b')]), ['a\nb']);
      expect(outputs(model(), []), ['(no tool output)']);
    });

    // transformMessages replaces the image first, so pi's "(see attached image)" is never reached.
    test('an image the model cannot see is replaced by a note', () {
      expect(outputs(model(), [image]), ['(tool image omitted: model does not support images)']);
      expect(outputs(model(), [TextContent('t'), image]), ['t\n(tool image omitted: model does not support images)']);
    });

    test('a model with image input gets the image', () {
      expect(outputs(model(input: ['text', 'image']), [TextContent('t'), image]), [
        [
          {'type': 'input_text', 'text': 't'},
          {'type': 'input_image', 'detail': 'auto', 'image_url': 'data:image/png;base64,abc'},
        ],
      ]);
    });
  });

  group('parameters', () {
    test('max_output_tokens is at least 16, and temperature is sent', () {
      final p = params(model(), const OpenAIResponsesOptions(maxTokens: 4, temperature: 0.3));
      expect((p['max_output_tokens'], p['temperature']), (16, 0.3));
    });

    test('the effort goes through the level map, with encrypted reasoning included', () {
      final p = params(model(levels: {'high': 'deep'}), const OpenAIResponsesOptions(reasoningEffort: 'high'));
      expect(p['reasoning'], {'effort': 'deep', 'summary': 'auto'});
      expect(p['include'], ['reasoning.encrypted_content']);
    });

    test('a summary alone asks for medium effort, and is sent', () {
      final p = params(model(), const OpenAIResponsesOptions(reasoningSummary: 'concise'));
      expect(p['reasoning'], {'effort': 'medium', 'summary': 'concise'});
    });

    test('a model without reasoning gets no reasoning field', () {
      expect(
        params(model(reasoning: false), const OpenAIResponsesOptions(reasoningEffort: 'high')).containsKey('reasoning'),
        isFalse,
      );
    });

    test('the prompt cache key is cut to 64 characters, not counting UTF-16 halves', () {
      final key = '😀' * 70;
      expect(params(model(), OpenAIResponsesOptions(sessionId: key))['prompt_cache_key'], '😀' * 64);
    });
  });
}
