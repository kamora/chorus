import 'package:chorus/chorus.dart';
import 'package:test/test.dart';

import 'anthropic_fake.dart';

// Ported from pi's Anthropic request tests: anthropic-eager-tool-input-compat,
// anthropic-empty-thinking-signature-compat, anthropic-error-tool-result-images, anthropic-force-adaptive-thinking,
// anthropic-temperature-compat and anthropic-thinking-disable. pi reads several models from its catalog, which chorus
// doesn't have yet; those cases use a model with the catalog's settings for that model, named in each test.
// Catalog-only checks are left out.

class _Captured implements Exception {}

/// The request body `streamSimpleAnthropic` builds, caught in `onPayload` before anything is sent.
Future<Json> payload(
  Model model,
  List<Message> messages, [
  SimpleStreamOptions options = const SimpleStreamOptions(),
]) async {
  Json? captured;
  await streamSimpleAnthropic(
    model,
    messages,
    SimpleStreamOptions(
      apiKey: 'fake-key',
      reasoning: options.reasoning,
      temperature: options.temperature,
      cacheRetention: options.cacheRetention,
      onPayload: (body, _) {
        captured = body;
        throw _Captured();
      },
    ),
  ).result;
  return captured!;
}

AssistantMessage assistant(List<Content> content, {String provider = 'anthropic', String model = 'claude-test'}) {
  return AssistantMessage(
    content: content,
    api: 'anthropic-messages',
    provider: provider,
    model: model,
    stopReason: StopReason.stop,
    timestamp: 2,
  );
}

const lookup = Tool('lookup', 'Look up a value', {
  'type': 'object',
  'properties': {
    'value': {'type': 'string'},
  },
  'required': ['value'],
});

void main() {
  group('Anthropic eager tool input streaming compatibility', () {
    Future<FakeServer> capture(Json compat, {bool tools = true}) async {
      final server = FakeServer('');
      await streamAnthropic(claude(compat: {'forceAdaptiveThinking': true, ...compat}), [
        if (tools) SystemMessage(const [], added: const [lookup], timestamp: 0),
        user('Use the tool'),
      ], AnthropicOptions(client: server.client(), cacheRetention: CacheRetention.none)).result;
      return server;
    }

    test('sends per-tool eager_input_streaming by default', () async {
      final server = await capture({});
      expect((server.body['tools'] as List).first['eager_input_streaming'], isTrue);
      expect(server.requests.single.headers['anthropic-beta'], isNull);
    });

    test('uses the legacy fine-grained tool streaming beta when eager tool input streaming is disabled', () async {
      final server = await capture({'supportsEagerToolInputStreaming': false});
      expect((server.body['tools'] as List).first, isNot(contains('eager_input_streaming')));
      expect(server.requests.single.headers['anthropic-beta'], 'fine-grained-tool-streaming-2025-05-14');
    });

    test('does not send the legacy fine-grained tool streaming beta when there are no tools', () async {
      final server = await capture({'supportsEagerToolInputStreaming': false}, tools: false);
      expect(server.body, isNot(contains('tools')));
      expect(server.requests.single.headers['anthropic-beta'], isNull);
    });
  });

  group('Anthropic empty thinking signature compat', () {
    Model mimo([bool? empty]) {
      return claude(
        id: 'mimo-v2.6-pro',
        provider: 'xiaomi-token-plan-ams',
        input: const ['text'],
        compat: {'allowEmptySignature': ?empty},
      );
    }

    List<Message> context(String signature, [String thinking = 'internal reasoning']) {
      return [
        user('first'),
        assistant(
          [ThinkingContent(thinking, signature: signature)],
          provider: 'xiaomi-token-plan-ams',
          model: 'mimo-v2.6-pro',
        ),
        user('second'),
      ];
    }

    List<Object?> replayed(Json body) {
      return ((body['messages'] as List).firstWhere((m) => m['role'] == 'assistant') as Json)['content'] as List;
    }

    test('converts empty-signature thinking to text by default', () async {
      expect(replayed(await payload(mimo(), context(''))), [
        {'type': 'text', 'text': 'internal reasoning'},
      ]);
    });

    test('preserves empty thinking text when the signature is present', () async {
      expect(replayed(await payload(mimo(), context('signed-thinking', ''))), [
        {'type': 'thinking', 'thinking': '', 'signature': 'signed-thinking'},
      ]);
    });

    test('preserves empty-signature thinking when allowEmptySignature is enabled', () async {
      expect(replayed(await payload(mimo(true), context(' '))), [
        {'type': 'thinking', 'thinking': 'internal reasoning', 'signature': ''},
      ]);
    });
  });

  group('Anthropic error tool results with images', () {
    final model = claude(reasoning: false);

    Json image(String data) {
      return {
        'type': 'image',
        'source': {'type': 'base64', 'media_type': 'image/png', 'data': data},
      };
    }

    // pi's regression for #10794: Anthropic rejects error tool results that contain images.
    test('moves the images after the tool results of the turn', () async {
      expect(
        ((await payload(model, [
              user('Show the screenshots'),
              assistant([ToolCall('call-1', 'codemode', {}), ToolCall('call-2', 'read', {})])
                ..stopReason = StopReason.toolUse,
              ToolResultMessage(
                id: 'call-1',
                name: 'codemode',
                content: [
                  TextContent('Script failed'),
                  const ImageContent('AAAA', 'image/png'),
                  const ImageContent('CCCC', 'image/png'),
                  TextContent('Script error'),
                ],
                isError: true,
                timestamp: 3,
              ),
              ToolResultMessage(
                id: 'call-2',
                name: 'read',
                content: const [ImageContent('BBBB', 'image/png')],
                isError: false,
                timestamp: 4,
              ),
            ], const SimpleStreamOptions(cacheRetention: CacheRetention.none)))['messages']
            as List)[2],
        {
          'role': 'user',
          'content': [
            {
              'type': 'tool_result',
              'tool_use_id': 'call-1',
              'content': 'Script failed\n(image attached after the tool results)\nScript error',
              'is_error': true,
            },
            {
              'type': 'tool_result',
              'tool_use_id': 'call-2',
              'content': [
                {'type': 'text', 'text': '(see attached image)'},
                image('BBBB'),
              ],
              'is_error': false,
            },
            {'type': 'text', 'text': 'Images from the error result of tool call call-1 (codemode):'},
            image('AAAA'),
            image('CCCC'),
          ],
        },
      );
    });

    test('keeps error tool results without images unchanged', () async {
      expect(
        ((await payload(model, [
              user('Run it'),
              assistant([ToolCall('call-1', 'codemode', {})])..stopReason = StopReason.toolUse,
              ToolResultMessage(
                id: 'call-1',
                name: 'codemode',
                content: [TextContent('Script failed')],
                isError: true,
                timestamp: 3,
              ),
            ], const SimpleStreamOptions(cacheRetention: CacheRetention.none)))['messages']
            as List)[2],
        {
          'role': 'user',
          'content': [
            {'type': 'tool_result', 'tool_use_id': 'call-1', 'content': 'Script failed', 'is_error': true},
          ],
        },
      );
    });
  });

  group('Anthropic forceAdaptiveThinking compat override', () {
    Model vendor([Json compat = const {}]) => claude(id: 'vendor--claude-opus-latest', compat: compat);

    test('sends legacy thinking payload for custom model ids by default', () async {
      final body = await payload(vendor(), [user('Hello')], const SimpleStreamOptions(reasoning: ThinkingLevel.medium));
      expect((body['thinking'] as Json)['type'], 'enabled');
      expect(body, isNot(contains('output_config')));
    });

    test('sends adaptive thinking payload when compat.forceAdaptiveThinking is true', () async {
      final body = await payload(vendor({'forceAdaptiveThinking': true}), [
        user('Hello'),
      ], const SimpleStreamOptions(reasoning: ThinkingLevel.medium));
      expect(body['thinking'], {'type': 'adaptive', 'display': 'summarized'});
      expect(body['output_config'], {'effort': 'medium'});
    });

    // Claude Fable 5 in pi's catalog: adaptive, and xhigh maps to native xhigh.
    test('uses adaptive thinking with native xhigh effort', () async {
      final body = await payload(
        claude(
          id: 'claude-fable-5',
          compat: {'forceAdaptiveThinking': true},
          levels: {'off': null, 'xhigh': 'xhigh', 'max': 'max'},
        ),
        [user('Hello')],
        const SimpleStreamOptions(reasoning: ThinkingLevel.xhigh),
      );
      expect(body['thinking'], {'type': 'adaptive', 'display': 'summarized'});
      expect(body['output_config'], {'effort': 'xhigh'});
    });

    test('preserves thinking.type=disabled when reasoning is off regardless of override', () async {
      final body = await payload(vendor({'forceAdaptiveThinking': true}), [user('Hello')]);
      expect(body['thinking'], {'type': 'disabled'});
      expect(body, isNot(contains('output_config')));
    });
  });

  group('Anthropic temperature compatibility', () {
    test('omits temperature for models with supportsTemperature disabled', () async {
      expect(
        (await payload(claude(id: 'vendor--claude-opus-4-7', compat: {'supportsTemperature': false}), [
          user('Hello'),
        ], const SimpleStreamOptions(temperature: 0))),
        isNot(contains('temperature')),
      );
    });

    // Claude Opus 4.8 in pi's catalog supports mid-conversation effort, which rules temperature out.
    test('omits temperature for models with mid-conversation effort', () async {
      expect(
        (await payload(claude(id: 'claude-opus-4-8', compat: {'supportsMidConvoEffort': true}), [
          user('Hello'),
        ], const SimpleStreamOptions(temperature: 0))),
        isNot(contains('temperature')),
      );
    });

    test('keeps temperature for other models', () async {
      expect(
        (await payload(claude(id: 'claude-sonnet-4-6'), [
          user('Hello'),
        ], const SimpleStreamOptions(temperature: 0)))['temperature'],
        0,
      );
    });
  });

  group('Anthropic thinking disable payload', () {
    test('sends thinking.type=disabled for budget-based reasoning models when thinking is off', () async {
      final body = await payload(claude(id: 'claude-sonnet-4-5'), [user('Hello')]);
      expect(body['thinking'], {'type': 'disabled'});
      expect(body, isNot(contains('output_config')));
    });

    test('sends thinking.type=disabled for adaptive reasoning models when thinking is off', () async {
      final body = await payload(claude(id: 'claude-opus-4-6', compat: {'forceAdaptiveThinking': true}), [
        user('Hello'),
      ]);
      expect(body['thinking'], {'type': 'disabled'});
      expect(body, isNot(contains('output_config')));
    });

    // Claude Fable 5 in pi's catalog maps `off` to null: its thinking cannot be turned off.
    test('omits thinking.type=disabled when the model cannot turn thinking off', () async {
      final body = await payload(
        claude(id: 'claude-fable-5', compat: {'forceAdaptiveThinking': true}, levels: {'off': null}),
        [user('Hello')],
      );
      expect(body, isNot(contains('thinking')));
      expect(body, isNot(contains('output_config')));
    });

    test('uses adaptive thinking when reasoning is enabled', () async {
      final body = await payload(claude(id: 'claude-sonnet-5', compat: {'forceAdaptiveThinking': true}), [
        user('Hello'),
      ], const SimpleStreamOptions(reasoning: ThinkingLevel.high));
      expect(body['thinking'], {'type': 'adaptive', 'display': 'summarized'});
      expect(body['output_config'], {'effort': 'high'});
    });
  });
}
