/// Live checks against the Gemini API, through the built-in providers: a plain answer, a strict tool round trip
/// through the Agent (validated function calling), thinking at a Gemini level with its thought signature sent back in a
/// second turn, budget-based thinking (on gemini-2.5-flash-lite), prompt caching, an image, an image a tool returns
/// (nested in the function response), and a real API error. The request checks read the requests actually sent.
///
/// Run with a key in the environment; it is never printed:
///
///     GEMINI_API_KEY=... dart run tool/live_google.dart [model]
///
/// The model comes from the catalog and defaults to gemini-3.1-flash-lite, the cheapest Gemini 3 model.
/// Exits with 1 when a check fails, and with 2 without a key or for an unknown model.
library;

import 'dart:convert';
import 'dart:io';

import 'package:chorus/chorus.dart';

import 'live_common.dart';

Json _generation(Json body) => body['generationConfig'] as Json? ?? const {};

Future<void> main(List<String> args) async {
  final key = Platform.environment['GEMINI_API_KEY'];
  if (key == null || key.isEmpty) {
    stderr.writeln('Set GEMINI_API_KEY.');
    exit(2);
  }

  final model = getBuiltinModel('google', args.firstOrNull ?? 'gemini-3.1-flash-lite');
  if (model == null) {
    stderr.writeln('Unknown model. Google models: ${getBuiltinModels('google').map((m) => m.id).join(', ')}');
    exit(2);
  }

  final options = SimpleStreamOptions(apiKey: key, maxTokens: 4000);
  final models = builtinModels();
  final plain = await models.completeSimple(model, [
    UserMessage([TextContent('Reply with exactly: pong')], timestamp: 1),
  ], options);
  check('plain answer', text(plain).toLowerCase().contains('pong'), plain);

  final adding = Recorder();
  final add = Agent(
    model: model,
    systemPrompt: 'Use the add tool for arithmetic, then answer with the number only.',
    tools: [arithmetic('add', 'Adds', (a, b) => a + b)],
    options: SimpleStreamOptions(apiKey: key, maxTokens: 4000, httpClient: adding),
    stream: models.streamSimple,
  );
  await add.promptText('What is 1234 + 4321?');
  final sum = add.messages.last as AssistantMessage;
  check(
    'strict tool round trip (validated calling)',
    has(adding.requests.first.body['toolConfig'], (c) => c['mode'] == 'VALIDATED') &&
        add.messages.whereType<ToolResultMessage>().isNotEmpty &&
        text(sum).contains('5555'),
    sum,
  );

  final thinkingRequests = Recorder();
  final thinking = Agent(
    model: model,
    level: ThinkingLevel.low,
    options: SimpleStreamOptions(apiKey: key, maxTokens: 4000, httpClient: thinkingRequests),
    stream: models.streamSimple,
  );
  await thinking.promptText('Is 391 prime? Answer yes or no.');
  final first = thinking.messages.last as AssistantMessage;
  final signed = first.content.any(
    (c) => switch (c) {
      ThinkingContent(:final signature?) || TextContent(:final signature?) => signature.isNotEmpty,
      _ => false,
    },
  );
  stdout.writeln(
    '     thinking: ${first.content.whereType<ThinkingContent>().length} block(s), signed: $signed, '
    'config: ${jsonEncode(_generation(thinkingRequests.requests.first.body)['thinkingConfig'])}',
  );
  check('thinking at a Gemini level', switch (_generation(thinkingRequests.requests.first.body)['thinkingConfig']) {
    {'thinkingLevel': 'LOW'} => signed && text(first).toLowerCase().startsWith('no'),
    _ => false,
  }, first);

  await thinking.promptText('And 397?');
  final second = thinking.messages.last as AssistantMessage;
  check(
    'thought signature sent back in a 2nd turn',
    jsonEncode(thinkingRequests.requests.last.body['contents']).contains('thoughtSignature') &&
        second.stopReason == StopReason.stop &&
        text(second).toLowerCase().startsWith('yes'),
    second,
  );

  final budget = getBuiltinModel('google', 'gemini-2.5-flash-lite')!;
  final budgetRequests = Recorder();
  final older = Agent(
    model: budget,
    level: ThinkingLevel.low,
    options: SimpleStreamOptions(apiKey: key, maxTokens: 4000, httpClient: budgetRequests),
    stream: models.streamSimple,
  );
  await older.promptText('Is 391 prime? Answer yes or no.');
  final budgeted = older.messages.last as AssistantMessage;
  check('budget thinking (${budget.id})', switch (_generation(budgetRequests.requests.first.body)['thinkingConfig']) {
    {'thinkingBudget': final int tokens} => tokens > 0 && text(budgeted).toLowerCase().startsWith('no'),
    _ => false,
  }, budgeted);

  // A system prompt well above Gemini's implicit cache minimum, sent twice; Gemini caches implicitly, so a hit is
  // likely but not promised.
  final transcript = <Message>[
    SystemMessage([
      TextContent(List.generate(800, (i) => 'Rule $i: answer briefly and never mention rule numbers.').join('\n')),
    ], timestamp: 0),
    UserMessage([TextContent('Reply with exactly: one')], timestamp: 1),
  ];
  final written = await models.completeSimple(model, transcript, options);
  transcript.addAll([
    written,
    UserMessage([TextContent('Reply with exactly: two')], timestamp: 2),
  ]);
  final read = await models.completeSimple(model, transcript, options);
  stdout.writeln('     cache: read ${written.usage.cacheRead}, then ${read.usage.cacheRead}');
  check('prompt caching (implicit)', read.usage.cacheRead > 0, read);

  final image = await models.completeSimple(model, [
    UserMessage([
      const ImageContent(redPng, 'image/png'),
      TextContent('What color is this image? Answer with one word.'),
    ], timestamp: 1),
  ], options);
  check('image', text(image).toLowerCase().contains('red'), image);

  final looking = Recorder();
  final look = Agent(
    model: model,
    systemPrompt: 'Use the look tool, then answer with one word.',
    tools: [
      AgentTool(
        name: 'look',
        description: 'Shows the picture to describe.',
        parameters: const {'type': 'object', 'properties': <String, Object?>{}},
        execute: (id, args, signal, onUpdate) async => AgentToolResult([const ImageContent(redPng, 'image/png')]),
      ),
    ],
    options: SimpleStreamOptions(apiKey: key, maxTokens: 4000, httpClient: looking),
    stream: models.streamSimple,
  );
  await look.promptText('Look at the picture. What color is it?');
  final seen = look.messages.last as AssistantMessage;
  check(
    'image from a tool, in the function response',
    has(
          looking.requests.last.body['contents'],
          (p) => p['functionResponse'] is Map && has(p, (i) => i['inlineData'] != null),
        ) &&
        text(seen).toLowerCase().contains('red'),
    seen,
  );

  final rejected = await models.completeSimple(model, [
    UserMessage([TextContent('hi')], timestamp: 1),
  ], const SimpleStreamOptions(apiKey: 'not-a-key', maxTokens: 100));
  check(
    'real error (bad key)',
    rejected.stopReason == StopReason.error && (rejected.errorMessage?.contains('"code":400') ?? false),
    rejected,
  );

  exit(failed ? 1 : 0);
}
