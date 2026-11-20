/// Live checks against the OpenAI Responses API, through the built-in providers: a plain answer, a strict tool round
/// trip through the Agent, reasoning with its encrypted item, that item sent back in a second turn, prompt caching, an
/// image, a real API error, and the in-place changes: effort changed mid-conversation (configuration_update items),
/// a tool added mid-conversation (additional_tools items), a system message mid-conversation, and the session
/// headers. The change checks read the requests actually sent.
///
/// Run with a key in the environment; it is never printed:
///
///     OPENAI_API_KEY=... dart run tool/live_openai.dart [model]
///
/// The model comes from the catalog and defaults to gpt-6-luna, the cheapest one with every feature checked here.
/// Exits with 1 when a check fails, and with 2 without a key or for an unknown model.
library;

import 'dart:io';

import 'package:chorus/chorus.dart';

import 'live_common.dart';

Future<void> main(List<String> args) async {
  final key = Platform.environment['OPENAI_API_KEY'];
  if (key == null || key.isEmpty) {
    stderr.writeln('Set OPENAI_API_KEY.');
    exit(2);
  }

  final model = getBuiltinModel('openai', args.firstOrNull ?? 'gpt-6-luna');
  if (model == null) {
    stderr.writeln('Unknown model. OpenAI models: ${getBuiltinModels('openai').map((m) => m.id).join(', ')}');
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
    'strict tool round trip',
    has(adding.requests.first.body, (t) => t['name'] == 'add' && t['strict'] == true) &&
        add.messages.whereType<ToolResultMessage>().isNotEmpty &&
        text(sum).contains('5555'),
    sum,
  );

  final thinking = Agent(model: model, level: ThinkingLevel.low, options: options, stream: models.streamSimple);
  await thinking.promptText('Is 391 prime? Answer yes or no.');
  final first = thinking.messages.last as AssistantMessage;
  check(
    'reasoning with its encrypted item',
    first.content.whereType<ThinkingContent>().any((c) => c.signature?.contains('encrypted_content') ?? false) &&
        text(first).toLowerCase().startsWith('no'),
    first,
  );

  await thinking.promptText('And 397?');
  final second = thinking.messages.last as AssistantMessage;
  check(
    'reasoning sent back in a 2nd turn',
    second.stopReason == StopReason.stop && text(second).toLowerCase().startsWith('yes'),
    second,
  );

  // A system prompt well above OpenAI's 1024-token cache minimum, sent twice in one session.
  final cached = SimpleStreamOptions(
    apiKey: key,
    maxTokens: 200,
    sessionId: 'chorus-live-${DateTime.now().millisecondsSinceEpoch}',
  );
  final transcript = <Message>[
    SystemMessage([
      TextContent(List.generate(800, (i) => 'Rule $i: answer briefly and never mention rule numbers.').join('\n')),
    ], timestamp: 0),
    UserMessage([TextContent('Reply with exactly: one')], timestamp: 1),
  ];
  final written = await models.completeSimple(model, transcript, cached);
  transcript.addAll([
    written,
    UserMessage([TextContent('Reply with exactly: two')], timestamp: 2),
  ]);
  final read = await models.completeSimple(model, transcript, cached);
  stdout.writeln('     cache: read ${written.usage.cacheRead}, then ${read.usage.cacheRead}');
  check('prompt caching', read.usage.cacheRead > 0, read);

  final image = await models.completeSimple(model, [
    UserMessage([
      const ImageContent(redPng, 'image/png'),
      TextContent('What color is this image? Answer with one word.'),
    ], timestamp: 1),
  ], options);
  check('image', text(image).toLowerCase().contains('red'), image);

  final rejected = await models.completeSimple(model, [
    UserMessage([TextContent('hi')], timestamp: 1),
  ], const SimpleStreamOptions(apiKey: 'sk-not-a-key', maxTokens: 100));
  check(
    'real error (bad key)',
    rejected.stopReason == StopReason.error && (rejected.errorMessage?.contains('(401)') ?? false),
    rejected,
  );

  // In-place changes, in one conversation: the second turn raises the effort and adds a tool and a system message.
  final recorder = Recorder();
  final session = 'chorus-live-changes-${DateTime.now().millisecondsSinceEpoch}';
  final agent = Agent(
    model: model,
    systemPrompt: 'Use the tools for arithmetic, then answer with the number only.',
    tools: [arithmetic('add', 'Adds', (a, b) => a + b)],
    level: ThinkingLevel.low,
    options: SimpleStreamOptions(apiKey: key, maxTokens: 4000, httpClient: recorder, sessionId: session),
    stream: models.streamSimple,
  );
  await agent.promptText('What is 1234 + 4321?');
  final turn = recorder.requests.length;
  agent
    ..level = ThinkingLevel.high
    ..tools = [...agent.tools, arithmetic('multiply', 'Multiplies', (a, b) => a * b)]
    ..messages = [
      ...agent.messages,
      SystemMessage([
        TextContent('After the number, add the word done.'),
      ], timestamp: DateTime.now().millisecondsSinceEpoch),
    ];
  await agent.promptText('What is 12 * 34?');
  final product = agent.messages.last as AssistantMessage;
  final changed = recorder.requests[turn].body;
  final input = changed['input'] as List<Object?>;
  final efforts = [
    for (final item in input)
      if (item case {'type': 'configuration_update', 'reasoning': {'effort': final String effort}}) effort,
  ];
  stdout.writeln(
    '     efforts in the 2nd turn: $efforts, request effort: ${(changed['reasoning'] as Json?)?['effort']}',
  );
  check('effort changed mid-conversation', efforts.contains('low') && efforts.last == 'high', product);
  check(
    'tool added mid-conversation',
    changed['tools'] == null &&
        has(input, (i) => i['type'] == 'additional_tools' && has(i['tools'], (t) => t['name'] == 'multiply')) &&
        agent.messages.whereType<ToolResultMessage>().any((r) => r.name == 'multiply') &&
        text(product).contains('408'),
    product,
  );
  check(
    'system message mid-conversation',
    has(input, (m) => m['role'] == 'developer' && '${m['content']}'.contains('add the word done')) &&
        text(product).toLowerCase().contains('done'),
    product,
  );
  final headers = recorder.requests.first.headers;
  check(
    'session headers and cache key',
    headers['session-id'] == session &&
        headers['x-client-request-id'] == session &&
        recorder.requests.every((r) => r.body['prompt_cache_key'] == session),
    product,
  );

  exit(failed ? 1 : 0);
}
