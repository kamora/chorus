/// Live checks against the Anthropic API, through the built-in providers: a plain answer, a tool round trip through the
/// Agent, adaptive thinking, signed thinking sent back in a second turn, budget-based thinking (on claude-haiku-4-5),
/// prompt caching, an image, a real API error, and the beta features: strict tools, effort changed mid-conversation,
/// tools and system messages changed mid-conversation, message threads, and server-side fallbacks (on
/// claude-fable-5-2, the cheapest model that has them). The beta checks read the requests actually sent.
///
/// Run with a key in the environment; it is never printed:
///
///     ANTHROPIC_API_KEY=... dart run tool/live.dart [model]
///
/// The model comes from the catalog and defaults to claude-haiku-5-5. Exits with 1 when a check fails, and with 2
/// without a key or for an unknown model.
library;

import 'dart:convert';
import 'dart:io';

import 'package:chorus/chorus.dart';

import 'live_common.dart';

Future<void> main(List<String> args) async {
  final key = Platform.environment['ANTHROPIC_API_KEY'];
  if (key == null || key.isEmpty) {
    stderr.writeln('Set ANTHROPIC_API_KEY.');
    exit(2);
  }

  final model = getBuiltinModel('anthropic', args.firstOrNull ?? 'claude-haiku-5-5');
  if (model == null) {
    stderr.writeln('Unknown model. Anthropic models: ${getBuiltinModels('anthropic').map((m) => m.id).join(', ')}');
    exit(2);
  }

  final options = SimpleStreamOptions(apiKey: key, maxTokens: 4000);
  final models = builtinModels();
  final plain = await models.completeSimple(model, [
    UserMessage([TextContent('Reply with exactly: pong')], timestamp: 1),
  ], options);
  check('plain answer', text(plain).toLowerCase().contains('pong'), plain);

  final add = Agent(
    model: model,
    systemPrompt: 'Use the add tool for arithmetic, then answer with the number only.',
    tools: [
      AgentTool(
        name: 'add',
        description: 'Adds two numbers.',
        parameters: const {
          'type': 'object',
          'properties': {
            'a': {'type': 'number'},
            'b': {'type': 'number'},
          },
          'required': ['a', 'b'],
        },
        execute: (id, args, signal, onUpdate) async {
          final json = args as Json;
          return AgentToolResult([TextContent('${(json['a'] as num) + (json['b'] as num)}')]);
        },
      ),
    ],
    options: options,
    stream: models.streamSimple,
  );
  await add.promptText('What is 1234 + 4321?');
  final answer = add.messages.last as AssistantMessage;
  check(
    'tool round trip',
    add.messages.whereType<ToolResultMessage>().isNotEmpty && text(answer).contains('5555'),
    answer,
  );

  final thinking = Agent(model: model, level: ThinkingLevel.low, options: options, stream: models.streamSimple);
  await thinking.promptText('Is 391 prime? Answer yes or no.');
  final first = thinking.messages.last as AssistantMessage;
  check(
    'adaptive thinking',
    first.content.whereType<ThinkingContent>().any((c) => c.signature?.isNotEmpty ?? false) &&
        text(first).toLowerCase().startsWith('no'),
    first,
  );

  await thinking.promptText('And 397?');
  final second = thinking.messages.last as AssistantMessage;
  check(
    'thinking sent back in a 2nd turn',
    second.stopReason == StopReason.stop && text(second).toLowerCase().startsWith('yes'),
    second,
  );

  final budget = getBuiltinModel('anthropic', 'claude-haiku-4-5')!;
  final older = Agent(model: budget, level: ThinkingLevel.low, options: options, stream: models.streamSimple);
  await older.promptText('Is 391 prime? Answer yes or no.');
  final budgeted = older.messages.last as AssistantMessage;
  check(
    'budget thinking (${budget.id})',
    budgeted.content.whereType<ThinkingContent>().any((c) => c.signature?.isNotEmpty ?? false) &&
        text(budgeted).toLowerCase().startsWith('no'),
    budgeted,
  );

  // A system prompt well above every model's cache minimum, sent twice in one session.
  final cached = SimpleStreamOptions(
    apiKey: key,
    maxTokens: 200,
    sessionId: 'chorus-live-${DateTime.now().millisecondsSinceEpoch}',
    cacheRetention: CacheRetention.short,
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
  stdout.writeln('     cache: written ${written.usage.cacheWrite}, then read ${read.usage.cacheRead}');
  check('prompt caching', written.usage.cacheWrite > 0 && read.usage.cacheRead >= written.usage.cacheWrite, read);

  final image = await models.completeSimple(model, [
    UserMessage([
      const ImageContent(redPng, 'image/png'),
      TextContent('What color is this image? Answer with one word.'),
    ], timestamp: 1),
  ], options);
  check('image', text(image).toLowerCase().contains('red'), image);

  final rejected = await models.completeSimple(model, [
    UserMessage([TextContent('hi')], timestamp: 1),
  ], const SimpleStreamOptions(apiKey: 'not-a-key', maxTokens: 100));
  check(
    'real error (bad key)',
    rejected.stopReason == StopReason.error && (rejected.errorMessage?.isNotEmpty ?? false),
    rejected,
  );

  // Beta features, in one conversation: the second turn raises the effort and adds a tool and a system message.
  final recorder = Recorder();

  // Each request as built, before a message thread cuts it down to the messages the server does not hold yet.
  final payloads = <Json>[];
  final beta = Agent(
    model: model,
    systemPrompt: 'Use the tools for arithmetic, then answer with the number only.',
    tools: [arithmetic('add', 'Adds', (a, b) => a + b)],
    level: ThinkingLevel.low,
    options: SimpleStreamOptions(
      apiKey: key,
      maxTokens: 4000,
      httpClient: recorder,
      sessionId: 'chorus-live-threads-${DateTime.now().millisecondsSinceEpoch}',
      onPayload: (payload, _) {
        payloads.add(jsonDecode(jsonEncode(payload)) as Json);
        return null;
      },
    ),
    stream: models.streamSimple,
  );
  await beta.promptText('What is 1234 + 4321?');
  final sum = beta.messages.last as AssistantMessage;
  final firstTurn = recorder.requests.length;
  check(
    'strict tools',
    has(recorder.requests.first.body['tools'], (t) => t['name'] == 'add' && t['strict'] == true) &&
        text(sum).contains('5555'),
    sum,
  );

  beta
    ..level = ThinkingLevel.high
    ..tools = [...beta.tools, arithmetic('multiply', 'Multiplies', (a, b) => a * b)]
    ..messages = [
      ...beta.messages,
      SystemMessage([
        TextContent('After the number, add the word done.'),
      ], timestamp: DateTime.now().millisecondsSinceEpoch),
    ];
  await beta.promptText('What is 12 * 34?');
  final product = beta.messages.last as AssistantMessage;
  final changed = recorder.requests[firstTurn];
  final efforts = [
    for (final m in payloads[firstTurn]['messages'] as List<Object?>)
      if (m case {'output_config': {'effort': final String effort}}) effort,
  ];
  stdout.writeln('     efforts in the 2nd turn: $efforts');
  check(
    'effort changed mid-conversation',
    changed.betas.contains('mid-conversation-output-config') && efforts.contains('low') && efforts.last == 'high',
    product,
  );
  check(
    'tool added mid-conversation',
    changed.betas.contains('inline-tools') &&
        has(changed.body['messages'], (m) => m['type'] == 'tool_addition') &&
        beta.messages.whereType<ToolResultMessage>().any((r) => r.name == 'multiply') &&
        text(product).contains('408'),
    product,
  );
  check(
    'system message mid-conversation',
    has(changed.body['messages'], (m) => m['role'] == 'system' && jsonEncode(m).contains('add the word done')) &&
        text(product).toLowerCase().contains('done'),
    product,
  );

  final threads = [for (final r in recorder.requests) (r.body['thread'] as Map?)?['type']];
  final fallbacks = [
    for (final m in beta.messages.whereType<AssistantMessage>())
      ...?m.diagnostics?.where((d) => d.type == 'anthropic_thread_fallback'),
  ];
  stdout.writeln('     threads: $threads, fallbacks: ${fallbacks.length}');
  check(
    'message threads',
    recorder.requests.every((r) => r.betas.contains('message-threads')) &&
        threads.first == 'create' &&
        threads.contains('continue') &&
        fallbacks.isEmpty,
    product,
  );

  final fable = getBuiltinModel('anthropic', 'claude-fable-5-2')!;
  final fallbackRecorder = Recorder();
  final served = await models.completeSimple(fable, [
    UserMessage([TextContent('Reply with exactly: pong')], timestamp: 1),
  ], SimpleStreamOptions(apiKey: key, maxTokens: 200, httpClient: fallbackRecorder));
  final sent = fallbackRecorder.requests.single;
  stdout.writeln('     fallbacks sent: ${sent.body['fallbacks']}, answered by ${served.responseModel ?? fable.id}');
  check(
    'server-side fallbacks accepted (${fable.id})',
    sent.betas.contains('server-side-fallback') &&
        sent.body['fallbacks'] is List &&
        text(served).toLowerCase().contains('pong'),
    served,
  );

  exit(failed ? 1 : 0);
}
