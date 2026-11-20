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
import 'package:http/http.dart' as http;

String _text(AssistantMessage message) => message.content.whereType<TextContent>().map((c) => c.text).join().trim();

var _failed = false;

/// Keeps the body and the beta header of every request it sends.
class _Recorder extends http.BaseClient {
  final _inner = http.Client();
  final requests = <({Json body, String betas})>[];

  @override
  Future<http.StreamedResponse> send(http.BaseRequest request) {
    if (request is http.Request) {
      requests.add((
        body: jsonDecode(request.body) as Json,
        betas: [
          for (final MapEntry(:key, :value) in request.headers.entries)
            if (key.toLowerCase() == 'anthropic-beta') value,
        ].join(','),
      ));
    }

    return _inner.send(request);
  }
}

/// Whether [value] holds a map that passes [test], at any depth.
bool _has(Object? value, bool Function(Map<Object?, Object?> map) test) {
  return switch (value) {
    Map() => test(value) || value.values.any((v) => _has(v, test)),
    List() => value.any((v) => _has(v, test)),
    _ => false,
  };
}

AgentTool _arithmetic(String name, String verb, num Function(num a, num b) apply) {
  return AgentTool(
    name: name,
    description: '$verb two numbers.',
    parameters: const {
      'type': 'object',
      'properties': {
        'a': {'type': 'number'},
        'b': {'type': 'number'},
      },
      'required': ['a', 'b'],
      'additionalProperties': false,
    },
    constrainedSampling: const JsonSchemaSampling(Strictness.require),
    execute: (id, args, signal, onUpdate) async {
      final json = args as Json;
      return AgentToolResult([TextContent('${apply(json['a'] as num, json['b'] as num)}')]);
    },
  );
}

void _check(String name, bool passed, AssistantMessage message) {
  _failed = _failed || !passed;
  stdout.writeln('${passed ? 'ok  ' : 'FAIL'} $name: ${message.stopReason.name}, "${_text(message)}"');
  if (message.errorMessage != null) {
    stdout.writeln('     ${message.errorMessage}');
  }
}

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
  _check('plain answer', _text(plain).toLowerCase().contains('pong'), plain);

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
  _check(
    'tool round trip',
    add.messages.whereType<ToolResultMessage>().isNotEmpty && _text(answer).contains('5555'),
    answer,
  );

  final thinking = Agent(model: model, level: ThinkingLevel.low, options: options, stream: models.streamSimple);
  await thinking.promptText('Is 391 prime? Answer yes or no.');
  final first = thinking.messages.last as AssistantMessage;
  _check(
    'adaptive thinking',
    first.content.whereType<ThinkingContent>().any((c) => c.signature?.isNotEmpty ?? false) &&
        _text(first).toLowerCase().startsWith('no'),
    first,
  );

  await thinking.promptText('And 397?');
  final second = thinking.messages.last as AssistantMessage;
  _check(
    'thinking sent back in a 2nd turn',
    second.stopReason == StopReason.stop && _text(second).toLowerCase().startsWith('yes'),
    second,
  );

  final budget = getBuiltinModel('anthropic', 'claude-haiku-4-5')!;
  final older = Agent(model: budget, level: ThinkingLevel.low, options: options, stream: models.streamSimple);
  await older.promptText('Is 391 prime? Answer yes or no.');
  final budgeted = older.messages.last as AssistantMessage;
  _check(
    'budget thinking (${budget.id})',
    budgeted.content.whereType<ThinkingContent>().any((c) => c.signature?.isNotEmpty ?? false) &&
        _text(budgeted).toLowerCase().startsWith('no'),
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
  _check('prompt caching', written.usage.cacheWrite > 0 && read.usage.cacheRead >= written.usage.cacheWrite, read);

  final image = await models.completeSimple(model, [
    UserMessage([
      const ImageContent(_red, 'image/png'),
      TextContent('What color is this image? Answer with one word.'),
    ], timestamp: 1),
  ], options);
  _check('image', _text(image).toLowerCase().contains('red'), image);

  final rejected = await models.completeSimple(model, [
    UserMessage([TextContent('hi')], timestamp: 1),
  ], const SimpleStreamOptions(apiKey: 'not-a-key', maxTokens: 100));
  _check(
    'real error (bad key)',
    rejected.stopReason == StopReason.error && (rejected.errorMessage?.isNotEmpty ?? false),
    rejected,
  );

  // Beta features, in one conversation: the second turn raises the effort and adds a tool and a system message.
  final recorder = _Recorder();
  final beta = Agent(
    model: model,
    systemPrompt: 'Use the tools for arithmetic, then answer with the number only.',
    tools: [_arithmetic('add', 'Adds', (a, b) => a + b)],
    level: ThinkingLevel.low,
    options: SimpleStreamOptions(
      apiKey: key,
      maxTokens: 4000,
      httpClient: recorder,
      sessionId: 'chorus-live-threads-${DateTime.now().millisecondsSinceEpoch}',
    ),
    stream: models.streamSimple,
  );
  await beta.promptText('What is 1234 + 4321?');
  final sum = beta.messages.last as AssistantMessage;
  final firstTurn = recorder.requests.length;
  _check(
    'strict tools',
    _has(recorder.requests.first.body['tools'], (t) => t['name'] == 'add' && t['strict'] == true) &&
        _text(sum).contains('5555'),
    sum,
  );

  beta
    ..level = ThinkingLevel.high
    ..tools = [...beta.tools, _arithmetic('multiply', 'Multiplies', (a, b) => a * b)]
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
    for (final m in changed.body['messages'] as List<Object?>)
      if (m case {'output_config': {'effort': final String effort}}) effort,
  ];
  stdout.writeln('     efforts in the 2nd turn: $efforts');
  _check(
    'effort changed mid-conversation',
    changed.betas.contains('mid-conversation-output-config') && efforts.contains('low') && efforts.last == 'high',
    product,
  );
  _check(
    'tool added mid-conversation',
    changed.betas.contains('inline-tools') &&
        _has(changed.body['messages'], (m) => m['type'] == 'tool_addition') &&
        beta.messages.whereType<ToolResultMessage>().any((r) => r.name == 'multiply') &&
        _text(product).contains('408'),
    product,
  );
  _check(
    'system message mid-conversation',
    _has(changed.body['messages'], (m) => m['role'] == 'system' && jsonEncode(m).contains('add the word done')) &&
        _text(product).toLowerCase().contains('done'),
    product,
  );

  final threads = [for (final r in recorder.requests) (r.body['thread'] as Map?)?['type']];
  final fallbacks = [
    for (final m in beta.messages.whereType<AssistantMessage>())
      ...?m.diagnostics?.where((d) => d.type == 'anthropic_thread_fallback'),
  ];
  stdout.writeln('     threads: $threads, fallbacks: ${fallbacks.length}');
  _check(
    'message threads',
    recorder.requests.every((r) => r.betas.contains('message-threads')) &&
        threads.first == 'create' &&
        threads.contains('continue') &&
        fallbacks.isEmpty,
    product,
  );

  final fable = getBuiltinModel('anthropic', 'claude-fable-5-2')!;
  final fallbackRecorder = _Recorder();
  final served = await models.completeSimple(fable, [
    UserMessage([TextContent('Reply with exactly: pong')], timestamp: 1),
  ], SimpleStreamOptions(apiKey: key, maxTokens: 200, httpClient: fallbackRecorder));
  final sent = fallbackRecorder.requests.single;
  stdout.writeln('     fallbacks sent: ${sent.body['fallbacks']}, answered by ${served.responseModel ?? fable.id}');
  _check(
    'server-side fallbacks accepted (${fable.id})',
    sent.betas.contains('server-side-fallback') &&
        sent.body['fallbacks'] is List &&
        _text(served).toLowerCase().contains('pong'),
    served,
  );

  exit(_failed ? 1 : 0);
}

/// A 16x16 red PNG.
const _red =
    'iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAAFklEQVR4nGO4IyJCEmIY1TCqYfhqAAACcQQQFwFJQgAAAABJRU5ErkJggg==';
