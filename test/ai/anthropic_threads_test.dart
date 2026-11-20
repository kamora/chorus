// Message threads. Ported from pi's anthropic-message-threads.test.ts, without its Claude Code OAuth test (chorus does
// not pose as Claude Code).

import 'dart:async';
import 'dart:convert';

import 'package:anthropic_sdk_lite_dart/anthropic_sdk_lite_dart.dart';
import 'package:chorus/chorus.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:test/test.dart';

const threadsBeta = 'message-threads-2026-08-12';
const grammarError =
    'The compiled grammar is too large, which would cause performance issues. Simplify your tool schemas or reduce '
    'the number of strict tools.';

/// A request as the fake endpoint saw it: the body and the beta list.
typedef Captured = ({Json body, List<String> betas});

/// How the fake endpoint answers one request.
typedef Responder = http.Response Function(Captured request);

String events(List<Json> list) => list.map((e) => 'event: ${e['type']}\ndata: ${jsonEncode(e)}\n\n').join();

http.Response sseResponse(String body) => http.Response(body, 200, headers: {'content-type': 'text/event-stream'});

/// A reply: optional signed thinking, the text "answer", tool calls; or one that ends with an error event or without
/// `message_stop`.
http.Response reply(
  Captured request, {
  String? id,
  String? model,
  String? thinking,
  List<({String id, String name, Json input})> calls = const [],
  String? stopReason,
  String? end,
}) {
  final list = <Json>[
    {
      'type': 'message_start',
      'message': {
        'id': id ?? 'msg_x',
        'model': model ?? request.body['model'],
        'usage': {'input_tokens': 10, 'output_tokens': 0},
      },
    },
  ];
  var index = 0;
  if (thinking != null) {
    list.addAll([
      {
        'type': 'content_block_start',
        'index': index,
        'content_block': {'type': 'thinking', 'thinking': '', 'signature': ''},
      },
      {
        'type': 'content_block_delta',
        'index': index,
        'delta': {'type': 'thinking_delta', 'thinking': thinking},
      },
      {
        'type': 'content_block_delta',
        'index': index,
        'delta': {'type': 'signature_delta', 'signature': 'sig'},
      },
      {'type': 'content_block_stop', 'index': index},
    ]);
    index++;
  }

  list.addAll([
    {
      'type': 'content_block_start',
      'index': index,
      'content_block': {'type': 'text', 'text': ''},
    },
    {
      'type': 'content_block_delta',
      'index': index,
      'delta': {'type': 'text_delta', 'text': 'answer'},
    },
    {'type': 'content_block_stop', 'index': index},
  ]);
  index++;
  for (final call in calls) {
    list.addAll([
      {
        'type': 'content_block_start',
        'index': index,
        'content_block': {'type': 'tool_use', 'id': call.id, 'name': call.name, 'input': <String, Object?>{}},
      },
      {
        'type': 'content_block_delta',
        'index': index,
        'delta': {'type': 'input_json_delta', 'partial_json': jsonEncode(call.input)},
      },
      {'type': 'content_block_stop', 'index': index},
    ]);
    index++;
  }

  if (end == 'error_event') {
    return sseResponse(
      events(list) +
          events([
            {
              'type': 'error',
              'error': {'type': 'overloaded_error', 'message': 'Overloaded'},
            },
          ]),
    );
  }

  list.add({
    'type': 'message_delta',
    'delta': {'stop_reason': stopReason ?? (calls.isEmpty ? 'end_turn' : 'tool_use')},
    'usage': {'output_tokens': 1},
  });
  if (end != 'no_message_stop') {
    list.add({'type': 'message_stop'});
  }

  return sseResponse(events(list));
}

http.Response errorResponse(int status, String message, [String? code, Map<String, String> headers = const {}]) {
  final type = switch (status) {
    404 => 'not_found_error',
    429 => 'rate_limit_error',
    _ => 'invalid_request_error',
  };
  return http.Response(
    jsonEncode({
      'type': 'error',
      'error': {
        'type': type,
        'message': message,
        if (code != null) 'details': {'error_code': code},
      },
    }),
    status,
    headers: headers,
  );
}

/// The fields Anthropic requires to stay the same within a thread (observed live).
String fingerprint(Captured request) {
  final b = request.body;
  return jsonEncode([
    b['model'],
    b['thinking'],
    b['output_config'],
    b['tool_choice'],
    b.containsKey('fallbacks'),
    request.betas,
  ]);
}

var sessions = 0;

/// A fake Anthropic endpoint: records request bodies and beta lists, and answers from [queue] or with a plain reply.
/// Like the real server, it rejects a `continue` whose fixed fields differ from the thread's.
class Harness {
  final requests = <Captured>[];
  final queue = <Responder>[];
  final _fingerprints = <String, String>{};
  final session = 'thread-test-${++sessions}';

  Anthropic client() {
    return Anthropic(
      apiKey: 'test-key',
      baseURL: 'https://api.anthropic.com',
      httpClient: MockClient((r) async {
        final beta = r.headers['anthropic-beta'];
        final request = (body: jsonDecode(r.body) as Json, betas: beta == null ? <String>[] : beta.split(','));
        requests.add(request);
        if (queue.isNotEmpty) {
          return queue.removeAt(0)(request);
        }

        final id = 'msg_${requests.length}';
        final previous = (request.body['thread'] as Json?)?['previous_message_id'] as String?;
        final expected = previous == null ? fingerprint(request) : _fingerprints[previous];
        if (expected != fingerprint(request)) {
          return errorResponse(400, 'fixed field changed', 'thread_fingerprint_mismatch');
        }

        if (request.body.containsKey('thread')) {
          _fingerprints[id] = expected!;
        }

        return reply(request, id: id);
      }),
    );
  }
}

final baseModel = Model(
  id: 'claude-test',
  name: 'Test Claude',
  api: 'anthropic-messages',
  provider: 'anthropic',
  baseUrl: 'https://api.anthropic.com',
  cost: const ModelCost(input: 0, output: 0, cacheRead: 0, cacheWrite: 0),
  contextWindow: 200000,
  maxTokens: 100,
);

Model variant({String? id, String? provider, String? baseUrl, bool reasoning = false, Json compat = const {}}) {
  return Model(
    id: id ?? baseModel.id,
    name: baseModel.name,
    api: baseModel.api,
    provider: provider ?? baseModel.provider,
    baseUrl: baseUrl ?? baseModel.baseUrl,
    reasoning: reasoning,
    cost: baseModel.cost,
    contextWindow: baseModel.contextWindow,
    maxTokens: baseModel.maxTokens,
    compat: compat,
  );
}

Tool tool(String name, [ConstrainedSampling? sampling]) {
  return Tool(name, 'Tool $name', const {
    'type': 'object',
    'properties': {
      'value': {'type': 'string'},
    },
    'required': ['value'],
  }, constrainedSampling: sampling);
}

UserMessage user(String text) => UserMessage([TextContent(text)], timestamp: 1);

ToolResultMessage toolResult(String id, String name) {
  return ToolResultMessage(id: id, name: name, content: [TextContent('result')], isError: false, timestamp: 1);
}

/// The options of one turn; the harness's client and session are added.
typedef Turn = ({
  Model? model,
  String? systemPrompt,
  List<Tool>? tools,
  AnthropicOptions Function(AnthropicOptions base)? options,
});

const Turn plain = (model: null, systemPrompt: null, tools: null, options: null);

Future<AssistantMessage> turn(Harness h, List<Message> messages, [Turn input = plain]) {
  final base = AnthropicOptions(client: h.client(), sessionId: h.session, maxRetries: 0);
  return streamAnthropic(input.model ?? baseModel, [
    SystemMessage(
      [TextContent(input.systemPrompt ?? 'You are a test assistant.')],
      added: input.tools ?? [tool('lookup')],
      timestamp: 0,
    ),
    ...messages,
  ], input.options?.call(base) ?? base).result;
}

/// Runs one turn and appends the reply to the conversation.
Future<AssistantMessage> say(Harness h, List<Message> messages, [Turn input = plain]) async {
  final result = await turn(h, messages, input);
  messages.add(result);
  return result;
}

Json? threadOf(Captured request) => request.body['thread'] as Json?;

/// A copy of [base] with other settings.
AnthropicOptions Function(AnthropicOptions) withOptions({
  Object? toolChoice,
  CacheRetention? cacheRetention,
  bool? thinkingEnabled,
  int? thinkingBudgetTokens,
  String? effort,
  int? maxRetries,
  int? maxTokens,
  bool? disableContinuation,
  Map<String, String?>? headers,
  AbortSignal? signal,
  FutureOr<Json?> Function(Json payload, Model model)? onPayload,
  FutureOr<void> Function(Json event, Model model)? onEvent,
}) {
  return (base) => AnthropicOptions(
    client: base.client,
    sessionId: base.sessionId,
    maxRetries: maxRetries ?? base.maxRetries,
    toolChoice: toolChoice,
    cacheRetention: cacheRetention,
    thinkingEnabled: thinkingEnabled,
    thinkingBudgetTokens: thinkingBudgetTokens,
    effort: effort,
    maxTokens: maxTokens,
    disableContinuation: disableContinuation,
    headers: headers,
    signal: signal,
    onPayload: onPayload,
    onEvent: onEvent,
  );
}

void main() {
  tearDown(cleanupSessionResources);

  group('Anthropic message threads', () {
    test('creates a thread, then continues it with only the new messages', () async {
      final h = Harness();
      final messages = <Message>[user('first')];
      await say(h, messages);
      messages.add(user('second'));
      final result = await say(h, messages);

      expect(result.stopReason, StopReason.stop);
      final [create, next] = h.requests;
      expect(threadOf(create), {'type': 'create'});
      expect(create.body.containsKey('system'), isTrue);
      expect(create.body.containsKey('tools'), isTrue);
      expect(threadOf(next), {'type': 'continue', 'previous_message_id': 'msg_1'});
      expect(next.body.containsKey('system'), isFalse);
      expect(next.body.containsKey('tools'), isFalse);

      // The cache marker moved from the first user message to the new one.
      expect((create.body['messages'] as List<Object?>).first, {
        'role': 'user',
        'content': [
          {
            'type': 'text',
            'text': 'first',
            'cache_control': {'type': 'ephemeral'},
          },
        ],
      });
      expect(next.body['messages'], [
        {
          'role': 'user',
          'content': [
            {
              'type': 'text',
              'text': 'second',
              'cache_control': {'type': 'ephemeral'},
            },
          ],
        },
      ]);
      expect(create.betas, contains(threadsBeta));
      expect(next.betas, create.betas);
    });

    test('continues after a reply that ends in tool calls, without placeholder results in the prefix', () async {
      final h = Harness();
      h.queue.add(
        (r) => reply(
          r,
          id: 'msg_tool',
          calls: [
            (id: 'toolu_1', name: 'lookup', input: {'value': 'x'}),
          ],
        ),
      );
      final messages = <Message>[user('look it up')];
      final first = await say(h, messages);
      expect(first.stopReason, StopReason.toolUse);
      messages.add(toolResult('toolu_1', 'lookup'));
      await say(h, messages);

      final next = h.requests[1];
      expect(threadOf(next), {'type': 'continue', 'previous_message_id': 'msg_tool'});
      final sent = (next.body['messages'] as List<Object?>).cast<Json>();
      expect(sent, hasLength(1));
      expect(sent.single['role'], 'user');
      final block = (sent.single['content'] as List<Object?>).first as Json;
      expect((block['type'], block['tool_use_id']), ('tool_result', 'toolu_1'));
    });

    test('continues when the effort level changes between turns', () async {
      final model = variant(
        reasoning: true,
        compat: const {'forceAdaptiveThinking': true, 'supportsMidConvoEffort': true},
      );
      final h = Harness();
      h.queue.add((r) => reply(r, id: 'msg_a', thinking: 'thought'));
      final messages = <Message>[user('first')];
      await say(h, messages, (
        model: model,
        systemPrompt: null,
        tools: null,
        options: withOptions(thinkingEnabled: true, effort: 'high'),
      ));
      messages.add(user('second'));
      await say(h, messages, (
        model: model,
        systemPrompt: null,
        tools: null,
        options: withOptions(thinkingEnabled: true, effort: 'low'),
      ));

      List<Object?> shape(Captured r) => [
        for (final m in (r.body['messages'] as List<Object?>).cast<Json>())
          (m['output_config'] as Json?)?['effort'] ?? m['role'],
      ];
      expect(shape(h.requests[0]), ['user', 'high']);
      expect(threadOf(h.requests[1]), {'type': 'continue', 'previous_message_id': 'msg_a'});
      expect(shape(h.requests[1]), ['user', 'low']);
    });

    test('continues when a system message with a native tool change follows the reply', () async {
      final model = variant(
        compat: const {'supportsMidConvoSystemMessages': true, 'supportsMidConvoToolChanges': true},
      );
      final h = Harness();
      final messages = <Message>[user('first')];
      final input = (model: model, systemPrompt: null, tools: null, options: null);
      await say(h, messages, input);
      messages.addAll([
        SystemMessage([TextContent('Update')], added: [tool('other')], timestamp: 2),
        user('second'),
      ]);
      await say(h, messages, input);

      final next = h.requests[1];
      expect(threadOf(next)?['type'], 'continue');
      final sent = (next.body['messages'] as List<Object?>).cast<Json>();
      expect(sent.map((m) => m['role']), ['user', 'system']);
      expect(jsonEncode(sent[1]['content']), contains('tool_addition'));
    });

    for (final (name, before, after) in <(String, Turn, Turn)>[
      ('system prompt', plain, (model: null, systemPrompt: 'Different prompt.', tools: null, options: null)),
      ('tools', plain, (model: null, systemPrompt: null, tools: [tool('lookup'), tool('other')], options: null)),
      ('model', plain, (model: variant(id: 'claude-other'), systemPrompt: null, tools: null, options: null)),
      ('tool choice', plain, (model: null, systemPrompt: null, tools: null, options: withOptions(toolChoice: 'any'))),
      (
        'cache retention',
        plain,
        (model: null, systemPrompt: null, tools: null, options: withOptions(cacheRetention: CacheRetention.long)),
      ),
      (
        'thinking level',
        (
          model: variant(reasoning: true),
          systemPrompt: null,
          tools: null,
          options: withOptions(thinkingEnabled: true, thinkingBudgetTokens: 1024),
        ),
        (
          model: variant(reasoning: true),
          systemPrompt: null,
          tools: null,
          options: withOptions(thinkingEnabled: true, thinkingBudgetTokens: 2048),
        ),
      ),
    ]) {
      test('creates a new thread when the $name changes', () async {
        final h = Harness();
        final messages = <Message>[user('first')];
        await say(h, messages, before);
        messages.add(user('second'));
        await say(h, messages, after);
        expect(threadOf(h.requests[1]), {'type': 'create'});
        expect(h.requests[1].body['messages'], hasLength(3));
      });
    }

    test('creates a new thread when the request has no messages beyond the stored prefix', () async {
      final h = Harness();
      final messages = <Message>[user('first')];
      await say(h, messages);
      await turn(h, messages);
      expect(threadOf(h.requests[1]), {'type': 'create'});
    });

    for (final (code, status) in [
      ('thread_not_found', 404),
      ('thread_already_continued', 400),
      ('thread_continue_in_flight', 400),
      ('thread_fingerprint_mismatch', 400),
    ]) {
      test('starts a new thread with the full request after $code', () async {
        final h = Harness();
        final messages = <Message>[user('first')];
        await say(h, messages);
        messages.add(user('second'));
        h.queue.add((_) => errorResponse(status, 'thread rejected', code));
        var payloads = 0;
        final result = await say(h, messages, (
          model: null,
          systemPrompt: null,
          tools: null,
          options: withOptions(
            onPayload: (_, _) {
              payloads++;
              return null;
            },
          ),
        ));

        expect(result.stopReason, StopReason.stop);
        expect(payloads, 1);
        expect(h.requests, hasLength(3));
        final [_, rejected, resend] = h.requests;
        expect(threadOf(rejected)?['type'], 'continue');
        expect(threadOf(resend), {'type': 'create'});
        expect(resend.body['messages'], hasLength(3));
        expect(resend.body.containsKey('system'), isTrue);
        expect(resend.betas, rejected.betas);
        expect(result.diagnostics, hasLength(1));
        expect(result.diagnostics!.single.type, 'anthropic_thread_fallback');
        expect(result.diagnostics!.single.details, {'reason': code});

        messages.add(user('third'));
        await say(h, messages);
        expect(threadOf(h.requests[3]), {'type': 'continue', 'previous_message_id': 'msg_3'});
      });
    }

    for (final (name, code, message) in [
      ('thread_unsupported_request', 'thread_unsupported_request', 'rejected'),
      ('a 400 without error code about the thread field', null, 'thread: Extra inputs are not permitted'),
    ]) {
      test('turns threads off for the session after $name on continue', () async {
        final h = Harness();
        final messages = <Message>[user('first')];
        await say(h, messages);
        messages.add(user('second'));
        h.queue.add((_) => errorResponse(400, message, code));
        final result = await say(h, messages);

        expect(result.stopReason, StopReason.stop);
        expect(result.diagnostics?.first.type, 'anthropic_thread_fallback');
        expect(result.diagnostics?.first.details, {'reason': code ?? '400'});
        messages.add(user('third'));
        await say(h, messages);
        final later = h.requests[3];
        expect(later.body.containsKey('thread'), isFalse);

        // The resend was accepted with the threads beta, so the beta list stays the same.
        expect(later.betas, h.requests[0].betas);
      });
    }

    test('shows the error when the resend fails too and keeps threads on', () async {
      final h = Harness();
      final messages = <Message>[user('first')];
      await say(h, messages);
      messages.add(user('second'));
      h.queue.addAll([
        (_) => errorResponse(400, 'thread: Extra inputs are not permitted'),
        (_) => errorResponse(400, 'Real problem'),
      ]);
      final failed = await turn(h, messages);
      expect(failed.stopReason, StopReason.error);
      expect(failed.errorMessage, contains('Real problem'));
      expect(h.requests, hasLength(3));

      await turn(h, messages);
      expect(threadOf(h.requests[3]), {'type': 'create'});
    });

    for (final (status, message) in [
      (400, 'prompt is too long: 250000 tokens > 200000 maximum'),
      (404, 'model: claude-test'),
    ]) {
      test('passes a $status that is not about the thread on without resending', () async {
        final h = Harness();
        final messages = <Message>[user('first')];
        await say(h, messages);
        messages.add(user('second'));
        h.queue.add((_) => errorResponse(status, message));
        final failed = await turn(h, messages);
        expect(failed.stopReason, StopReason.error);
        expect(failed.errorMessage, contains(message));
        expect(h.requests, hasLength(2));
        expect(failed.diagnostics, isNull);

        // Threads stay on: after a fix such as compaction, the next request starts a new thread.
        await turn(h, messages);
        expect(threadOf(h.requests[2]), {'type': 'create'});
      });
    }

    test('leaves strict tool rejections to the strict tool fallback', () async {
      final h = Harness();
      h.queue.add((_) => errorResponse(400, grammarError));
      final result = await turn(
        h,
        [user('first')],
        (
          model: variant(compat: const {'supportsStrictTools': true}),
          systemPrompt: null,
          tools: [tool('lookup', const JsonSchemaSampling(Strictness.prefer))],
          options: null,
        ),
      );
      expect(result.stopReason, StopReason.stop);
      expect(h.requests.map((r) => threadOf(r)?['type']), ['create', 'create']);
      expect(result.diagnostics?.map((d) => d.type), ['anthropic_strict_tools_fallback']);
    });

    test('starts a new thread when a retried continue finds the thread in flight', () async {
      final h = Harness();
      final messages = <Message>[user('first')];
      await say(h, messages);
      messages.add(user('second'));
      h.queue.addAll([
        (_) => errorResponse(503, 'overloaded', null, {'retry-after-ms': '0'}),
        (_) => errorResponse(400, 'in flight', 'thread_continue_in_flight'),
      ]);
      final result = await say(h, messages, (
        model: null,
        systemPrompt: null,
        tools: null,
        options: withOptions(maxRetries: 1),
      ));
      expect(result.stopReason, StopReason.stop);
      expect(h.requests.map((r) => threadOf(r)?['type']), ['create', 'continue', 'continue', 'create']);
    });

    for (final (name, header) in [
      ('without the threads beta', 'fine-grained-tool-streaming-2025-05-14'),
      ('set to null', null),
    ]) {
      test('fails once, then turns threads off, with an explicit anthropic-beta header $name', () async {
        final h = Harness();
        http.Response respond(Captured r) {
          if (r.body.containsKey('thread') && !r.betas.contains(threadsBeta)) {
            return errorResponse(400, 'thread: Extra inputs are not permitted');
          }

          return reply(r, id: 'msg_${h.requests.length}');
        }

        h.queue.addAll([respond, respond, respond]);
        final input = (
          model: null,
          systemPrompt: null,
          tools: null,
          options: withOptions(headers: {'anthropic-beta': header}),
        );
        final messages = <Message>[user('first')];
        expect((await say(h, messages, input)).stopReason, StopReason.stop);
        messages.add(user('second'));
        await say(h, messages, input);
        expect(h.requests.map((r) => threadOf(r)?['type']), ['create', null, null]);
      });
    }

    for (final (name, spec) in [
      ('an aborted reply', 'abort'),
      ('a stream error', 'error_event'),
      ('a stream that ends early', 'no_message_stop'),
      ('a refusal', 'refusal'),
      ('a fallback model reply', 'fallback'),
    ]) {
      test('does not record $name', () async {
        final h = Harness();
        final signal = AbortSignal();
        if (spec != 'abort') {
          h.queue.add(
            (r) => reply(
              r,
              id: 'msg_bad',
              end: switch (spec) {
                'error_event' || 'no_message_stop' => spec,
                _ => null,
              },
              stopReason: spec == 'refusal' ? 'refusal' : null,
              model: spec == 'fallback' ? 'claude-fallback' : null,
            ),
          );
        }

        final messages = <Message>[user('first')];
        final first = await turn(h, messages, (
          model: null,
          systemPrompt: null,
          tools: null,
          options: spec == 'abort'
              ? withOptions(
                  signal: signal,
                  onEvent: (event, _) {
                    if (event['type'] == 'message_start') {
                      signal.abort();
                    }
                  },
                )
              : null,
        ));
        messages.addAll([first, user('second')]);
        await turn(h, messages);
        expect(threadOf(h.requests[1]), {'type': 'create'});
      });
    }

    test('does not use threads with cacheRetention none', () async {
      final h = Harness();
      await say(
        h,
        [user('first')],
        (model: null, systemPrompt: null, tools: null, options: withOptions(cacheRetention: CacheRetention.none)),
      );
      expect(h.requests[0].body.containsKey('thread'), isFalse);
      expect(h.requests[0].betas, isNot(contains(threadsBeta)));
    });

    // pi #11987
    test('uses threads for any provider on api.anthropic.com, not for other endpoints', () async {
      final alias = Harness();
      final aliasInput = (model: variant(provider: 'anthropic-1-api'), systemPrompt: null, tools: null, options: null);
      final messages = <Message>[user('first')];
      await say(alias, messages, aliasInput);
      messages.add(user('second'));
      await say(alias, messages, aliasInput);
      expect(threadOf(alias.requests[0]), {'type': 'create'});
      expect(threadOf(alias.requests[1]), {'type': 'continue', 'previous_message_id': 'msg_1'});

      final gateway = Harness();
      await say(
        gateway,
        [user('first')],
        (model: variant(baseUrl: 'https://gateway.example.com'), systemPrompt: null, tools: null, options: null),
      );
      expect(gateway.requests[0].body.containsKey('thread'), isFalse);
      expect(gateway.requests[0].betas, isNot(contains(threadsBeta)));
    });

    test('sends cache warming replays without a thread and keeps the conversation\'s thread', () async {
      final h = Harness();
      final messages = <Message>[user('first')];
      await say(h, messages);
      await turn(h, messages.sublist(0, 1), (
        model: null,
        systemPrompt: null,
        tools: null,
        options: withOptions(maxTokens: 1, disableContinuation: true),
      ));
      messages.add(user('second'));
      await say(h, messages);

      final [real, warm, next] = h.requests;
      expect(warm.body.containsKey('thread'), isFalse);
      expect(warm.body['max_tokens'], 1);
      expect(warm.body['messages'], real.body['messages']);
      expect(warm.betas, real.betas);
      expect(threadOf(next), {'type': 'continue', 'previous_message_id': 'msg_1'});
    });
  });
}
