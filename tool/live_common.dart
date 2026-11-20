/// What the live check scripts share: printing checks, recording requests, and the tools and image they use.
library;

import 'dart:convert';
import 'dart:io';

import 'package:chorus/chorus.dart';
import 'package:http/http.dart' as http;

/// The text of [message], trimmed.
String text(AssistantMessage message) => message.content.whereType<TextContent>().map((c) => c.text).join().trim();

/// Whether a check failed.
var failed = false;

/// Prints a check's outcome with the reply it judged.
void check(String name, bool passed, AssistantMessage message) {
  failed = failed || !passed;
  stdout.writeln('${passed ? 'ok  ' : 'FAIL'} $name: ${message.stopReason.name}, "${text(message)}"');
  if (message.errorMessage != null) {
    stdout.writeln('     ${message.errorMessage}');
  }
}

/// Keeps the body, the headers and the Anthropic beta header of every request it sends.
class Recorder extends http.BaseClient {
  final _inner = http.Client();
  final requests = <({Json body, Map<String, String> headers, String betas})>[];

  @override
  Future<http.StreamedResponse> send(http.BaseRequest request) {
    if (request is http.Request) {
      final headers = {for (final MapEntry(:key, :value) in request.headers.entries) key.toLowerCase(): value};
      requests.add((body: jsonDecode(request.body) as Json, headers: headers, betas: headers['anthropic-beta'] ?? ''));
    }

    return _inner.send(request);
  }
}

/// Whether [value] holds a map that passes [test], at any depth.
bool has(Object? value, bool Function(Map<Object?, Object?> map) test) {
  return switch (value) {
    Map() => test(value) || value.values.any((v) => has(v, test)),
    List() => value.any((v) => has(v, test)),
    _ => false,
  };
}

/// A strict tool that applies [apply] to two numbers.
AgentTool arithmetic(String name, String verb, num Function(num a, num b) apply) {
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

/// A 16x16 red PNG.
const redPng =
    'iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAAFklEQVR4nGO4IyJCEmIY1TCqYfhqAAACcQQQFwFJQgAAAABJRU5ErkJggg==';
