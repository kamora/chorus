// Tool call arguments checked against the tool's JSON Schema with pluto, coerced first: models often send "42" for a
// number, a single value for an array, or null for an omitted optional property.

import 'dart:convert';

import 'package:pluto/formats.dart';
import 'package:pluto/pluto.dart';

import 'types.dart';

// One compiled validator per schema object: tool schemas are long-lived, and the Expando drops entries with them.
final _validators = Expando<Validator>();

// A $ref that names nothing (a dangling definition in an MCP server's schema) rejects only values at its place, instead
// of making every call of the tool fail.
Validator _validator(Json schema) {
  return _validators[schema] ??= compile(schema, formats: formats, rejectUnresolved: true);
}

/// A pluto error path (a JSON Pointer) as the dotted path the model sees: `/edits/0/oldText` is `edits.0.oldText`.
String _path(String pointer) {
  if (pointer.isEmpty) {
    return 'root';
  }

  return pointer.substring(1).split('/').map((s) => s.replaceAll('~1', '/').replaceAll('~0', '~')).join('.');
}

/// Finds the call's tool by name and validates the call's arguments against its schema. Throws when there is no such
/// tool or the arguments are invalid.
Object? validateToolCall(List<Tool> tools, ToolCall call) {
  final tool = tools.where((t) => t.name == call.name).firstOrNull;
  if (tool == null) {
    throw StateError('Tool "${call.name}" not found');
  }

  return validateToolArguments(tool, call);
}

/// The call's arguments, coerced and validated against the tool's JSON Schema: always a copy, so a tool that changes
/// its arguments never changes the call in the transcript. Throws a StateError for the model when they are invalid:
/// each error as `  - path: message`, then the arguments received.
Object? validateToolArguments(Tool tool, ToolCall call) {
  if (call.argumentsError != null) {
    throw StateError('Invalid arguments for tool "${call.name}": ${call.argumentsError}');
  }

  final result = _validator(tool.parameters).parse(jsonDecode(jsonEncode(call.arguments)));
  if (result case Valid(:final value)) {
    return value;
  }

  final errors = (result as Invalid).errors.map((e) => '  - ${_path(e.path)}: ${e.message}').join('\n');
  throw StateError(
    'Validation failed for tool "${call.name}":\n$errors\n\n'
    'Received arguments:\n${const JsonEncoder.withIndent('  ').convert(call.arguments)}',
  );
}
