// Strict JSON-schema tools: a tool schema converted to the subset providers sample strictly, and the decision whether
// a tool is sent strict. Ported from pi's api/constrained-sampling.ts, without OpenAI's grammar tools.

import 'dart:convert';

import 'types.dart';

/// A schema the strict subset cannot express.
class UnsupportedStrictSchema implements Exception {
  const UnsupportedStrictSchema(this.message);

  final String message;

  @override
  String toString() => message;
}

/// Whether a provider's strict mode rejects this schema keyword with this value.
typedef UnsupportedKeyword = bool Function(String key, Object? value);

const _unsupportedKeys = [
  r'$ref',
  r'$defs',
  'definitions',
  'allOf',
  'oneOf',
  'patternProperties',
  'dependentSchemas',
  'dependencies',
  'unevaluatedProperties',
  'propertyNames',
  'contains',
  'prefixItems',
  'not',
  'if',
  'then',
  'else',
];

bool _structured(Object? schema) {
  if (schema is! Map) {
    return false;
  }

  final type = schema['type'];
  final types = switch (type) {
    final String t => [t],
    final List<Object?> list => list,
    _ => const <Object?>[],
  };
  return types.contains('object') || types.contains('array') || schema['properties'] != null || schema['items'] != null;
}

bool _allowsNull(Object? schema) {
  if (schema is! Map) {
    return false;
  }

  final type = schema['type'];
  if (type == 'null' || (type is List && type.contains('null'))) {
    return true;
  }

  if (schema.containsKey('const') && schema['const'] == null) {
    return true;
  }

  final values = schema['enum'];
  if (values is List && values.contains(null)) {
    return true;
  }

  final any = schema['anyOf'];
  return any is List && any.any(_allowsNull);
}

void _makeStrict(Object? schema, UnsupportedKeyword? unsupported) {
  if (schema is! Map<String, Object?>) {
    throw const UnsupportedStrictSchema('boolean schemas are unsupported');
  }

  for (final key in _unsupportedKeys) {
    if (schema[key] != null) {
      throw UnsupportedStrictSchema('$key schemas are unsupported');
    }
  }

  if (unsupported != null) {
    for (final MapEntry(:key, :value) in schema.entries) {
      if (unsupported(key, value)) {
        throw UnsupportedStrictSchema('$key: ${jsonEncode(value)} is unsupported');
      }
    }
  }

  // OpenAI strict mode rejects nodes without a type; a bare enum counts as typed, a bare const does not.
  if (schema['type'] == null && schema['anyOf'] == null && schema['enum'] == null) {
    throw const UnsupportedStrictSchema('schemas without a type are unsupported');
  }

  final any = schema['anyOf'];
  if (any != null) {
    if (any is! List || any.isEmpty) {
      throw const UnsupportedStrictSchema('anyOf must contain at least one schema');
    }

    for (final variant in any) {
      if (_structured(variant)) {
        throw const UnsupportedStrictSchema('object and array unions are unsupported');
      }

      _makeStrict(variant, unsupported);
    }
  }

  final items = schema['items'];
  if (items != null) {
    if (items is List) {
      throw const UnsupportedStrictSchema('tuple schemas are unsupported');
    }

    _makeStrict(items, unsupported);
  }

  final object = schema['type'] == 'object';
  if (schema['properties'] != null && !object) {
    throw const UnsupportedStrictSchema('properties require type object');
  }

  if (!object) {
    return;
  }

  final additional = schema['additionalProperties'];
  if (additional != null && additional != false) {
    throw const UnsupportedStrictSchema('schema-valued or true additionalProperties is unsupported');
  }

  final properties = schema['properties'] ?? <String, Object?>{};
  if (properties is! Map<String, Object?>) {
    throw const UnsupportedStrictSchema('object properties must be a schema map');
  }

  final required = schema['required'] ?? const <Object?>[];
  if (required is! List || required.any((k) => k is! String)) {
    throw const UnsupportedStrictSchema('object required must be a string array');
  }

  if (required.any((k) => !properties.containsKey(k))) {
    throw const UnsupportedStrictSchema('required contains an unknown property');
  }

  for (final MapEntry(:key, :value) in properties.entries.toList()) {
    _makeStrict(value, unsupported);

    // An optional property becomes required and nullable: strict mode requires every property.
    if (!required.contains(key) && !_allowsNull(value)) {
      properties[key] = {
        'anyOf': [
          value,
          {'type': 'null'},
        ],
      };
    }
  }

  schema['properties'] = properties;
  schema['required'] = properties.keys.toList();
  schema['additionalProperties'] = false;
}

/// [schema] converted to the strict subset providers sample exactly: every property required (optional ones made
/// nullable), no other properties. The schema itself is not changed. Throws [UnsupportedStrictSchema].
Json makeStrictJsonSchema(Json schema, [UnsupportedKeyword? unsupported]) {
  final copy = jsonDecode(jsonEncode(schema));
  _makeStrict(copy, unsupported);
  if ((copy as Json)['type'] != 'object') {
    throw const UnsupportedStrictSchema('root schema must have type object');
  }

  return copy;
}

/// Whether [tool] asks for strict JSON-schema sampling only where possible.
bool prefersStrictSampling(Tool tool) {
  return switch (tool.constrainedSampling) {
    JsonSchemaSampling(strict: Strictness.prefer) => true,
    _ => false,
  };
}

/// Whether [tool] is sent strict: true when it asks for strict sampling and its schema converts, null when it is
/// sent as an ordinary tool. Throws a StateError for a tool that requires strict sampling and cannot have it.
/// [unsupported] adds the keywords a provider's strict mode rejects, so `prefer` tools with them fall back.
bool? resolveStrictSampling(Tool tool, bool supported, [UnsupportedKeyword? unsupported]) {
  final config = tool.constrainedSampling;
  if (config is! JsonSchemaSampling) {
    return null;
  }

  if (!supported) {
    if (config.strict == Strictness.require) {
      throw StateError(
        'Tool "${tool.name}" requires JSON-schema constrained sampling, but strict tools are unsupported.',
      );
    }

    return null;
  }

  try {
    makeStrictJsonSchema(tool.parameters, unsupported);
    return true;
  } on UnsupportedStrictSchema catch (error) {
    if (config.strict == Strictness.require) {
      throw StateError('Tool "${tool.name}" requires JSON-schema constrained sampling, but $error.');
    }

    return null;
  }
}

/// The parameters to send for [tool]: its strict schema when [strict], else its own.
Json strictToolParameters(Tool tool, bool? strict) =>
    strict == true ? makeStrictJsonSchema(tool.parameters) : tool.parameters;
