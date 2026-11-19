import 'types.dart';

/// Prices in dollars per million tokens.
class Rates {
  const Rates({required this.input, required this.output, required this.cacheRead, required this.cacheWrite});

  factory Rates.fromJson(Json json) {
    return Rates(
      input: (json['input'] as num).toDouble(),
      output: (json['output'] as num).toDouble(),
      cacheRead: (json['cacheRead'] as num).toDouble(),
      cacheWrite: (json['cacheWrite'] as num).toDouble(),
    );
  }

  final double input;
  final double output;
  final double cacheRead;
  final double cacheWrite;
}

/// Prices of a model, with tiers for large requests.
class ModelCost extends Rates {
  const ModelCost({
    required super.input,
    required super.output,
    required super.cacheRead,
    required super.cacheWrite,
    this.tiers = const [],
  });

  factory ModelCost.fromJson(Json json) {
    final rates = Rates.fromJson(json);
    return ModelCost(
      input: rates.input,
      output: rates.output,
      cacheRead: rates.cacheRead,
      cacheWrite: rates.cacheWrite,
      tiers: [for (final t in json['tiers'] as List<Object?>? ?? const []) Tier.fromJson(t as Json)],
    );
  }

  /// The tier with the highest threshold below a request's input tokens prices the whole request.
  final List<Tier> tiers;
}

class Tier extends Rates {
  const Tier({
    required this.above,
    required super.input,
    required super.output,
    required super.cacheRead,
    required super.cacheWrite,
  });

  factory Tier.fromJson(Json json) {
    final rates = Rates.fromJson(json);
    return Tier(
      above: json['inputTokensAbove'] as int,
      input: rates.input,
      output: rates.output,
      cacheRead: rates.cacheRead,
      cacheWrite: rates.cacheWrite,
    );
  }

  /// Applies when the request's input tokens exceed this.
  final int above;
}

/// A chat model, as pi's catalog describes it.
class Model {
  const Model({
    required this.id,
    required this.name,
    required this.api,
    required this.provider,
    required this.baseUrl,
    this.input = const ['text'],
    required this.cost,
    this.headers,
    this.reasoning = false,
    this.levels,
    required this.contextWindow,
    required this.maxTokens,
    this.compat = const {},
  });

  factory Model.fromJson(Json json) {
    return Model(
      id: json['id'] as String,
      name: json['name'] as String,
      api: json['api'] as String,
      provider: json['provider'] as String,
      baseUrl: json['baseUrl'] as String,
      input: (json['input'] as List<Object?>).cast<String>(),
      cost: ModelCost.fromJson(json['cost'] as Json),
      headers: (json['headers'] as Json?)?.cast<String, String>(),
      reasoning: json['reasoning'] as bool? ?? false,
      levels: (json['thinkingLevelMap'] as Json?)?.cast<String, String?>(),
      contextWindow: json['contextWindow'] as int,
      maxTokens: json['maxTokens'] as int,
      compat: json['compat'] as Json? ?? const {},
    );
  }

  final String id;
  final String name;

  /// The wire API, e.g. `anthropic-messages`.
  final String api;
  final String provider;
  final String baseUrl;

  /// `text`, and `image` when the model reads images.
  final List<String> input;
  final ModelCost cost;
  final Map<String, String>? headers;
  final bool reasoning;

  /// The provider value sent for each thinking level, by level name; `null` marks a level unsupported.
  final Map<String, String?>? levels;
  final int contextWindow;
  final int maxTokens;

  /// Settings for APIs whose providers differ, read by the API's own code (pi's `compat`).
  final Json compat;
}

/// Sets [usage]'s cost from [model]'s prices; Anthropic charges twice the input price for 1h cache writes.
Cost calculateCost(Model model, Usage usage) {
  Rates rates = model.cost;
  var threshold = -1;
  for (final tier in model.cost.tiers) {
    if (usage.input + usage.cacheRead + usage.cacheWrite > tier.above && tier.above > threshold) {
      rates = tier;
      threshold = tier.above;
    }
  }

  final long = usage.cacheWrite1h ?? 0;
  final cost = usage.cost
    ..input = rates.input / 1e6 * usage.input
    ..output = rates.output / 1e6 * usage.output
    ..cacheRead = rates.cacheRead / 1e6 * usage.cacheRead
    ..cacheWrite = (rates.cacheWrite * (usage.cacheWrite - long) + rates.input * 2 * long) / 1e6
    ..billed = false;
  cost.total = cost.input + cost.output + cost.cacheRead + cost.cacheWrite;
  return cost;
}
