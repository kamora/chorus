// Providers and the collection that routes requests to them, as pi's createModels() and createProvider() without auth
// and dynamic model lists (chorus takes keys as request options), and the read accessors of pi's built-in catalog.

import 'event_stream.dart';
import 'model.dart';
import 'models.generated.dart';
import 'options.dart';
import 'providers/anthropic.dart';
import 'types.dart';

/// Streams one API: [stream] with the API's own options, [streamSimple] with the provider-neutral ones.
class ProviderStreams {
  const ProviderStreams({required this.stream, required this.streamSimple});

  final AssistantMessageEventStream Function(Model model, List<Message> messages, StreamOptions options) stream;
  final AssistantMessageEventStream Function(Model model, List<Message> messages, SimpleStreamOptions options)
  streamSimple;
}

/// A provider: its models and the APIs they are served on, by API id (`anthropic-messages`, ...).
class Provider {
  Provider({required this.id, String? name, this.baseUrl, required this.models, required this.apis})
    : name = name ?? id;

  final String id;
  final String name;
  final String? baseUrl;
  final List<Model> models;
  final Map<String, ProviderStreams> apis;

  /// Streams [messages] to [model] on its API; an API the provider does not serve ends the stream with an error.
  AssistantMessageEventStream stream(
    Model model,
    List<Message> messages, [
    StreamOptions options = const StreamOptions(),
  ]) {
    final api = apis[model.api];
    if (api == null) {
      return errorStream(model, 'Provider $id has no API implementation for "${model.api}"', options.signal);
    }

    return api.stream(model, messages, options);
  }

  /// Streams with provider-neutral options, as [stream] does.
  AssistantMessageEventStream streamSimple(
    Model model,
    List<Message> messages, [
    SimpleStreamOptions options = const SimpleStreamOptions(),
  ]) {
    final api = apis[model.api];
    if (api == null) {
      return errorStream(model, 'Provider $id has no API implementation for "${model.api}"', options.signal);
    }

    return api.streamSimple(model, messages, options);
  }
}

/// Providers by id, and requests routed to the provider that owns a model. `streamSimple` fits [Agent]'s `stream`.
class Models {
  final _providers = <String, Provider>{};

  /// Adds [provider], or replaces the one with its id.
  void setProvider(Provider provider) => _providers[provider.id] = provider;

  void deleteProvider(String id) => _providers.remove(id);

  void clearProviders() => _providers.clear();

  List<Provider> getProviders() => List.of(_providers.values);

  Provider? getProvider(String id) => _providers[id];

  /// The models of one provider, or of every provider.
  List<Model> getModels([String? provider]) {
    if (provider != null) {
      return List.of(_providers[provider]?.models ?? const []);
    }

    return [for (final p in _providers.values) ...p.models];
  }

  Model? getModel(String provider, String id) => _providers[provider]?.models.where((m) => m.id == id).firstOrNull;

  /// Streams through the provider that owns [model]; an unknown provider ends the stream with an error.
  AssistantMessageEventStream stream(
    Model model,
    List<Message> messages, [
    StreamOptions options = const StreamOptions(),
  ]) {
    final provider = _providers[model.provider];
    if (provider == null) {
      return errorStream(model, 'Unknown provider: ${model.provider}', options.signal);
    }

    return provider.stream(model, messages, options);
  }

  Future<AssistantMessage> complete(
    Model model,
    List<Message> messages, [
    StreamOptions options = const StreamOptions(),
  ]) {
    return stream(model, messages, options).result;
  }

  /// Streams with provider-neutral options, as [stream] does.
  AssistantMessageEventStream streamSimple(
    Model model,
    List<Message> messages, [
    SimpleStreamOptions options = const SimpleStreamOptions(),
  ]) {
    final provider = _providers[model.provider];
    if (provider == null) {
      return errorStream(model, 'Unknown provider: ${model.provider}', options.signal);
    }

    return provider.streamSimple(model, messages, options);
  }

  Future<AssistantMessage> completeSimple(
    Model model,
    List<Message> messages, [
    SimpleStreamOptions options = const SimpleStreamOptions(),
  ]) {
    return streamSimple(model, messages, options).result;
  }
}

/// A stream that ends at once with an error response: `aborted` when [signal] has fired.
AssistantMessageEventStream errorStream(Model model, String error, AbortSignal? signal) {
  final reason = (signal?.aborted ?? false) ? StopReason.aborted : StopReason.error;
  final message = AssistantMessage(
    api: model.api,
    provider: model.provider,
    model: model.id,
    stopReason: reason,
    errorMessage: error,
    timestamp: DateTime.now().millisecondsSinceEpoch,
  );
  return AssistantMessageEventStream()..push(ErrorEvent(reason, message));
}

/// The built-in Anthropic provider: the catalog's Anthropic models on the Messages API.
Provider anthropicProvider() {
  return Provider(
    id: 'anthropic',
    name: 'Anthropic',
    baseUrl: 'https://api.anthropic.com',
    models: getBuiltinModels('anthropic'),
    apis: {
      'anthropic-messages': ProviderStreams(
        stream: (model, messages, options) =>
            streamAnthropic(model, messages, options is AnthropicOptions ? options : AnthropicOptions.from(options)),
        streamSimple: streamSimpleAnthropic,
      ),
    },
  );
}

/// Every built-in provider chorus has an API for, freshly built.
List<Provider> builtinProviders() => [anthropicProvider()];

/// A collection with every built-in provider registered.
Models builtinModels() {
  final models = Models();
  for (final provider in builtinProviders()) {
    models.setProvider(provider);
  }

  return models;
}

/// The providers of the built-in catalog, including those chorus has no API for yet.
List<String> getBuiltinProviders() => List.of(builtinCatalog.keys);

/// One model of the built-in catalog.
Model? getBuiltinModel(String provider, String id) => builtinCatalog[provider]?[id];

/// The models of one provider of the built-in catalog.
List<Model> getBuiltinModels(String provider) => List.of(builtinCatalog[provider]?.values ?? const []);

/// The thinking levels [model] supports: only `off` without reasoning; `xhigh` and `max` only when its level map names
/// them; a level the map sets to null never.
List<ThinkingLevel> getSupportedThinkingLevels(Model model) {
  if (!model.reasoning) {
    return const [ThinkingLevel.off];
  }

  return [
    for (final level in ThinkingLevel.values)
      if (_supported(model.levels, level)) level,
  ];
}

// A level the map names is supported unless it maps to null; an unnamed one unless it is xhigh or max.
bool _supported(Map<String, String?>? levels, ThinkingLevel level) {
  if (levels?.containsKey(level.name) ?? false) {
    return levels![level.name] != null;
  }

  return level != ThinkingLevel.xhigh && level != ThinkingLevel.max;
}

/// [level] when [model] supports it, else the nearest supported level above it, else the nearest below.
ThinkingLevel clampThinkingLevel(Model model, ThinkingLevel level) {
  final available = getSupportedThinkingLevels(model);
  if (available.contains(level)) {
    return level;
  }

  final above = ThinkingLevel.values.skip(level.index + 1).where(available.contains);
  if (above.isNotEmpty) {
    return above.first;
  }

  return ThinkingLevel.values.take(level.index).where(available.contains).lastOrNull ?? ThinkingLevel.off;
}

/// Whether two models are the same: the same id from the same provider.
bool modelsAreEqual(Model? a, Model? b) {
  if (a == null || b == null) {
    return false;
  }

  return a.id == b.id && a.provider == b.provider;
}
