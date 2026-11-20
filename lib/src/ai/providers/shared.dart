// What the providers share.

/// Thrown by the provider for a failure it detects itself; the message goes to the response's `errorMessage`.
class ProviderException implements Exception {
  const ProviderException(this.message);

  final String message;

  @override
  String toString() => message;
}

final _surrogates = RegExp(r'[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]');

/// [text] without unpaired UTF-16 surrogates, which many APIs reject.
String sanitizeSurrogates(String text) => text.replaceAll(_surrogates, '');
