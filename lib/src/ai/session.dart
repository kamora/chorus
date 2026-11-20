// State providers keep per session (by StreamOptions.sessionId), and its cleanup when a session ends. Ported from
// pi's session-resources.ts.

/// Frees what is kept for one session, or for every session when [sessionId] is null.
typedef SessionCleanup = void Function(String? sessionId);

final _cleanups = <SessionCleanup>{};

/// Registers [cleanup] for [cleanupSessionResources]; the returned function unregisters it.
void Function() registerSessionCleanup(SessionCleanup cleanup) {
  _cleanups.add(cleanup);
  return () => _cleanups.remove(cleanup);
}

/// Frees what providers keep for [sessionId], or for every session. Every cleanup runs even when one throws; their
/// errors are thrown together afterwards.
void cleanupSessionResources([String? sessionId]) {
  final errors = <Object>[];
  for (final cleanup in List.of(_cleanups)) {
    try {
      cleanup(sessionId);
    } catch (error) {
      errors.add(error);
    }
  }

  if (errors.isNotEmpty) {
    throw StateError('Failed to clean up session resources: ${errors.join('; ')}');
  }
}

/// A set of session ids that keeps at most [max] entries (the first added go first) and empties with
/// [cleanupSessionResources].
class SessionSet {
  SessionSet(this.max) {
    registerSessionCleanup((id) {
      if (id == null) {
        _ids.clear();
      } else {
        _ids.remove(id);
      }
    });
  }

  final int max;
  final _ids = <String>{};

  /// Adds [id]; does nothing without one.
  void add(String? id) {
    if (id == null) {
      return;
    }

    _ids.add(id);
    if (_ids.length > max) {
      _ids.remove(_ids.first);
    }
  }

  bool contains(String? id) => id != null && _ids.contains(id);
}
