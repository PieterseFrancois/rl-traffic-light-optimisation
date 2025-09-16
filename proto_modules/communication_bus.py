# modules/intersection/communication_bus.py
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Set
import threading
import numpy as np


class CommunicationBus:
    """
    In-process pub/sub for intersection embeddings.

    - Subscribers register a callback per publisher id.
    - When a publisher posts a new embedding, all its subscribers are called.
    - Optionally retains the last embedding per publisher and replays it to
      new subscribers upon subscribe.

    Thread-safe for simple single-process training/eval loops.
    """

    def __init__(self, retain_last: bool = True):
        self._lock = threading.RLock()
        # publisher_id -> {subscriber_id -> callback(sender_id, embedding)}
        self._subs: Dict[str, Dict[str, Callable[[str, np.ndarray], None]]] = {}
        # subscriber_id -> set[publisher_id]
        self._subs_by_sub: Dict[str, Set[str]] = {}
        # publisher_id -> last np.ndarray
        self._last: Dict[str, np.ndarray] = {}
        self._retain_last = bool(retain_last)

    # --------------------------- publish -------------------------------------

    def publish_embedding(self, sender_id: str, embedding: np.ndarray) -> None:
        arr = np.asarray(embedding, dtype=np.float32)
        with self._lock:
            if self._retain_last:
                self._last[sender_id] = arr.copy()
            for cb in self._subs.get(sender_id, {}).values():
                cb(sender_id, arr)

    # --------------------------- subscribe -----------------------------------

    def subscribe_embeddings(
        self,
        subscriber_id: str,
        publisher_id: str,
        callback: Callable[[str, np.ndarray], None],
        replay_last: Optional[bool] = None,
    ) -> None:
        """
        Subscribe subscriber_id to publisher_id.
        If replay_last is True and a last value exists, the callback is invoked once immediately.
        """
        with self._lock:
            self._subs.setdefault(publisher_id, {})[subscriber_id] = callback
            self._subs_by_sub.setdefault(subscriber_id, set()).add(publisher_id)
            if (
                self._retain_last if replay_last is None else replay_last
            ) and publisher_id in self._last:
                # Call outside lock to avoid re-entrancy surprises
                last = self._last[publisher_id].copy()
        if (
            self._retain_last if replay_last is None else replay_last
        ) and publisher_id in self._last:
            callback(publisher_id, last)

    def unsubscribe_embeddings(self, subscriber_id: str, publisher_id: str) -> None:
        with self._lock:
            if publisher_id in self._subs and subscriber_id in self._subs[publisher_id]:
                del self._subs[publisher_id][subscriber_id]
            if subscriber_id in self._subs_by_sub:
                self._subs_by_sub[subscriber_id].discard(publisher_id)

    def unsubscribe_all(self, subscriber_id: str) -> None:
        with self._lock:
            pubs = list(self._subs_by_sub.get(subscriber_id, set()))
            for pid in pubs:
                if pid in self._subs and subscriber_id in self._subs[pid]:
                    del self._subs[pid][subscriber_id]
            self._subs_by_sub.pop(subscriber_id, None)

    # --------------------------- inspection ----------------------------------

    def publishers(self) -> List[str]:
        with self._lock:
            return list(self._subs.keys() | self._last.keys())

    def subscriptions_for(self, subscriber_id: str) -> Set[str]:
        with self._lock:
            return set(self._subs_by_sub.get(subscriber_id, set()))

    def get_last_embedding(self, publisher_id: str) -> Optional[np.ndarray]:
        with self._lock:
            arr = self._last.get(publisher_id)
            return None if arr is None else arr.copy()
