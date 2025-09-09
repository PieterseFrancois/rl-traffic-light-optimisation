# modules/intersection/communication.py
from __future__ import annotations

from typing import Callable, Dict, Optional, Set
import numpy as np

from ..communication_bus import CommunicationBus


class IntersectionCommClient:
    """
    Thin client an intersection uses to talk on the CommunicationBus.

    - Keeps a set of intersection ids it is subscribed to.
    - For publishing, call publish_embedding(...) with your own id.
      A guard ensures you do not accidentally publish as another id.
    """

    def __init__(self, intersection_id: str, bus: CommunicationBus):
        self.intersection_id = str(intersection_id)
        self.bus = bus
        self._subs: Set[str] = set()

    # --------------------------- publish -------------------------------------

    def publish_embedding(self, sender_id: str, embedding: np.ndarray) -> None:
        if sender_id != self.intersection_id:
            raise ValueError(f"Client {self.intersection_id} cannot publish as {sender_id}")
        self.bus.publish_embedding(sender_id, embedding)

    # --------------------------- subscribe -----------------------------------

    def subscribe_embeddings(
        self,
        publisher_id: str,
        callback: Callable[[str, np.ndarray], None],
        replay_last: Optional[bool] = None,
    ) -> None:
        self.bus.subscribe_embeddings(self.intersection_id, publisher_id, callback, replay_last=replay_last)
        self._subs.add(publisher_id)

    def unsubscribe_embeddings(self, publisher_id: str) -> None:
        self.bus.unsubscribe_embeddings(self.intersection_id, publisher_id)
        self._subs.discard(publisher_id)

    def unsubscribe_all(self) -> None:
        self.bus.unsubscribe_all(self.intersection_id)
        self._subs.clear()

    # --------------------------- inspection ----------------------------------

    @property
    def subscriptions(self) -> tuple[str, ...]:
        """Intersections this client is currently subscribed to."""
        return tuple(sorted(self._subs))

    def last_from(self, publisher_id: str) -> Optional[np.ndarray]:
        """Fetch the last retained embedding for a given publisher, if any."""
        return self.bus.get_last_embedding(publisher_id)
