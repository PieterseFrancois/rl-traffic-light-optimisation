# tests/test_communication.py
"""
Communication Network Test Suite

Covers
------
1) Basic pub/sub delivery:
   Only subscribed clients receive a publisher's embedding.
2) Replay-last on subscribe:
   Late subscribers get the last retained embedding immediately.
3) No replay when bus does not retain:
   Subscribing with replay_last=True does not replay if the bus has no last.
4) Unsubscribe and unsubscribe_all:
   After unsubscribing, clients stop receiving.
5) Sender guard:
   Client cannot publish as another intersection id.
6) Subscriptions property:
   Tracks current subscriptions in sorted order.
7) Bus publishers() and get_last_embedding():
   Publishers set updates; get_last returns a copy (immutable from external code).
8) Concurrency smoke:
   Concurrent publishers deliver to subscribers without loss (best-effort check).
"""

import threading
import time
import numpy as np
import pytest
import importlib
import sys


# -------------------------
# Import with alias helper
# -------------------------

@pytest.fixture(scope="module")
def comm_modules():
    """
    Import the comm bus and client. We alias the bus module to a top-level
    name to match the client's import path if needed.
    """
    bus_mod = importlib.import_module("modules.communication_bus")
    sys.modules.setdefault("communication_bus", bus_mod)  # alias for client import
    client_mod = importlib.import_module("modules.intersection.communication")
    return bus_mod, client_mod


# -------------------------
# Small sink for callbacks
# -------------------------

class _Sink:
    def __init__(self):
        self.events: list[tuple[str, np.ndarray]] = []

    def cb(self, sender_id: str, emb: np.ndarray):
        # Store a copy to avoid aliasing side effects in assertions
        self.events.append((sender_id, np.asarray(emb, dtype=np.float32).copy()))

    def clear(self):
        self.events.clear()


# ----------------
# Test utilities
# ----------------

def _arr(v):
    return np.asarray(v, dtype=np.float32)


# ---------------
# The test cases
# ---------------

def test_basic_pub_sub_delivery(comm_modules):
    """
    1) Basic pub/sub delivery: only subscribers receive messages.
    """
    Bus, Client = comm_modules
    bus = Bus.CommunicationBus(retain_last=True)
    A = Client.IntersectionCommClient("A", bus)
    B = Client.IntersectionCommClient("B", bus)
    C = Client.IntersectionCommClient("C", bus)

    sinkA, sinkC = _Sink(), _Sink()
    A.subscribe_embeddings("B", sinkA.cb, replay_last=False)  # A listens to B
    # C not subscribed to B

    m = _arr([1, 2, 3],)
    B.publish_embedding("B", m)

    # A should receive exactly one event from B; C none
    assert len(sinkA.events) == 1
    assert sinkA.events[0][0] == "B"
    assert np.array_equal(sinkA.events[0][1], m)
    assert len(sinkC.events) == 0


def test_replay_last_on_subscribe(comm_modules):
    """
    2) Replay-last on subscribe: new subscriber gets last immediately.
    """
    Bus, Client = comm_modules
    bus = Bus.CommunicationBus(retain_last=True)
    B = Client.IntersectionCommClient("B", bus)
    A = Client.IntersectionCommClient("A", bus)

    last = _arr([7, 7, 7])
    B.publish_embedding("B", last)

    sinkA = _Sink()
    A.subscribe_embeddings("B", sinkA.cb, replay_last=True)  # should replay immediately

    assert len(sinkA.events) == 1
    assert sinkA.events[0][0] == "B"
    assert np.array_equal(sinkA.events[0][1], last)


def test_no_replay_when_bus_not_retaining(comm_modules):
    """
    3) No replay when bus does not retain:
       Even with replay_last=True, nothing replays if the bus held no last.
    """
    Bus, Client = comm_modules
    bus = Bus.CommunicationBus(retain_last=False)  # no retention
    B = Client.IntersectionCommClient("B", bus)
    A = Client.IntersectionCommClient("A", bus)

    # Publish once (bus does not retain)
    B.publish_embedding("B", _arr([1, 2, 3]))

    sinkA = _Sink()
    A.subscribe_embeddings("B", sinkA.cb, replay_last=True)  # no last to replay
    assert len(sinkA.events) == 0

    # Next publish should be delivered normally
    B.publish_embedding("B", _arr([4, 5, 6]))
    assert len(sinkA.events) == 1


def test_unsubscribe_and_unsubscribe_all(comm_modules):
    """
    4) Unsubscribe and unsubscribe_all stop delivery.
    """
    Bus, Client = comm_modules
    bus = Bus.CommunicationBus(retain_last=True)
    A = Client.IntersectionCommClient("A", bus)
    D = Client.IntersectionCommClient("D", bus)
    B = Client.IntersectionCommClient("B", bus)

    sinkA, sinkD = _Sink(), _Sink()
    A.subscribe_embeddings("B", sinkA.cb, replay_last=False)
    D.subscribe_embeddings("B", sinkD.cb, replay_last=False)

    B.publish_embedding("B", _arr([0]))
    assert len(sinkA.events) == 1 and len(sinkD.events) == 1

    # A unsubscribes from B; D still subscribed
    A.unsubscribe_embeddings("B")
    B.publish_embedding("B", _arr([1]))
    assert len(sinkA.events) == 1  # unchanged
    assert len(sinkD.events) == 2  # received second

    # D unsubscribes from all
    D.unsubscribe_all()
    B.publish_embedding("B", _arr([2]))
    assert len(sinkD.events) == 2  # no new events


def test_sender_guard(comm_modules):
    """
    5) Sender guard: client cannot publish as another id.
    """
    Bus, Client = comm_modules
    bus = Bus.CommunicationBus()
    A = Client.IntersectionCommClient("A", bus)
    with pytest.raises(ValueError):
        A.publish_embedding("B", _arr([1.0]))  # wrong sender id


def test_subscriptions_property(comm_modules):
    """
    6) Subscriptions property returns sorted tuple of current subscriptions.
    """
    Bus, Client = comm_modules
    bus = Bus.CommunicationBus()
    A = Client.IntersectionCommClient("A", bus)
    sink = _Sink()
    A.subscribe_embeddings("C", sink.cb)
    A.subscribe_embeddings("B", sink.cb)
    assert A.subscriptions == ("B", "C")
    A.unsubscribe_embeddings("B")
    assert A.subscriptions == ("C",)


def test_bus_publishers_and_get_last_embedding_copy(comm_modules):
    """
    7) publishers() set updates; get_last_embedding returns a copy resistant to external mutation.
    """
    Bus, Client = comm_modules
    bus = Bus.CommunicationBus(retain_last=True)
    B = Client.IntersectionCommClient("B", bus)

    x = _arr([9, 9, 9])
    B.publish_embedding("B", x)
    pubs = set(bus.publishers())
    assert "B" in pubs

    got1 = bus.get_last_embedding("B")
    assert got1 is not None and np.array_equal(got1, x)

    # mutate the returned copy; bus internal state must not change
    got1[0] = 123.0  # type: ignore[index]
    got2 = bus.get_last_embedding("B")
    assert got2 is not None and got2[0] == 9.0


def test_concurrency_smoke(comm_modules):
    """
    8) Concurrency smoke: concurrent publishers deliver reliably.
       We do a light check on counts rather than strict ordering.
    """
    Bus, Client = comm_modules
    bus = Bus.CommunicationBus(retain_last=True)
    A = Client.IntersectionCommClient("A", bus)
    B = Client.IntersectionCommClient("B", bus)
    C = Client.IntersectionCommClient("C", bus)

    sinkA = _Sink()
    A.subscribe_embeddings("B", sinkA.cb)
    A.subscribe_embeddings("C", sinkA.cb)

    def pub(client, name, n=20):
        for i in range(n):
            client.publish_embedding(name, _arr([i]))
            # tiny sleep to shuffle interleaving
            time.sleep(0.001)

    t1 = threading.Thread(target=pub, args=(B, "B", 20))
    t2 = threading.Thread(target=pub, args=(C, "C", 20))
    t1.start(); t2.start()
    t1.join(); t2.join()

    # A should have received 40 events total (allow a small buffer for CI jitter if needed)
    assert len(sinkA.events) == 40
    # Check that both senders appear
    senders = {sid for sid, _ in sinkA.events}
    assert senders == {"B", "C"}
