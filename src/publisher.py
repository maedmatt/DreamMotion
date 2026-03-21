from __future__ import annotations

import json
import os

import zmq

ZMQ_ADDRESS = os.environ.get("ZMQ_PUB_ADDRESS", "tcp://*:5555")

_context: zmq.Context | None = None  # pyright: ignore[reportMissingTypeArgument]
_socket: zmq.Socket | None = None  # pyright: ignore[reportMissingTypeArgument]


def init_publisher() -> None:
    """Bind the ZMQ PUB socket. Call once at startup."""
    global _context, _socket
    _context = zmq.Context()
    _socket = _context.socket(zmq.PUB)
    _socket.bind(ZMQ_ADDRESS)  # pyright: ignore[reportOptionalMemberAccess]
    print(f"ZMQ publisher bound to {ZMQ_ADDRESS}")


def publish_motion(
    metadata: dict,
    pt_bytes: bytes,
    topic: bytes = b"motion",
) -> None:
    """Publish a motion trajectory as a multipart ZMQ message.

    Frames:
        0: topic (e.g. b"motion")
        1: metadata JSON (prompt, duration, num_frames, ...)
        2: raw .pt bytes
    """
    if _socket is None:
        return
    _socket.send_multipart(
        [
            topic,
            json.dumps(metadata).encode(),
            pt_bytes,
        ]
    )


def close_publisher() -> None:
    """Clean up the ZMQ socket and context."""
    global _context, _socket
    if _socket is not None:
        _socket.close()
        _socket = None
    if _context is not None:
        _context.term()
        _context = None
