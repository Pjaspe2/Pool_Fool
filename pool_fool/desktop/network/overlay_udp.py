from __future__ import annotations

import socket
import threading
from typing import Callable

from pool_fool.shared.schemas import OverlayMessage


class OverlaySender:
    def __init__(self, host: str, port: int) -> None:
        self._addr = (host, port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, msg: OverlayMessage) -> None:
        data = msg.to_json().encode("utf-8")
        if len(data) > 60000:
            return
        self._sock.sendto(data, self._addr)

    def close(self) -> None:
        self._sock.close()


class OverlayReceiver:
    def __init__(self, host: str, port: int, on_message: Callable[[OverlayMessage], None]) -> None:
        self._on_message = on_message
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((host, port))
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            try:
                data, _ = self._sock.recvfrom(65535)
                msg = OverlayMessage.from_json(data.decode("utf-8"))
                self._on_message(msg)
            except OSError:
                break
            except (ValueError, KeyError):
                continue

    def stop(self) -> None:
        self._running = False
        self._sock.close()
        if self._thread:
            self._thread.join(timeout=1.0)
