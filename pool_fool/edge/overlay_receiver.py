from __future__ import annotations

import socket
import threading
from typing import Callable

from pool_fool.shared.schemas import OverlayMessage


class EdgeOverlayReceiver:
    def __init__(self, port: int, on_message: Callable[[OverlayMessage], None], host: str = "0.0.0.0") -> None:
        self._port = port
        self._host = host
        self._on_message = on_message
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._running = False
        self._thread: threading.Thread | None = None
        self._latest: OverlayMessage | None = None

    @property
    def latest(self) -> OverlayMessage | None:
        return self._latest

    def start(self) -> None:
        self._sock.bind((self._host, self._port))
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            try:
                data, _ = self._sock.recvfrom(65535)
                msg = OverlayMessage.from_json(data.decode("utf-8"))
                self._latest = msg
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
