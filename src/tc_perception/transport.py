"""Local codec-only simulation and length-prefixed, JSON-header TCP transport.

Explicit framed protocol: 4-byte uint32 network-order header-length, JSON
header containing image bytes length + frame id, followed by JPEG, then a
framed JSON reply. No untrusted pickle and no arbitrary code execution.
"""
from __future__ import annotations
import json
import socket
import struct
import threading
import time
from typing import Callable
import cv2
import numpy as np
from .types import Detection
from .detectors import Detector

MAX_HEADER = 65536
MAX_IMAGE = 8 * 1024 * 1024


def jpeg_encode(image: np.ndarray, quality: int = 80) -> bytes:
    success, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        raise ValueError("JPEG encoding failed")
    return buf.tobytes()


def jpeg_decode(payload: bytes) -> np.ndarray:
    decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError("Invalid JPEG payload")
    return decoded


def receive_exact(sock: socket.socket, count: int) -> bytes:
    data = bytearray()
    while len(data) < count:
        chunk = sock.recv(count - len(data))
        if not chunk:
            raise ConnectionError("Connection closed during frame")
        data.extend(chunk)
    return bytes(data)


def send_message(sock: socket.socket, header: dict, payload: bytes = b"") -> int:
    header = {**header, "payload_len": len(payload)}
    raw_header = json.dumps(header, separators=(",", ":")).encode("utf8")
    if len(raw_header) > MAX_HEADER or len(payload) > MAX_IMAGE:
        raise ValueError("Message too large")
    blob = struct.pack("!I", len(raw_header)) + raw_header + payload
    sock.sendall(blob)
    return len(blob)


def read_message(sock: socket.socket) -> tuple[dict, bytes]:
    header_size = struct.unpack("!I", receive_exact(sock, 4))[0]
    if not 0 < header_size <= MAX_HEADER:
        raise ValueError("Invalid header size")
    header = json.loads(receive_exact(sock, header_size))
    payload_size = int(header.get("payload_len", -1))
    if not 0 <= payload_size <= MAX_IMAGE:
        raise ValueError("Invalid payload size")
    return header, receive_exact(sock, payload_size)


class LocalEdge:
    """Same encode/decode/detector as TCP, without socket network delay."""
    def __init__(self, detector: Detector, jpeg_quality: int = 80):
        self.detector, self.jpeg_quality = detector, jpeg_quality

    def infer(self, crop: np.ndarray, frame_id: int) -> tuple[list[Detection], int, float]:
        payload = jpeg_encode(crop, self.jpeg_quality)
        decoded = jpeg_decode(payload)
        started = time.perf_counter()
        detections = self.detector.detect(decoded)
        infer_ms = (time.perf_counter() - started) * 1000
        # Count actual frame protocol overhead, not just compressed pixels.
        header = json.dumps({"frame_id":frame_id,"payload_len":len(payload)}, separators=(",", ":")).encode()
        return detections, 4 + len(header) + len(payload), infer_ms


class TCPEdgeClient:
    def __init__(self, host: str, port: int, quality: int = 80):
        self.sock = socket.create_connection((host, port), timeout=15)
        self.sock.settimeout(20)
        self.quality = quality

    def infer(self, crop: np.ndarray, frame_id: int) -> tuple[list[Detection], int, float]:
        payload = jpeg_encode(crop, self.quality)
        sent = send_message(self.sock, {"frame_id": frame_id}, payload)
        response, _ = read_message(self.sock)
        if response.get("frame_id") != frame_id:
            raise ValueError("Mismatched edge reply")
        if "error" in response:
            raise RuntimeError(response["error"])
        dets = [Detection(d["label"], float(d["confidence"]), tuple(d["box"]))
                for d in response["detections"]]
        return dets, sent, float(response["inference_ms"])

    def close(self) -> None:
        self.sock.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def handle_client(sock: socket.socket, detector: Detector) -> None:
    with sock:
        while True:
            try:
                header, payload = read_message(sock)
            except ConnectionError:
                return
            start = time.perf_counter()
            try:
                frame = jpeg_decode(payload)
                detections = [d.to_dict() for d in detector.detect(frame)]
                inference_ms = (time.perf_counter() - start) * 1000
                send_message(sock, {"frame_id":header["frame_id"],
                                    "detections":detections,"inference_ms":inference_ms})
            except Exception as exc:
                send_message(sock, {"frame_id":header.get("frame_id"),"error":str(exc)})


def serve(host: str, port: int, detector: Detector) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, port))
        server.listen(4)
        print(f"Edge server listening on {host}:{port}", flush=True)
        try:
            while True:
                conn, addr = server.accept()
                threading.Thread(target=handle_client, args=(conn, detector), daemon=True).start()
        except KeyboardInterrupt:
            print("Stopping edge server")
