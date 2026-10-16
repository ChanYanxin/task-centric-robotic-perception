import socket
import threading
import cv2
import numpy as np
import pytest
from tc_perception.detectors import ColorShapeDetector
from tc_perception.transport import (
    LocalEdge, TCPEdgeClient, handle_client, jpeg_encode, jpeg_decode,
    send_message, read_message)


def test_jpeg_round_trip():
    frame=np.zeros((80,100,3),dtype=np.uint8)
    frame[20:65,30:80]=(0,0,255)
    out=jpeg_decode(jpeg_encode(frame))
    assert out.shape==frame.shape
    assert np.abs(out.astype(int)-frame.astype(int)).mean()<5


def test_socket_pair_framing():
    a,b=socket.socketpair()
    try:
        sent=send_message(a,{"frame_id":5},b"payload")
        header,data=read_message(b)
        assert sent>len(data)
        assert header["frame_id"]==5 and data==b"payload"
    finally:
        a.close();b.close()


def test_end_to_end_tcp_edge():
    # Actual loopback TCP over ephemeral OS-assigned port, not a mocked transport.
    sock=socket.socket()
    sock.bind(("127.0.0.1",0));sock.listen(1)
    host,port=sock.getsockname()
    def server():
        conn,_=sock.accept()
        handle_client(conn,ColorShapeDetector())
        sock.close()
    thread=threading.Thread(target=server,daemon=True)
    thread.start()
    image=np.zeros((80,100,3),dtype=np.uint8)
    cv2.rectangle(image,(20,25),(75,60),(0,0,255),-1)
    with TCPEdgeClient(host,port) as edge:
        dets,sent,elapsed=edge.infer(image,1)
    thread.join(timeout=2)
    assert any(d.label=="obstacle" for d in dets)
    assert sent>100 and elapsed>=0
