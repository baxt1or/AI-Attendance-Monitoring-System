
import threading
import serial
import numpy as np
import cv2

PORT = "/dev/tty.usbserial-10"
BAUD = 460800
START_MARKER = bytes([0xAA, 0xBB, 0xCC, 0xDD])
MAX_FRAME_SIZE = 200_000


class ESP32Cam:
    def __init__(self, port=PORT, baud=BAUD):
        self._ser = serial.Serial(port, baud, timeout=1)
        self._frame = None
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self):
        buf = b""
        while self._running:
            waiting = self._ser.in_waiting
            buf += self._ser.read(max(waiting, 1))

            # scan for the latest complete frame
            latest = None
            search = 0
            while True:
                pos = buf.find(START_MARKER, search)
                if pos == -1:
                    break
                if len(buf) < pos + 8:
                    break
                size = int.from_bytes(buf[pos+4:pos+8], 'big')
                if size == 0 or size > MAX_FRAME_SIZE:
                    search = pos + 4
                    continue
                end = pos + 8 + size
                if len(buf) < end:
                    break
                latest = buf[pos+8:end]
                buf = buf[end:]
                search = 0

            if latest is not None:
                frame = cv2.imdecode(np.frombuffer(latest, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    with self._lock:
                        self._frame = frame

    def read(self):
        """Returns (True, frame) if a frame is available, (False, None) otherwise."""
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    def release(self):
        self._running = False
        self._thread.join(timeout=2)
        self._ser.close()
