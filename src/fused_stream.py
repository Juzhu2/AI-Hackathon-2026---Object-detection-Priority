"""Bridge between the geo fuser and the prioritizer / VLM loop.

Two small writers, both safe to read from other processes while running:

  FusedStream  - one clean_record() per line (JSON Lines), flushed every frame.
                 fastpath.py tails this file. Unlike CleanJsonWriter it is valid
                 at every moment, not only after close().
  FrameShare   - writes the latest camera frame to a JPEG atomically
                 (temp file + os.replace) so vlm_loop.py never reads half a file.
                 Throttled: the VLM needs ~1 frame per second, not 30.
"""

import json
import os
import time

from geometry_fuser import clean_record


class FusedStream:
    def __init__(self, path, meta=None):
        self.path = str(path)
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self.file = open(self.path, "w", encoding="utf-8", buffering=1)  # truncate at start
        if meta is not None:
            # No "objects" key, so fastpath skips this line.
            self.file.write(json.dumps({"type": "meta", **meta}) + "\n")
            self.file.flush()

    def write(self, record):
        """Call after fuser.update(record) and after record["latency_ms"] is set."""
        self.file.write(json.dumps(clean_record(record), separators=(",", ":")) + "\n")
        self.file.flush()

    def close(self):
        if not self.file.closed:
            self.file.close()


class FrameShare:
    def __init__(self, path, cv2, every_s=0.5, quality=80):
        self.path = str(path)
        self.tmp = self.path + ".tmp.jpg"      # must end in .jpg so OpenCV picks the codec
        self.cv2, self.every_s, self.quality = cv2, every_s, quality
        self._last = 0.0
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)

    def write(self, frame):
        now = time.monotonic()
        if now - self._last < self.every_s:
            return
        self._last = now
        ok = self.cv2.imwrite(self.tmp, frame, [self.cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if ok:
            os.replace(self.tmp, self.path)    # atomic on the same filesystem

    def close(self):
        for p in (self.tmp,):
            try:
                os.remove(p)
            except OSError:
                pass
