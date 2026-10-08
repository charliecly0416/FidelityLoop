"""Bounded FIFO disk writer; event timestamps/order belong to the controller."""
import queue
import threading


class JournalDrainTimeout(TimeoutError):
    """The writer may still mutate raw files; only the exited parent may seal."""


class OrderedJournal:
    def __init__(self, path, *, max_pending_bytes=16 * 1024 * 1024):
        # Opening before observation also fails early on an unwritable directory.
        self.stream = path.open('x', encoding='utf-8', buffering=1)
        self.pending = queue.Queue()
        self.lock = threading.Lock()
        self.limit = max_pending_bytes
        self.pending_bytes = 0
        self.error = None
        self.closed = False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def check(self):
        if self.error is not None:
            raise RuntimeError('raw journal persistence failed') from self.error

    def write(self, text):
        self.check()
        size = len(text.encode('utf-8'))
        with self.lock:
            if self.closed:
                raise ValueError('journal already closed')
            if self.pending_bytes + size > self.limit:
                self.error = BufferError('raw journal backlog exceeded byte limit')
                raise self.error
            self.pending_bytes += size
            self.pending.put_nowait((text, size))

    def _run(self):
        try:
            while True:
                item = self.pending.get()
                if item is None:
                    break
                text, size = item
                self.stream.write(text)
                with self.lock:
                    self.pending_bytes -= size
        except BaseException as exc:
            self.error = exc
        finally:
            try:
                self.stream.close()
            except BaseException as exc:
                self.error = exc

    def close(self, timeout=120.0):
        # Call off the event loop. Return only after every accepted record has
        # been written and the stream closed, before accounting or sealing.
        with self.lock:
            if not self.closed:
                self.closed = True
                self.pending.put_nowait(None)
        self.thread.join(timeout=max(0.0, timeout))
        if self.thread.is_alive():
            error = JournalDrainTimeout('raw journal drain deadline exhausted')
            self.error = error
            raise error
        self.check()
