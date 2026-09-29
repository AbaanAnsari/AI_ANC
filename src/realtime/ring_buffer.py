"""
src/realtime/ring_buffer.py
============================
Thread-safe ring (circular) buffer for streaming audio I/O.

Used in the streaming pipeline to decouple audio capture from processing.

Features:
    - Configurable capacity (in samples)
    - Write (produce) and read (consume) operations
    - Peek without consuming
    - Available samples / free space queries
    - Reset
    - Float32 samples
    - Deterministic — no dynamic allocation after init
"""
from __future__ import annotations

import threading
from typing import Optional

import numpy as np


class RingBuffer:
    """
    Thread-safe circular audio sample buffer.

    Parameters
    ----------
    capacity : int
        Maximum number of float32 samples the buffer can hold.
    """

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be > 0, got {capacity}")
        self._capacity = int(capacity)
        self._buffer = np.zeros(self._capacity, dtype=np.float32)
        self._write_idx: int = 0
        self._read_idx: int = 0
        self._available: int = 0
        self._lock = threading.Lock()

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def available(self) -> int:
        """Number of samples available to read."""
        with self._lock:
            return self._available

    @property
    def free_space(self) -> int:
        """Number of samples that can be written."""
        with self._lock:
            return self._capacity - self._available

    def write(self, data: np.ndarray) -> int:
        """
        Write samples into the buffer.

        Writes as many samples as will fit. Returns the number written.
        Does NOT block if buffer is full.

        Parameters
        ----------
        data : np.ndarray
            1-D float32 array of samples to write.

        Returns
        -------
        int
            Number of samples actually written.
        """
        data = np.asarray(data, dtype=np.float32).ravel()
        n_to_write = len(data)

        with self._lock:
            free = self._capacity - self._available
            n_write = min(n_to_write, free)
            if n_write == 0:
                return 0

            # Write possibly wrapping around
            first_chunk = min(n_write, self._capacity - self._write_idx)
            self._buffer[self._write_idx: self._write_idx + first_chunk] = data[:first_chunk]

            if n_write > first_chunk:
                # Wrap around
                second_chunk = n_write - first_chunk
                self._buffer[:second_chunk] = data[first_chunk:first_chunk + second_chunk]

            self._write_idx = (self._write_idx + n_write) % self._capacity
            self._available += n_write

        return n_write

    def read(self, n_samples: int) -> np.ndarray:
        """
        Read and consume samples from the buffer.

        Returns as many samples as available, up to n_samples.

        Parameters
        ----------
        n_samples : int
            Number of samples to read.

        Returns
        -------
        np.ndarray, shape (k,) where k <= n_samples
            Consumed samples in float32.
        """
        with self._lock:
            n_read = min(n_samples, self._available)
            if n_read == 0:
                return np.empty(0, dtype=np.float32)

            output = np.empty(n_read, dtype=np.float32)

            first_chunk = min(n_read, self._capacity - self._read_idx)
            output[:first_chunk] = self._buffer[self._read_idx: self._read_idx + first_chunk]

            if n_read > first_chunk:
                second_chunk = n_read - first_chunk
                output[first_chunk:] = self._buffer[:second_chunk]

            self._read_idx = (self._read_idx + n_read) % self._capacity
            self._available -= n_read

        return output

    def peek(self, n_samples: int) -> np.ndarray:
        """
        Read samples without consuming them.

        Parameters
        ----------
        n_samples : int
            Number of samples to peek.

        Returns
        -------
        np.ndarray — samples without advancing read pointer.
        """
        with self._lock:
            n_peek = min(n_samples, self._available)
            if n_peek == 0:
                return np.empty(0, dtype=np.float32)

            output = np.empty(n_peek, dtype=np.float32)
            first_chunk = min(n_peek, self._capacity - self._read_idx)
            output[:first_chunk] = self._buffer[self._read_idx: self._read_idx + first_chunk]

            if n_peek > first_chunk:
                second_chunk = n_peek - first_chunk
                output[first_chunk:] = self._buffer[:second_chunk]

        return output

    def discard(self, n_samples: int) -> int:
        """
        Discard up to n_samples from the buffer without copying.

        Parameters
        ----------
        n_samples : int
            Number of samples to discard.

        Returns
        -------
        int
            Number of samples actually discarded.
        """
        with self._lock:
            n_discard = min(int(n_samples), self._available)
            if n_discard <= 0:
                return 0

            self._read_idx = (self._read_idx + n_discard) % self._capacity
            self._available -= n_discard
            return n_discard

    def reset(self) -> None:
        """Clear all buffer contents and reset to initial state."""
        with self._lock:
            self._buffer[:] = 0.0
            self._write_idx = 0
            self._read_idx = 0
            self._available = 0

