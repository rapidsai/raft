# Copyright (c) 2022, NVIDIA CORPORATION.

import os
import signal
import time

import pytest

from pylibraft.common.interruptible import cuda_interruptible, cuda_yield


def send_ctrl_c():
    # signal.raise_signal(signal.SIGINT) available only since python 3.8
    os.kill(os.getpid(), signal.SIGINT)


def test_should_cancel_via_interruptible():
    start_time = time.monotonic()
    with pytest.raises(RuntimeError, match="this thread was cancelled"):
        with cuda_interruptible():
            send_ctrl_c()
            cuda_yield()
            time.sleep(1.0)
    end_time = time.monotonic()
    assert (
        end_time < start_time + 0.5
    ), "The process seems to have waited, while it shouldn't have."


def test_should_cancel_via_python():
    start_time = time.monotonic()
    with pytest.raises(KeyboardInterrupt):
        send_ctrl_c()
        cuda_yield()
        time.sleep(1.0)
    end_time = time.monotonic()
    assert (
        end_time < start_time + 0.5
    ), "The process seems to have waited, while it shouldn't have."


def test_should_wait_no_interrupt():
    start_time = time.monotonic()
    with cuda_interruptible():
        cuda_yield()
        time.sleep(1.0)
    end_time = time.monotonic()
    assert (
        end_time > start_time + 0.5
    ), "The process seems to be cancelled, while it shouldn't be."


def test_should_wait_no_yield():
    start_time = time.monotonic()
    with cuda_interruptible():
        send_ctrl_c()
        time.sleep(1.0)
    end_time = time.monotonic()
    assert (
        end_time > start_time + 0.5
    ), "The process seems to be cancelled, while it shouldn't be."
