from __future__ import annotations

import importlib


def physical_memory_bytes() -> int:
    psutil = importlib.import_module("psutil")
    return int(psutil.virtual_memory().total)


def process_rss_bytes() -> int:
    psutil = importlib.import_module("psutil")
    return int(psutil.Process().memory_info().rss)
