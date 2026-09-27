"""Identify the shared instance, whose credential settings remain read-only."""

import os


def is_shared_instance() -> bool:
    return os.getenv("DEMO_SHARED_INSTANCE") == "1"
