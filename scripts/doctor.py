#!/usr/bin/env python
"""Pre-flight connectivity check for Reddit + FMP credentials.

Run this FIRST after filling in .env — it pings each live API with one minimal
request so a bad key fails in seconds, not deep inside the pipeline.
"""
import sys

import _bootstrap  # noqa: F401

from reddit_hype.pipeline import doctor

if __name__ == "__main__":
    sys.exit(0 if doctor() else 1)
