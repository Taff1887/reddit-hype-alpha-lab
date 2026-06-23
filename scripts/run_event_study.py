#!/usr/bin/env python
"""Run the forward-return event study around abnormal-attention spikes."""
import _bootstrap  # noqa: F401

from reddit_hype.pipeline import step_event_study

if __name__ == "__main__":
    out = step_event_study()
    es = out["event_study"]
    print(f"Event study ({es.attrs.get('n_events', '?')} events):")
    print(es.to_string(index=False))
    print("\nAlready-ran diagnostic:", out["already_ran"])

    decay = out.get("decay_attention")
    if decay is not None and not decay.empty:
        print("\n=== Hype-decay / reversal study (top vs bottom attention decile) ===")
        print(decay.to_string(index=False))
        print("VERDICT:", decay.attrs.get("verdict"))
