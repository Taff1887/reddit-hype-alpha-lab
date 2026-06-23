#!/usr/bin/env python
"""Run the bounded research / self-improvement loop and log all results."""
import _bootstrap  # noqa: F401

from reddit_hype.pipeline import step_research

if __name__ == "__main__":
    out = step_research()
    print("Hypothesis tests:")
    if not out["hypotheses"].empty:
        print(out["hypotheses"].to_string(index=False))
    print("\nIterations:")
    if not out["iterations"].empty:
        print(out["iterations"].to_string(index=False))
    print(f"\nFinal OOS Sharpe: {out.get('final_oos_sharpe')}")
    print(f"Final weights: {out.get('final_weights')}")
