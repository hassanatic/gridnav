"""python -m gridnav [--no-block] [--seed N]: run the warehouse scenario and print what happened."""

from __future__ import annotations

import argparse

from .sim import Scenario, run


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="gridnav")
    p.add_argument("--no-block", action="store_true", help="do not drop an obstacle on the path")
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args(argv)
    sc = Scenario(seed=a.seed, block_at=None if a.no_block else 4.0)
    r = run(sc)
    for tr in r.transitions:
        print(f"t={tr.t:6.1f}s  {tr.src.value:>9} -> {tr.dst.value:<9} {tr.reason}")
    print(f"\nfinal state      {r.state.value}")
    print(f"time             {r.time:.1f} s")
    print(f"distance to goal {r.position_error:.2f} m")
    print(f"max est. error   {r.max_estimate_error:.3f} m")
    print(f"mean NEES        {r.mean_nees:.2f}  (about 3 for a consistent filter)")
    print(f"replans          {r.replans}")
    return 0 if r.state.value == "arrived" else 1


if __name__ == "__main__":
    raise SystemExit(main())
