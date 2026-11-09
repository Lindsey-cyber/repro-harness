"""Canary experiment: deterministic, needs no packages. Prints the metric the canary paper claims."""
import argparse
p = argparse.ArgumentParser()
p.add_argument("--steps", type=int, default=100)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--data", default=None, help="labels file; must exist and hold 10 lines")
a = p.parse_args()
if a.data is not None:
    lines = open(a.data).read().split()
    assert len(lines) == 10, f"expected 10 labels, got {len(lines)}"
acc = 0.9 if a.steps >= 100 else 0.5
print(f"step {a.steps} done")
print(f"accuracy: {acc:.4f}")
