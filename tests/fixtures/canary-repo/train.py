"""Canary experiment: deterministic, needs no packages. Prints the metric the canary paper claims."""
import argparse
p = argparse.ArgumentParser()
p.add_argument("--steps", type=int, default=100)
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()
acc = 0.9 if a.steps >= 100 else 0.5
print(f"step {a.steps} done")
print(f"accuracy: {acc:.4f}")
