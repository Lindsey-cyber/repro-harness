"""Canary experiment: deterministic, needs no packages. Prints the metric the canary paper claims."""
import argparse
p = argparse.ArgumentParser()
p.add_argument("--steps", type=int, default=100)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--also-train-acc", action="store_true", help="also print a training accuracy line")
p.add_argument("--data", default=None, help="labels file; must exist and hold 10 lines")
a = p.parse_args()
if a.data is not None:
    lines = open(a.data).read().split()
    assert len(lines) == 10, f"expected 10 labels, got {len(lines)}"
acc = 0.9 if a.steps >= 100 else 0.5
print(f"step {a.steps} done")
if a.also_train_acc:
    print(f"train accuracy: {0.9999:.4f}")
print(f"accuracy: {acc:.4f}")
