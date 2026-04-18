from __future__ import annotations


def edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(
                min(
                    prev[j] + 1,
                    curr[j - 1] + 1,
                    prev[j - 1] + (ca != cb),
                )
            )
        prev = curr
    return prev[-1]


def cer(prediction: str, target: str) -> float:
    if not target:
        return 0.0 if not prediction else 1.0
    return edit_distance(prediction, target) / len(target)


def accuracy(predictions: list[str], targets: list[str]) -> float:
    if not targets:
        return 0.0
    correct = sum(pred == target for pred, target in zip(predictions, targets))
    return correct / len(targets)

