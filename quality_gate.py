
import json
import sys

with open("artifacts/metrics.json") as file:
    metrics = json.load(file)

print("Accuracy:", metrics["accuracy"])
print("F1 Score:", metrics["f1_score"])

MIN_F1 = 0.50

if metrics["f1_score"] < MIN_F1:
    print("QUALITY GATE FAILED")
    sys.exit(1)

print("QUALITY GATE PASSED")
