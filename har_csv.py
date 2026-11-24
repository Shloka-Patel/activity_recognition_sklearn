"""
Part A - Activity recognition from the UCI HAR feature CSVs (train.csv, test.csv) with scikit-learn.

This is a Python port of the MATLAB version, plus one experiment that fixes its weak spot
(sitting vs standing). Run:   python har_csv.py
"""
from pathlib import Path
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from common import evaluate

DATA = Path("data")
OUT = Path("results_csv")
OUT.mkdir(exist_ok=True)

TARGETS = ["WALKING", "CLIMBING STAIRS", "SITTING", "STANDING"]   # LAYING is left out, like the MATLAB code
STAIRS = {"WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS"}                # both are merged into CLIMBING STAIRS


def load(csv_name):
    """Read a CSV, merge the two stairs labels, keep only the target activities."""
    df = pd.read_csv(DATA / csv_name)
    labels = df["Activity"].astype(str).replace({s: "CLIMBING STAIRS" for s in STAIRS})
    keep = labels.isin(TARGETS)
    return df[keep].reset_index(drop=True), labels[keep].reset_index(drop=True)


def select(columns, signal_names):
    """Columns of the given signals that are a mean or a standard deviation."""
    return [c for c in columns
            if any(s in c for s in signal_names) and ("mean" in c or "std" in c)]


def run(name, signal_names, train, ytrain, test, ytest):
    cols = select(train.columns, signal_names)
    print(f"\n=== {name}: {len(cols)} features ===")
    # StandardScaler puts every feature on the same scale; SVC is the support vector machine.
    model = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10))
    model.fit(train[cols], ytrain)
    return evaluate(name, ytest, model.predict(test[cols]), OUT)


train, ytrain = load("train.csv")
test, ytest = load("test.csv")
print(f"Train windows: {len(train)}   Test windows: {len(test)}")

# 1) Same features as the MATLAB code: body acceleration only (gravity already removed)
acc_a, per_a = run("body_acc_only", ["tBodyAcc"], train, ytrain, test, ytest)
# 2) Add the gravity signals, which tell the phone's tilt (sitting vs standing)
acc_b, per_b = run("body_plus_gravity", ["tBodyAcc", "tGravityAcc"], train, ytrain, test, ytest)

summary = ["Activity recognition, scikit-learn SVM", "",
           f"Body acceleration only   : {acc_a:.2f}%",
           f"Body + gravity features  : {acc_b:.2f}%", "", "Per-activity accuracy (%):",
           f"{'activity':<18}{'body only':>10}{'+ gravity':>11}"]
summary += [f"{k:<18}{per_a[k]:>10.1f}{per_b[k]:>11.1f}" for k in per_a]
(OUT / "summary.txt").write_text("\n".join(summary))
print("\n" + "\n".join(summary))
