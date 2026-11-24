"""Small helpers shared by both scripts: metrics and the two result plots."""
import numpy as np
import matplotlib
matplotlib.use("Agg")                      # draw to files, no window needed
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

CLASSES = ["CLIMBING STAIRS", "SITTING", "STANDING", "WALKING"]


def evaluate(tag, ytest, pred, out_dir):
    """Print accuracy, save confusion matrix and per-activity accuracy plots."""
    acc = (np.asarray(ytest) == np.asarray(pred)).mean() * 100
    print(f"\n[{tag}] Overall accuracy: {acc:.2f}%")
    print(classification_report(ytest, pred, labels=CLASSES, digits=3, zero_division=0))

    cm = confusion_matrix(ytest, pred, labels=CLASSES)
    per_class = cm.diagonal() / cm.sum(axis=1) * 100      # recall of each activity

    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(ax=ax, cmap="Blues", xticks_rotation=25,
                                                            colorbar=False)
    ax.set_title(f"Confusion matrix: {tag}")
    fig.tight_layout()
    fig.savefig(out_dir / f"{tag}_confusion.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(CLASSES, per_class, color="#3399cc")
    for i, v in enumerate(per_class):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Per-activity accuracy: {tag}")
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=15)
    fig.tight_layout()
    fig.savefig(out_dir / f"{tag}_per_activity.png", dpi=150)
    plt.close(fig)
    return acc, dict(zip(CLASSES, per_class))
