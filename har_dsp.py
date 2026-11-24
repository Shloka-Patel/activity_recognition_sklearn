"""
Part B - Activity recognition with real signal processing on the raw accelerometer windows.

Each sample is a 2.56 s window (128 readings at 50 Hz) of the phone's acceleration. We separate
gravity from body motion, low-pass filter, and build time-domain and frequency-domain (FFT)
features ourselves, then classify with scikit-learn.  Run:   python har_dsp.py
"""
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from common import CLASSES, evaluate

DATA = Path("data")
ROOT = DATA / "UCI HAR Dataset"
OUT = Path("results_dsp")
OUT.mkdir(exist_ok=True)
URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"

FS = 50                                                  # sampling rate in Hz
SOS = butter(4, 10, btype="low", fs=FS, output="sos")    # 4th-order Butterworth low-pass at 10 Hz
LABELS = {1: "WALKING", 2: "CLIMBING STAIRS", 3: "CLIMBING STAIRS", 4: "SITTING", 5: "STANDING"}  # 6 = LAYING (dropped)


def ensure_dataset():
    """Download and unzip the UCI HAR dataset once."""
    if (ROOT / "train" / "Inertial Signals").exists():
        return
    DATA.mkdir(exist_ok=True)
    zpath = DATA / "uci_har.zip"
    print("Downloading the UCI HAR dataset (about 60 MB)...")
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r, open(zpath, "wb") as f:
        f.write(r.read())
    with zipfile.ZipFile(zpath) as z:
        z.extractall(DATA)
    for inner in list(DATA.glob("*.zip")):               # the download contains a second zip
        if inner != zpath:
            with zipfile.ZipFile(inner) as z:
                z.extractall(DATA)
    if not (ROOT / "train" / "Inertial Signals").exists():
        sys.exit("Could not find 'UCI HAR Dataset' after unzipping. Download it by hand from "
                 "archive.ics.uci.edu (dataset 240) and unzip it into the data/ folder.")


def load_split(split):
    """Return windows (n, 128, 3) of total acceleration in g, and the activity names."""
    base = ROOT / split / "Inertial Signals"
    axes = [pd.read_csv(base / f"total_acc_{a}_{split}.txt", sep=r"\s+", header=None).to_numpy() for a in "xyz"]
    windows = np.stack(axes, axis=2)
    y = pd.read_csv(ROOT / split / f"y_{split}.txt", header=None)[0].to_numpy()
    keep = np.isin(y, list(LABELS))
    return windows[keep], np.array([LABELS[v] for v in y[keep]])


def split_signal(windows):
    """Gravity = the slow part (window average). Body motion = what is left, smoothed by a low-pass filter."""
    gravity = windows.mean(axis=1)                               # (n, 3): phone orientation
    body = windows - gravity[:, None, :]                         # remove gravity
    body = sosfiltfilt(SOS, body, axis=1)                        # remove high-frequency noise
    return gravity, body


def features(windows):
    """Turn each 128 x 3 window into one row of 35 numbers."""
    gravity, body = split_signal(windows)
    mag = np.linalg.norm(body, axis=2)                           # overall motion strength
    sig = np.concatenate([body, mag[:, :, None]], axis=2)        # x, y, z, magnitude  -> (n, 128, 4)

    centred = sig - sig.mean(axis=1, keepdims=True)
    zcr = (np.diff(np.signbit(centred).astype(int), axis=1) != 0).mean(axis=1)   # zero-crossing rate

    jerk = np.diff(body, axis=1) * FS                            # how fast the acceleration changes
    jerk_mag = np.linalg.norm(jerk, axis=2)

    spec = np.abs(np.fft.rfft((mag - mag.mean(axis=1, keepdims=True)) * np.hanning(mag.shape[1]), axis=1)) ** 2
    freqs = np.fft.rfftfreq(mag.shape[1], d=1 / FS)
    energy = spec.sum(axis=1) + 1e-12
    p = spec / energy[:, None]
    bands = [(0.3, 1.0), (1.0, 3.0), (3.0, 10.0)]
    band_share = [spec[:, (freqs >= lo) & (freqs < hi)].sum(axis=1) / energy for lo, hi in bands]

    return np.column_stack([
        gravity,                                                 # 3  orientation (sitting vs standing)
        sig.std(axis=1), sig.max(axis=1), sig.min(axis=1),       # 12
        np.sqrt((sig ** 2).mean(axis=1)),                        # 4  RMS
        zcr,                                                     # 4
        jerk.std(axis=1), jerk_mag.mean(axis=1), jerk_mag.std(axis=1), jerk_mag.max(axis=1),   # 6
        np.log10(energy),                                        # 1  FFT energy
        freqs[spec.argmax(axis=1)],                              # 1  dominant frequency
        -(p * np.log2(p + 1e-12)).sum(axis=1),                   # 1  spectral entropy
        *band_share,                                             # 3  energy in 3 frequency bands
    ])


def demo_plot(windows, y):
    """One example window per activity: motion in time and its frequency spectrum."""
    fig, axes = plt.subplots(len(CLASSES), 2, figsize=(10, 8))
    t = np.arange(windows.shape[1]) / FS
    for row, name in enumerate(CLASSES):
        w = windows[np.where(y == name)[0][0]][None]
        _, body = split_signal(w)
        mag = np.linalg.norm(body[0], axis=1)
        spec = np.abs(np.fft.rfft((mag - mag.mean()) * np.hanning(len(mag))))
        axes[row, 0].plot(t, mag)
        axes[row, 0].set_ylabel(name, fontsize=8)
        axes[row, 1].plot(np.fft.rfftfreq(len(mag), 1 / FS), spec)
    axes[0, 0].set_title("Body motion magnitude (g) vs time (s)")
    axes[0, 1].set_title("Frequency spectrum (Hz)")
    fig.tight_layout()
    fig.savefig(OUT / "dsp_demo.png", dpi=150)
    plt.close(fig)


ensure_dataset()
Xw_train, ytrain = load_split("train")
Xw_test, ytest = load_split("test")
print(f"Train windows: {len(ytrain)}   Test windows: {len(ytest)}")
demo_plot(Xw_train, ytrain)

Xtrain, Xtest = features(Xw_train), features(Xw_test)
print(f"Features per window: {Xtrain.shape[1]}")

models = {
    "dsp_svm": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10)),
    "dsp_random_forest": RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1),
}
lines = ["Activity recognition with DSP features (train and test subjects are different people)", ""]
for tag, model in models.items():
    model.fit(Xtrain, ytrain)
    acc, per = evaluate(tag, ytest, model.predict(Xtest), OUT)
    lines.append(f"{tag:<20}{acc:.2f}%   " + "  ".join(f"{k}: {v:.1f}%" for k, v in per.items()))
(OUT / "summary.txt").write_text("\n".join(lines))
print("\n" + "\n".join(lines))
