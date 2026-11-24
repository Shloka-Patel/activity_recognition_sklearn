# Human Activity Recognition with Scikit-learn

Classifies four activities (walking, climbing stairs, sitting, standing) from smartphone
accelerometer data, using the UCI HAR dataset. Python port of an earlier MATLAB project.

| Script | What it does |
|---|---|
| `har_csv.py` | Uses the dataset's ready-made feature CSVs. Compares body-acceleration-only features with body + gravity features (SVM). |
| `har_dsp.py` | Uses the raw 50 Hz acceleration windows. Separates gravity, low-pass filters (Butterworth), builds time and FFT features by hand, classifies with SVM and Random Forest. |

## Run
```
pip install -r requirements.txt
python har_csv.py     # needs data/train.csv and data/test.csv
python har_dsp.py     # downloads the raw dataset automatically
```
Results (confusion matrices, per-activity accuracy, summary) are saved in `results_csv/` and `results_dsp/`.

## Results
(fill in after running)
