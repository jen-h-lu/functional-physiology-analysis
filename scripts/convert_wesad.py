import pickle
from pathlib import Path
import pandas as pd
import numpy as np


# -----------------------------
# Paths
# -----------------------------

data_dir = Path("../data/WESAD")
output_dir = Path("../data/processed")

output_dir.mkdir(exist_ok=True)


# -----------------------------
# Sampling frequency
# -----------------------------

fs = 700


# -----------------------------
# Process all subjects
# -----------------------------

subjects = sorted(data_dir.glob("S*/"))


for subject_folder in subjects:

    subject_id = subject_folder.name

    input_file = subject_folder / f"{subject_id}.pkl"


    if not input_file.exists():
        print(f"Skipping {subject_id}")
        continue


    print(f"Processing {subject_id}...")


    # Load WESAD file
    with open(input_file, "rb") as f:
        data = pickle.load(
            f,
            encoding="latin1"
        )


    # -----------------------------
    # Extract ECG
    # -----------------------------

    ecg = (
        data["signal"]
        ["chest"]
        ["ECG"]
        .flatten()
    )


    # -----------------------------
    # Extract labels
    # -----------------------------

    labels = data["label"]


    # Match lengths

    n = min(
        len(ecg),
        len(labels)
    )


    ecg = ecg[:n]
    labels = labels[:n]


    # -----------------------------
    # Create time vector
    # -----------------------------

    time = np.arange(n) / fs


    # -----------------------------
    # Create dataframe
    # -----------------------------

    df = pd.DataFrame(
        {
            "time": time,
            "ecg": ecg,
            "label": labels
        }
    )


    # -----------------------------
    # Save
    # -----------------------------

    output_file = (
        output_dir /
        f"{subject_id}_ecg.csv"
    )


    df.to_csv(
        output_file,
        index=False
    )


    print(
        f"Saved {output_file}"
    )


print("All subjects processed.")
