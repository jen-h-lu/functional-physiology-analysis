from pathlib import Path
import pandas as pd
from scipy.signal import butter, filtfilt


input_dir = Path("../data/processed")
output_dir = Path("../data/processed")

fs = 700


def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = fs / 2
    b, a = butter(
        order,
        [lowcut / nyq, highcut / nyq],
        btype="band"
    )
    return b, a


b, a = butter_bandpass(
    0.5,
    40,
    fs
)


files = sorted(
    input_dir.glob("S*_ecg.csv")
)


for file in files:

    subject = file.stem.replace("_ecg", "")

    print(f"Processing {subject}")

    df = pd.read_csv(file)

    filtered = filtfilt(
        b,
        a,
        df["ecg"]
    )

    df["filtered_ecg"] = filtered

    output = output_dir / f"{subject}_ecg_filtered.csv"

    df.to_csv(
        output,
        index=False
    )

    print(f"Saved {output}")
