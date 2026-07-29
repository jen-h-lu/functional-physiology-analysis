import pandas as pd
import numpy as np
from pathlib import Path
from scipy.signal import butter, filtfilt, find_peaks


processed_dir = Path("../data/processed")

fs = 700


subjects = sorted(
    processed_dir.glob("*_ecg.csv")
)


for file in subjects:

    subject_id = file.stem.replace("_ecg","")

    print(f"Processing {subject_id}")


    # Load ECG
    df = pd.read_csv(file)


    ecg = df["ecg"].values


    # Bandpass filter
    b, a = butter(
        4,
        [0.5/(fs/2),40/(fs/2)],
        btype="band"
    )


    filtered = filtfilt(
        b,
        a,
        ecg
    )


    # R peak detection
    peaks,_ = find_peaks(
        filtered,
        distance=350,
        height=0.5
    )


    peak_times = df["time"].iloc[peaks].values


    # RR intervals

    rr = np.diff(peak_times)

    rr = rr[
        (rr > 0.3) &
        (rr < 2)
    ]


    hr = 60 / rr


    hr_df = pd.DataFrame({

        "time": peak_times[1:len(hr)+1],

        "heart_rate": hr

    })


    output = processed_dir / f"{subject_id}_heart_rate.csv"


    hr_df.to_csv(
        output,
        index=False
    )


    print(
        f"Saved {output}"
    )
