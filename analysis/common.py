from pathlib import Path
import pandas as pd

TAGS = ("EVAL-accuracy", "EVAL-f1_score", "EVAL-precision", "EVAL-recall")

OUR_ARCH_NAME = "FPTT (Ours)"

RUNS_BY_ARCH = {
    OUR_ARCH_NAME: (
        "2024-04-10T11:47:49.666942",
        "2024-04-10T15:13:38.707179",
        "2024-04-10T16:55:46.637166",
        "2024-04-10T21:04:09.311050",
        "2024-04-10T23:15:38.166338",
    ),
    "STEVE": (
        "2024-04-08T13:58:49.159613",
        "2024-04-08T15:11:53.796694",
        "2024-04-08T16:26:55.925857",
        "2024-04-09T10:08:04.981652",
        "2024-04-09T13:50:29.125304",
    ),
    "Decoder only": (
        "2024-04-11T15:27:24.052067",
        "2024-04-11T19:02:29.148904",
        "2024-04-11T23:23:55.357344",
        "2024-04-12T08:00:24.739384",
        "2024-04-12T10:18:23.206939",
    ),
}

def read_run(run, tag):
    return pd.read_csv(
        Path("../logs") / Path(run) / Path(tag).with_suffix(".csv")
    )

def calculate_line(tag, arch):
    dataframes = []
    for run in RUNS_BY_ARCH[arch]:
        dataframes.append(read_run(run, tag))
    data = pd.concat(dataframes).groupby("step", as_index=False)

    mean = data.mean(numeric_only=True)
    sem = data.sem(numeric_only=True)

    return mean, sem