import pandas as pd
import numpy as np


def clean_data(df):
    # Remove rows with missing values
    df = df.dropna()

    # Convert duration to seconds
    df['Average view duration'] = df['Average view duration'].apply(
        lambda x: sum(int(i) * 60 ** idx for idx, i in enumerate(reversed(str(x).split(':'))))
    )

    return df
