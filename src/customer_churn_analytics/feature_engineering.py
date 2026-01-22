import pandas as pd

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["AvgChargesPerMonth"] = df["TotalCharges"] / (df["tenure"] + 1)

    service_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    df["TotalServices"] = (df[service_cols] == "Yes").sum(axis=1)

    df["LongTermContract"] = df["Contract"].isin(
        ["One year", "Two year"]
    ).astype(int)

    return df
