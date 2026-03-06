def get_features_and_target(df):
    X = df.drop(columns=["reversed"])
    y = df["reversed"]
    return X, y