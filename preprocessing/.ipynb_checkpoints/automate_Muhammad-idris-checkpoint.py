import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

def automate_preprocessing(input_path: str, output_path: str) -> pd.DataFrame:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File raw data tidak ditemukan di {input_path}")
    
    df = pd.read_csv(input_path)
    if ';' in open(input_path).readline():
        df = pd.read_csv(input_path, sep=';')

    df = df.drop_duplicates().dropna()

    if 'quality' in df.columns:
        df['target'] = (df['quality'] >= 6).astype(int)
        df = df.drop(columns=['quality'])

    features = df.drop(columns=['target'])
    target = df['target']

    scaler = StandardScaler()
    scaled_features = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)

    df_final = pd.concat([scaled_features, target.reset_index(drop=True)], axis=1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_final.to_csv(output_path, index=False)
    print(f"Preprocessing selesai! File disimpan ke: {output_path}")
    
    return df_final

if __name__ == "__main__":
    input_csv = "winequality_raw.csv"
    output_csv = "preprocessing/winequality_preprocessing.csv"
    automate_preprocessing(input_csv, output_csv)