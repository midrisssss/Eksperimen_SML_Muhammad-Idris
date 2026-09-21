import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

def automate_preprocessing(input_path: str, output_path: str) -> pd.DataFrame:
    if not os.path.exists(input_path):
        print(f"File {input_path} tidak ditemukan. Mengunduh data raw...")
        url = "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
        df_raw = pd.read_csv(url, sep=";")
        
        dir_name = os.path.dirname(input_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
            
        df_raw.to_csv(input_path, index=False)
        print(f"Data raw berhasil diunduh dan disimpan ke {input_path}")
    
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
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_csv = os.path.join(base_dir, "winequality_raw.csv")
    output_csv = os.path.join(base_dir, "preprocessing", "winequality_preprocessing.csv")
    
    automate_preprocessing(input_csv, output_csv)