import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    # A. Menghapus fitur tidak relevan
    cols_to_drop = ['id', 'dataset']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

    # B. Penanganan anomali kolesterol
    if 'chol' in df.columns:
        df['chol'] = df['chol'].replace(0, np.nan)

    # C. Menghapus kolom dengan missing values > 30%
    high_missing_cols = ['ca', 'thal', 'slope']
    df = df.drop(columns=[col for col in high_missing_cols if col in df.columns])

    # D. Imputasi Missing Values
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

    # E. Transformasi Target ke Binary
    if 'num' in df.columns:
        df['target'] = df['num'].apply(lambda x: 1 if x > 0 else 0)
        df = df.drop(columns=['num'])

    # F. Encoding Data Kategorikal
    df = pd.get_dummies(df, drop_first=True)

    # G. Standardisasi Fitur
    scaler = StandardScaler()
    target_col = 'target'
    
    if target_col in df.columns:
        features = df.drop(columns=[target_col])
        target = df[target_col]
        features_scaled = scaler.fit_transform(features)
        df_final = pd.DataFrame(features_scaled, columns=features.columns)
        df_final[target_col] = target.values
    else:
        features_scaled = scaler.fit_transform(df)
        df_final = pd.DataFrame(features_scaled, columns=df.columns)

    return df_final

if __name__ == "__main__":
    # Path relatif untuk kompatibilitas GitHub Actions
    input_path = 'heart_raw/heart.csv'
    output_dir = 'preprocessing/'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if os.path.exists(input_path):
        raw_df = pd.read_csv(input_path)
        clean_df = preprocess_data(raw_df)
        output_file = os.path.join(output_dir, 'heart_preprocessed.csv')
        clean_df.to_csv(output_file, index=False)
        print(f"Otomatisasi Berhasil. Data disimpan di: {output_file}")
    else:
        print(f"Gagal: File input tidak ditemukan di {os.getcwd()}/{input_path}")