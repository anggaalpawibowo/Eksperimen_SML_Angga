import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def create_directory(directory_path):
    """Fungsi untuk memastikan folder output sudah dibuat"""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Folder '{directory_path}' berhasil dibuat.")

def load_data(file_path):
    """Fungsi untuk memuat dataset mentah"""
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} tidak ditemukan!")
        sys.exit(1)
    print(f"Memuat data dari: {file_path}")
    return pd.read_csv(file_path)

def clean_data(df):
    """Fungsi otomatisasi pembersihan: missing values dan duplikat"""
    df_clean = df.copy()
    
    # 1. Menangani Missing Values pada kolom numerik dengan Median
    numeric_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            
    # 2. Menghapus data duplikat (jika ada)
    duplicate_count = df_clean.duplicated().sum()
    if duplicate_count > 0:
        df_clean = df_clean.drop_duplicates()
        print(f"Berhasil menghapus {duplicate_count} data duplikat.")
        
    return df_clean

def preprocess_features(df):
    """Fungsi otomatisasi rekayasa fitur (Drop ID, Encoding, & Scaling)"""
    df_processed = df.copy()
    
    # 1. Drop fitur tidak relevan (CustomerID)
    if 'CustomerID' in df_processed.columns:
        df_processed = df_processed.drop(columns=['CustomerID'])
        print("Fitur 'CustomerID' berhasil dibuang.")
        
    # 2. One-Hot Encoding untuk fitur kategorikal sesuai kolom Anda
    categorical_cols = ['Gender', 'Subscription Type', 'Contract Length']
    df_encoded = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True)
    print("Proses One-Hot Encoding selesai.")
    
    # 3. Standard Scaling untuk semua fitur numerik (kecuali kolom target 'Churn')
    if 'Churn' in df_encoded.columns:
        X = df_encoded.drop(columns=['Churn'])
        y = df_encoded['Churn']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Kembalikan ke dalam bentuk DataFrame agar mudah disimpan ke CSV
        df_final = pd.DataFrame(X_scaled, columns=X.columns)
        df_final['Churn'] = y.values
    else:
        scaler = StandardScaler()
        df_final = pd.DataFrame(scaler.fit_transform(df_encoded), columns=df_encoded.columns)
        
    print("Proses Feature Scaling selesai.")
    return df_final

def main():
    print("=== Memulai Pipa Otomatisasi Preprocessing Data ===")
    
    # Menentukan folder input dan output sesuai standarisasi Dicoding
    raw_data_dir = "customerchurn_raw"                         
    processed_data_dir = "namadataset_preprocessing"         
    input_file_name = "customerchurn_raw/customer_churn_dataset-training-master.csv"                    
    
    input_path = os.path.join(raw_data_dir, input_file_name)
    output_path = os.path.join(processed_data_dir, "cleaned_customer_churn.csv")
    
    # Menjalankan urutan pipa data otomatis
    create_directory(processed_data_dir)
    raw_df = load_data(input_path)
    cleaned_df = clean_data(raw_df)
    final_df = preprocess_features(cleaned_df)
    
    # Menyimpan data hasil transformasi ke folder baru
    final_df.to_csv(output_path, index=False)
    print(f"\nSukses! Data siap pakai disimpan di: {output_path}")
    print(f"Dimensi data akhir: {final_df.shape}")
    print("=== Pipa Otomatisasi Selesai Dilaksanakan ===")

if __name__ == "__main__":
    main()
