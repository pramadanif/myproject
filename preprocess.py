import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import datetime

def preprocess_data(input_file):
    """
    Melakukan pra-pemrosesan data servis motor sesuai dengan persyaratan yang ditentukan
    
    Args:
        input_file (str): Path ke file CSV input
        
    Returns:
        dict: Dictionary yang berisi data yang telah di-preprocessing dan objek preprocessing
    """
    print(f"Membaca data dari: {input_file}")
    df = pd.read_csv(input_file)
    
    print(f"Bentuk DataFrame awal: {df.shape}")
    print(f"Kolom awal: {df.columns.tolist()}")
    
    # Langkah 1: Pemilihan Fitur
    # Pilih hanya kolom yang kita minati
    selected_features = [
        'Kode Unit Motor',
        'Tahun Kendaraan',
        'Kilometer',
        'Usia_Kendaraan',
        'Kondisi_Oli',
        'Kondisi_Rem',
        'KM_Terakhir_Ganti_Oli',
        'Bulan_Terakhir_Ganti_Oli',
        'KM_Per_Tahun',
        'Jumlah_Keluhan',
        'Jns Service'
    ]
    
    # Memeriksa apakah semua fitur yang dipilih ada dalam DataFrame
    missing_features = [col for col in selected_features if col not in df.columns]
    if missing_features:
        print(f"Peringatan: Fitur berikut tidak ada dalam DataFrame: {missing_features}")
        print(f"Kolom yang tersedia: {df.columns.tolist()}")
        # Filter fitur yang tidak ada
        selected_features = [col for col in selected_features if col in df.columns]
    
    # Pertahankan hanya fitur yang dipilih
    df = df[selected_features]
    print(f"DataFrame setelah pemilihan fitur: {df.shape}")
    
    # Langkah 2: Pembersihan Data - Menghapus catatan yang tidak relevan di Jns Service
    # Konversi ke huruf kecil untuk perbandingan yang tidak peka huruf besar/kecil
    df['Jns Service'] = df['Jns Service'].str.lower()
    
    # Pertahankan hanya nilai 'repair' dan 'rutin'
    # Petakan semua varian repair dan rutin ke nilai standar
    repair_patterns = ['repair', 'perbaikan', 'jual part']
    rutin_patterns = ['rutin', 'berkala', 'kupon gratis', 'gratis']
    
    def standardize_service(service):
        service = service.lower()
        if any(pattern in service for pattern in repair_patterns):
            return 'repair'
        elif any(pattern in service for pattern in rutin_patterns):
            return 'rutin'
        else:
            return 'other'
    
    df['Jns Service'] = df['Jns Service'].apply(standardize_service)
    
    # Hapus baris dengan jenis layanan 'other'
    df = df[df['Jns Service'] != 'other']
    print(f"DataFrame setelah memfilter jenis layanan: {df.shape}")
    
    # Langkah 3: Penanganan Nilai yang Hilang (Missing Value)
    
    # Fitur numerik - imputasi dengan median
    numerical_features = [
        'Tahun Kendaraan', 
        'Kilometer', 
        'Usia_Kendaraan', 
        'KM_Terakhir_Ganti_Oli',
        'Bulan_Terakhir_Ganti_Oli',
        'KM_Per_Tahun',
        'Jumlah_Keluhan'
    ]
    
    # Simpan nilai median untuk penggunaan di masa depan
    medians = {}
    for feature in numerical_features:
        if feature in df.columns:
            median_value = df[feature].median()
            medians[feature] = median_value
            df[feature] = df[feature].fillna(median_value)
    
    # Fitur kategorikal - imputasi dengan modus
    categorical_features = ['Kode Unit Motor', 'Kondisi_Oli', 'Kondisi_Rem']
    
    # Simpan modus untuk penggunaan di masa depan
    modes = {}
    for feature in categorical_features:
        if feature in df.columns:
            mode_value = df[feature].mode()[0]
            modes[feature] = mode_value
            df[feature] = df[feature].fillna(mode_value)
    
    print("Nilai yang hilang telah diimputasi.")
    
    # Langkah 4: Transformasi Data Kategorikal dengan LabelEncoder
    
    # Inisialisasi dictionary untuk menyimpan encoder label
    label_encoders = {}
    
    # Terapkan encoding label ke fitur kategorikal
    for feature in categorical_features:
        if feature in df.columns:
            label_encoders[feature] = LabelEncoder()
            df[feature] = label_encoders[feature].fit_transform(df[feature])
    
    # Encode variabel target (Jns Service)
    label_encoders['Jns Service'] = LabelEncoder()
    y = label_encoders['Jns Service'].fit_transform(df['Jns Service'])
    
    print("Variabel kategorikal telah dienkode.")
    
    # Langkah 5: Siapkan fitur dan target
    X = df.drop('Jns Service', axis=1)
    
    # Langkah 6: Pembagian Dataset (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Set pelatihan: {X_train.shape[0]} sampel")
    print(f"Set pengujian: {X_test.shape[0]} sampel")
    
    # Kembalikan data yang telah di-preprocessing dan objek preprocessing
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'label_encoders': label_encoders,
        'medians': medians,
        'modes': modes,
        'feature_list': X.columns.tolist()
    }

def train_decision_tree(preprocessed_data):
    """
    Melatih pengklasifikasi Decision Tree menggunakan data yang telah di-preprocessing
    
    Args:
        preprocessed_data (dict): Dictionary yang berisi data yang telah di-preprocessing
        
    Returns:
        tuple: Model terlatih dan metrik performa
    """
    X_train = preprocessed_data['X_train']
    X_test = preprocessed_data['X_test']
    y_train = preprocessed_data['y_train']
    y_test = preprocessed_data['y_test']
    
    # Inisialisasi dan latih model Decision Tree
    dt_model = DecisionTreeClassifier(
        criterion='gini',
        max_depth=7,
        min_samples_split=10,
        random_state=42
    )
    
    dt_model.fit(X_train, y_train)
    
    # Buat prediksi pada data uji
    y_pred = dt_model.predict(X_test)
    
    # Hitung metrik performa
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print(f"\nPerforma Model:")
    print(f"Akurasi: {accuracy:.4f}")
    print(f"\nLaporan Klasifikasi:\n{class_report}")
    print(f"\nMatriks Konfusi:\n{conf_matrix}")
    
    # Hitung kepentingan fitur
    feature_importance = pd.DataFrame({
        'Fitur': preprocessed_data['feature_list'],
        'Tingkat_Kepentingan': dt_model.feature_importances_
    }).sort_values('Tingkat_Kepentingan', ascending=False)
    
    print(f"\nTingkat Kepentingan Fitur:\n{feature_importance}")
    
    return {
        'model': dt_model,
        'accuracy': accuracy,
        'classification_report': class_report,
        'confusion_matrix': conf_matrix,
        'feature_importance': feature_importance
    }

def save_model_data(model_results, preprocessed_data, output_dir):
    """
    Menyimpan model, objek preprocessing, dan hasil
    
    Args:
        model_results (dict): Dictionary yang berisi model dan metrik performa
        preprocessed_data (dict): Dictionary yang berisi objek preprocessing
        output_dir (str): Direktori untuk menyimpan hasil
    """
    import pickle
    from datetime import datetime
    
    # Buat direktori output jika belum ada
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Simpan model
    with open(os.path.join(output_dir, 'model.pkl'), 'wb') as f:
        pickle.dump(model_results['model'], f)
    
    # Simpan objek preprocessing
    with open(os.path.join(output_dir, 'preprocessing.pkl'), 'wb') as f:
        preprocessing_data = {
            'label_encoders': preprocessed_data['label_encoders'],
            'medians': preprocessed_data['medians'],
            'modes': preprocessed_data['modes'],
            'feature_list': preprocessed_data['feature_list']
        }
        pickle.dump(preprocessing_data, f)
    
    # Simpan metrik performa sebagai teks
    with open(os.path.join(output_dir, 'performance_metrics.txt'), 'w') as f:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"Hasil Pelatihan Model - {current_time}\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Akurasi Model: {model_results['accuracy']:.4f}\n\n")
        f.write(f"Laporan Klasifikasi:\n{model_results['classification_report']}\n\n")
        f.write(f"Matriks Konfusi:\n{model_results['confusion_matrix']}\n\n")
        
        f.write("Tingkat Kepentingan Fitur:\n")
        for idx, row in model_results['feature_importance'].iterrows():
            f.write(f"{row['Fitur']}: {row['Tingkat_Kepentingan']:.6f}\n")
    
    # Simpan tingkat kepentingan fitur sebagai CSV
    model_results['feature_importance'].to_csv(
        os.path.join(output_dir, 'feature_importance.csv'),
        index=False
    )
    
    print(f"\nModel dan data preprocessing disimpan ke: {output_dir}")

if __name__ == "__main__":
    # Tentukan jalur input dan output
    input_file = os.path.join(os.path.dirname(__file__), 'indoperkasa2_modified.csv')
    output_dir = os.path.join(os.path.dirname(__file__), 'model_output')
    
    # Pra-pemrosesan data
    print("Memulai pra-pemrosesan data...")
    preprocessed_data = preprocess_data(input_file)
    
    # Melatih model
    print("\nMelatih model decision tree...")
    model_results = train_decision_tree(preprocessed_data)
    
    # Menyimpan model dan data
    print("\nMenyimpan model dan hasil...")
    save_model_data(model_results, preprocessed_data, output_dir)
    
    print("\nPra-pemrosesan dan pelatihan model selesai dengan sukses.")
