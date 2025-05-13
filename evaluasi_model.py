import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.model_selection import cross_val_score, StratifiedKFold
import joblib
from datetime import datetime

class EvaluasiModelServisMotor:
    """
    Kelas untuk mengevaluasi model sistem pendukung keputusan servis motor
    """
    
    def __init__(self, model_dir=None, data_file=None):
        """
        Inisialisasi evaluator
        
        Args:
            model_dir (str): Direktori tempat model dan data preprocessing disimpan
            data_file (str): Path ke file data CSV untuk pengujian tambahan
        """
        self.model_dir = model_dir or os.path.join(os.path.dirname(__file__), 'model_output')
        self.data_file = data_file or os.path.join(os.path.dirname(__file__), 'indoperkasa2_modified.csv')
        
        # Muat model dan preprocessing data
        self.model = None
        self.preprocessing_data = None
        self.load_model_and_preprocessing()
        
        # Metrik evaluasi
        self.metrics = {}
        
    def load_model_and_preprocessing(self):
        """
        Memuat model dan data preprocessing dari file yang disimpan
        """
        try:
            # Muat model
            model_path = os.path.join(self.model_dir, 'model.pkl')
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                print(f"Model berhasil dimuat dari {model_path}")
            else:
                print(f"File model tidak ditemukan di {model_path}")
                
            # Muat data preprocessing
            preprocessing_path = os.path.join(self.model_dir, 'preprocessing.pkl')
            if os.path.exists(preprocessing_path):
                with open(preprocessing_path, 'rb') as f:
                    self.preprocessing_data = pickle.load(f)
                print(f"Data preprocessing berhasil dimuat dari {preprocessing_path}")
            else:
                print(f"File preprocessing tidak ditemukan di {preprocessing_path}")
                
        except Exception as e:
            print(f"Terjadi kesalahan saat memuat model dan preprocessing: {e}")
    
    def preprocess_testing_data(self):
        """
        Melakukan preprocessing pada data pengujian tambahan
        
        Returns:
            tuple: X_test, y_test setelah preprocessing
        """
        try:
            # Muat data
            df = pd.read_csv(self.data_file)
            print(f"Data berhasil dimuat dari {self.data_file}")
            
            # Periksa apakah ada kolom yang diperlukan
            feature_list = self.preprocessing_data['feature_list']
            missing_features = [col for col in feature_list if col not in df.columns]
            if missing_features:
                print(f"Peringatan: Fitur berikut tidak ditemukan dalam data: {missing_features}")
            
            # Preprocessing data
            # 1. Filter hanya jenis layanan 'repair' dan 'rutin'
            df['Jns Service'] = df['Jns Service'].str.lower()
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
            df = df[df['Jns Service'] != 'other']
            
            # 2. Handle missing values
            label_encoders = self.preprocessing_data['label_encoders']
            medians = self.preprocessing_data['medians']
            modes = self.preprocessing_data['modes']
            
            # Impute missing values in numerical features
            for feature, median in medians.items():
                if feature in df.columns:
                    df[feature] = df[feature].fillna(median)
            
            # Impute missing values in categorical features
            for feature, mode in modes.items():
                if feature in df.columns:
                    df[feature] = df[feature].fillna(mode)
            
            # 3. Encode categorical features
            for feature, encoder in label_encoders.items():
                if feature in df.columns and feature != 'Jns Service':
                    # Handle unknown categories
                    try:
                        df[feature] = encoder.transform(df[feature])
                    except ValueError:
                        print(f"Peringatan: Kategori baru ditemukan di kolom {feature}. Menggunakan nilai default.")
                        most_common_value = 0  # Default ke kategori pertama
                        df[feature] = most_common_value
            
            # 4. Prepare features and target
            # Encode target
            y = label_encoders['Jns Service'].transform(df['Jns Service'])
            X = df[feature_list]
            
            return X, y
            
        except Exception as e:
            print(f"Terjadi kesalahan saat preprocessing data pengujian: {e}")
            return None, None
    
    def evaluasi_model(self, X_test=None, y_test=None):
        """
        Mengevaluasi performa model menggunakan berbagai metrik
        
        Args:
            X_test (DataFrame): Fitur pengujian
            y_test (Series): Target pengujian
            
        Returns:
            dict: Hasil metrik evaluasi
        """
        if X_test is None or y_test is None:
            X_test, y_test = self.preprocess_testing_data()
            
        if X_test is None or y_test is None or self.model is None:
            print("Data pengujian atau model tidak tersedia untuk evaluasi.")
            return {}
        
        # Prediksi
        y_pred = self.model.predict(X_test)
        
        # Menghitung metrik
        self.metrics['accuracy'] = accuracy_score(y_test, y_pred)
        self.metrics['precision'] = precision_score(y_test, y_pred, average='weighted')
        self.metrics['recall'] = recall_score(y_test, y_pred, average='weighted')
        self.metrics['f1_score'] = f1_score(y_test, y_pred, average='weighted')
        self.metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred)
        self.metrics['classification_report'] = classification_report(y_test, y_pred)
        
        # Validasi silang
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.model, X_test, y_test, cv=cv, scoring='accuracy')
        self.metrics['cv_mean_accuracy'] = np.mean(cv_scores)
        self.metrics['cv_std_accuracy'] = np.std(cv_scores)
        
        # Simpan data untuk visualisasi
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = y_pred
        
        print("\nHasil Evaluasi Model:")
        print(f"Akurasi: {self.metrics['accuracy']:.4f}")
        print(f"Presisi (weighted): {self.metrics['precision']:.4f}")
        print(f"Recall (weighted): {self.metrics['recall']:.4f}")
        print(f"F1-Score (weighted): {self.metrics['f1_score']:.4f}")
        print(f"Akurasi Validasi Silang (mean): {self.metrics['cv_mean_accuracy']:.4f} ± {self.metrics['cv_std_accuracy']:.4f}")
        print("\nLaporan Klasifikasi:")
        print(self.metrics['classification_report'])
        
        return self.metrics
    
    def visualisasi_hasil(self, output_dir=None):
        """
        Membuat visualisasi hasil evaluasi
        
        Args:
            output_dir (str): Direktori untuk menyimpan visualisasi
        """
        if not hasattr(self, 'X_test') or not hasattr(self, 'y_test') or not hasattr(self, 'y_pred'):
            print("Data hasil evaluasi tidak tersedia. Jalankan evaluasi_model terlebih dahulu.")
            return
            
        output_dir = output_dir or os.path.join(self.model_dir, 'evaluasi')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # 1. Confusion Matrix
        plt.figure(figsize=(10, 8))
        labels = self.preprocessing_data['label_encoders']['Jns Service'].classes_
        cm = self.metrics['confusion_matrix']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title('Matriks Konfusi')
        plt.xlabel('Prediksi')
        plt.ylabel('Aktual')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
        plt.close()
        
        # 2. Feature Importance
        plt.figure(figsize=(12, 8))
        feature_importance = pd.DataFrame({
            'Fitur': self.preprocessing_data['feature_list'],
            'Kepentingan': self.model.feature_importances_
        }).sort_values('Kepentingan', ascending=False)
        
        sns.barplot(x='Kepentingan', y='Fitur', data=feature_importance, palette='viridis')
        plt.title('Tingkat Kepentingan Fitur')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
        plt.close()
        
        # 3. Cross-validation scores
        plt.figure(figsize=(10, 6))
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.model, self.X_test, self.y_test, cv=cv, scoring='accuracy')
        
        plt.bar(range(1, len(cv_scores) + 1), cv_scores, color='skyblue')
        plt.axhline(y=np.mean(cv_scores), color='red', linestyle='--', label=f'Rata-rata: {np.mean(cv_scores):.4f}')
        plt.title('Akurasi Validasi Silang (5-fold)')
        plt.xlabel('Fold')
        plt.ylabel('Akurasi')
        plt.xticks(range(1, len(cv_scores) + 1))
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cross_validation.png'))
        plt.close()
        
        print(f"Visualisasi hasil disimpan di {output_dir}")
    
    def buat_laporan_evaluasi(self, output_dir=None):
        """
        Membuat laporan evaluasi lengkap
        
        Args:
            output_dir (str): Direktori untuk menyimpan laporan
        """
        if not self.metrics:
            print("Tidak ada metrik evaluasi. Jalankan evaluasi_model terlebih dahulu.")
            return
            
        output_dir = output_dir or os.path.join(self.model_dir, 'evaluasi')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        report_path = os.path.join(output_dir, 'laporan_evaluasi.txt')
        
        with open(report_path, 'w') as f:
            # Header
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"Laporan Evaluasi Model Sistem Pendukung Keputusan Servis Motor\n")
            f.write(f"Tanggal: {current_time}\n")
            f.write("=" * 70 + "\n\n")
            
            # Ringkasan Model
            f.write("1. RINGKASAN MODEL\n")
            f.write("-" * 70 + "\n")
            if hasattr(self.model, 'get_params'):
                f.write(f"Tipe Model: {type(self.model).__name__}\n")
                f.write(f"Parameter Model: {self.model.get_params()}\n\n")
            
            # Metrik Evaluasi
            f.write("2. METRIK EVALUASI\n")
            f.write("-" * 70 + "\n")
            f.write(f"Akurasi: {self.metrics['accuracy']:.4f}\n")
            f.write(f"Presisi (weighted): {self.metrics['precision']:.4f}\n")
            f.write(f"Recall (weighted): {self.metrics['recall']:.4f}\n")
            f.write(f"F1-Score (weighted): {self.metrics['f1_score']:.4f}\n")
            f.write(f"Akurasi Validasi Silang (mean): {self.metrics['cv_mean_accuracy']:.4f} ± {self.metrics['cv_std_accuracy']:.4f}\n\n")
            
            # Laporan Klasifikasi
            f.write("3. LAPORAN KLASIFIKASI DETIL\n")
            f.write("-" * 70 + "\n")
            f.write(f"{self.metrics['classification_report']}\n\n")
            
            # Matriks Konfusi
            f.write("4. MATRIKS KONFUSI\n")
            f.write("-" * 70 + "\n")
            cm = self.metrics['confusion_matrix']
            labels = self.preprocessing_data['label_encoders']['Jns Service'].classes_
            
            # Format confusion matrix for text
            f.write("    " + " ".join([f"{label:^10}" for label in labels]) + "\n")
            for i, label in enumerate(labels):
                f.write(f"{label:4} " + " ".join([f"{cm[i, j]:^10}" for j in range(len(labels))]) + "\n")
            f.write("\n")
            
            # Interpretasi Hasil
            f.write("5. INTERPRETASI HASIL\n")
            f.write("-" * 70 + "\n")
            
            # Accuracy interpretation
            if self.metrics['accuracy'] >= 0.9:
                acc_interp = "sangat baik"
            elif self.metrics['accuracy'] >= 0.8:
                acc_interp = "baik"
            elif self.metrics['accuracy'] >= 0.7:
                acc_interp = "cukup"
            else:
                acc_interp = "kurang baik"
                
            f.write(f"Model memiliki tingkat akurasi yang {acc_interp} ({self.metrics['accuracy']:.2%}).\n")
            
            # Feature importance interpretation
            if hasattr(self.model, 'feature_importances_'):
                features = self.preprocessing_data['feature_list']
                importances = self.model.feature_importances_
                sorted_idx = np.argsort(importances)[::-1]
                
                f.write("Fitur paling berpengaruh dalam model (berurutan):\n")
                for i in range(min(5, len(features))):
                    feature_idx = sorted_idx[i]
                    f.write(f"- {features[feature_idx]}: {importances[feature_idx]:.4f}\n")
            
            f.write("\n")
            
            # Rekomendasi
            f.write("6. REKOMENDASI\n")
            f.write("-" * 70 + "\n")
            
            if self.metrics['accuracy'] < 0.7:
                f.write("Berdasarkan hasil evaluasi, model dapat ditingkatkan dengan:\n")
                f.write("- Menambah data training untuk meningkatkan performa\n")
                f.write("- Melakukan feature engineering untuk menemukan fitur yang lebih prediktif\n")
                f.write("- Mencoba algoritma klasifikasi lain seperti Random Forest atau Gradient Boosting\n")
            else:
                f.write("Model sudah memiliki performa yang baik dan dapat digunakan untuk sistem pendukung keputusan.\n")
                f.write("Untuk pemeliharaan model, direkomendasikan:\n")
                f.write("- Memperbarui model secara berkala dengan data baru\n")
                f.write("- Melakukan monitoring terhadap performa model\n")
                f.write("- Memvalidasi hasil prediksi dengan pendapat teknisi ahli\n")
            
            f.write("\n")
            
            # Kesimpulan
            f.write("7. KESIMPULAN\n")
            f.write("-" * 70 + "\n")
            f.write("Berdasarkan evaluasi yang telah dilakukan, model sistem pendukung keputusan servis motor ini ")
            
            if self.metrics['accuracy'] >= 0.8:
                f.write("dapat diandalkan untuk membantu menentukan jenis layanan yang diperlukan. ")
            else:
                f.write("cukup baik, tetapi masih memerlukan perbaikan lebih lanjut. ")
                
            f.write("Sistem pendukung keputusan ini dapat membantu teknisi dalam memberikan rekomendasi jenis layanan yang tepat berdasarkan berbagai kondisi kendaraan dan riwayat perawatan.\n\n")
            
            f.write("=" * 70 + "\n")
            f.write("Laporan disiapkan oleh Sistem Evaluasi Model Servis Motor\n")
            
        print(f"Laporan evaluasi lengkap disimpan di {report_path}")
    
    def prediksi_rekomendasi(self, data_kendaraan):
        """
        Membuat prediksi dan rekomendasi untuk sebuah kendaraan baru
        
        Args:
            data_kendaraan (dict): Data kendaraan untuk prediksi
            
        Returns:
            dict: Hasil prediksi dan rekomendasi
        """
        if self.model is None or self.preprocessing_data is None:
            print("Model atau data preprocessing tidak tersedia")
            return None
            
        try:
            # Convert input data to DataFrame
            input_df = pd.DataFrame([data_kendaraan])
            
            # Preprocessing input data
            label_encoders = self.preprocessing_data['label_encoders']
            feature_list = self.preprocessing_data['feature_list']
            
            # Handle categorical features
            for feature, encoder in label_encoders.items():
                if feature in input_df.columns and feature != 'Jns Service':
                    try:
                        input_df[feature] = encoder.transform(input_df[feature])
                    except ValueError:
                        print(f"Peringatan: Kategori baru ditemukan di kolom {feature}. Menggunakan nilai default.")
                        most_common_value = 0  # Default ke kategori pertama
                        input_df[feature] = most_common_value
            
            # Check for missing features
            for feature in feature_list:
                if feature not in input_df.columns:
                    print(f"Fitur {feature} tidak ditemukan dalam data input. Menggunakan nilai default.")
                    if feature in self.preprocessing_data['medians']:
                        input_df[feature] = self.preprocessing_data['medians'][feature]
                    elif feature in self.preprocessing_data['modes']:
                        input_df[feature] = self.preprocessing_data['modes'][feature]
                    else:
                        input_df[feature] = 0  # Default fallback
            
            # Reorder columns to match model's expected features
            input_df = input_df[feature_list]
            
            # Make prediction
            prediction_idx = self.model.predict(input_df)[0]
            prediction_proba = self.model.predict_proba(input_df)[0]
            
            # Convert prediction index to label
            jns_service = label_encoders['Jns Service'].inverse_transform([prediction_idx])[0]
            confidence = prediction_proba[prediction_idx]
            
            # Determine parts recommendation
            parts_recommendation = self.buat_rekomendasi_part(data_kendaraan)
            
            # Create result
            result = {
                'jenis_servis': jns_service,
                'kepercayaan': float(confidence),
                'rekomendasi_part': parts_recommendation
            }
            
            return result
            
        except Exception as e:
            print(f"Terjadi kesalahan saat membuat prediksi: {e}")
            return None
    
    def buat_rekomendasi_part(self, data_kendaraan):
        """
        Membuat rekomendasi part berdasarkan kilometer dan kondisi kendaraan
        
        Args:
            data_kendaraan (dict): Data kendaraan
            
        Returns:
            dict: Rekomendasi part
        """
        try:
            # Definisi interval penggantian part berdasarkan transmisi
            parts_intervals = {
                'matic': {
                    'Kampas Rem': 15000,
                    'Kampas Kopling CVT': 20000,
                    'V-Belt': 25000,
                    'Filter Udara': 10000,
                    'Busi': 8000,
                    'Roller CVT': 20000,
                    'Oli Mesin': 2000
                },
                'manual': {
                    'Kampas Rem': 15000,
                    'Rantai': 20000,
                    'Gir Depan': 25000,
                    'Gir Belakang': 25000,
                    'Filter Udara': 10000,
                    'Busi': 8000,
                    'Oli Mesin': 2000
                }
            }
            
            # Menentukan jenis transmisi berdasarkan Kode Unit Motor
            kode_motor = data_kendaraan.get('Kode Unit Motor', '').upper()
            
            # Tentukan transmisi berdasarkan kode motor (contoh logika sederhana)
            transmisi = 'matic'  # Default ke matic
            if any(manual_type in kode_motor for manual_type in ['VIXION', 'R15', 'XSR', 'MT']):
                transmisi = 'manual'
            
            kilometer = data_kendaraan.get('Kilometer', 0)
            
            # Membuat rekomendasi part
            rekomendasi = {}
            parts = parts_intervals[transmisi]
            
            for part, interval in parts.items():
                if kilometer >= interval:
                    jumlah_interval = kilometer // interval
                    km_interval_terdekat = jumlah_interval * interval
                    km_interval_berikutnya = (jumlah_interval + 1) * interval
                    status = 'Perlu Penggantian'
                else:
                    km_interval_terdekat = 0
                    km_interval_berikutnya = interval
                    status = 'Belum Perlu'
                
                rekomendasi[part] = {
                    'interval': interval,
                    'km_terdekat': km_interval_terdekat,
                    'km_berikutnya': km_interval_berikutnya,
                    'status': status
                }
            
            # Tambahkan rekomendasi khusus berdasarkan kondisi
            if 'Kondisi_Oli' in data_kendaraan:
                kondisi_oli = data_kendaraan['Kondisi_Oli']
                if kondisi_oli in ['Perlu Ganti', 'Kritis']:
                    rekomendasi['Oli Mesin']['status'] = 'Perlu Penggantian Segera'
            
            if 'Kondisi_Rem' in data_kendaraan:
                kondisi_rem = data_kendaraan['Kondisi_Rem']
                if kondisi_rem in ['Perlu Penggantian', 'Perlu Penyetelan']:
                    rekomendasi['Kampas Rem']['status'] = 'Perlu Pengecekan'
            
            return rekomendasi
            
        except Exception as e:
            print(f"Terjadi kesalahan saat membuat rekomendasi part: {e}")
            return {}


def main():
    """
    Fungsi utama untuk menjalankan evaluasi
    """
    print("=" * 70)
    print("EVALUASI MODEL SISTEM PENDUKUNG KEPUTUSAN SERVIS MOTOR")
    print("=" * 70)
    
    # Inisialisasi evaluator
    evaluator = EvaluasiModelServisMotor()
    
    # Evaluasi model
    print("\nMemulai evaluasi model...")
    evaluator.evaluasi_model()
    
    # Visualisasi hasil
    print("\nMembuat visualisasi hasil...")
    evaluator.visualisasi_hasil()
    
    # Buat laporan evaluasi
    print("\nMembuat laporan evaluasi...")
    evaluator.buat_laporan_evaluasi()
    
    # Contoh prediksi
    print("\nContoh prediksi untuk kendaraan baru:")
    contoh_data = {
        'Kode Unit Motor': 'NMAX-NEO-S',
        'Tahun Kendaraan': 2021,
        'Kilometer': 15000,
        'Usia_Kendaraan': 2,
        'Kondisi_Oli': 'Perlu Ganti',
        'Kondisi_Rem': 'Baik',
        'KM_Terakhir_Ganti_Oli': 12000,
        'Bulan_Terakhir_Ganti_Oli': 4,
        'KM_Per_Tahun': 7500,
        'Jumlah_Keluhan': 3
    }
    
    hasil_prediksi = evaluator.prediksi_rekomendasi(contoh_data)
    
    if hasil_prediksi:
        print(f"\nJenis servis yang direkomendasikan: {hasil_prediksi['jenis_servis']}")
        print(f"Tingkat kepercayaan: {hasil_prediksi['kepercayaan']:.2%}")
        print("\nRekomendasi part:")
        for part, info in hasil_prediksi['rekomendasi_part'].items():
            print(f"- {part}: {info['status']} (interval {info['interval']} km)")
    
    print("\nEvaluasi selesai. Lihat folder 'model_output/evaluasi' untuk hasil lengkap.")
    print("=" * 70)


if __name__ == "__main__":
    main()
