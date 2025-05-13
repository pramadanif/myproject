import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

class SistemPendukungKeputusanServisMotor:
    def __init__(self, file_data):
        # Load Data
        self.df = pd.read_csv(file_data)
        
        # Fitur yang akan digunakan
        self.features = [
            'Kode Unit Motor', 'Tahun Kendaraan', 'Kilometer', 
            'Kondisi_Oli', 'Kondisi_Rem', 'Jumlah_Keluhan',
            'Bulan_Terakhir_Ganti_Oli', 'KM_Terakhir_Ganti_Oli',
            'Usia_Kendaraan', 'KM_Per_Tahun'
        ]
        
        # Definisi part dan interval kilometer berdasarkan transmisi
        self.parts_intervals = {
            'matic': {
                'Kampas Rem': 15000,
                'Kampas Kopling CVT': 20000,
                'V-Belt': 25000,
                'Filter Udara': 10000,
                'Busi': 8000,
                'Roller CVT': 20000,
                'Cover CVT': 30000,
                'Oil CVT': 15000,
                'Bearing CVT': 40000,
                'Seal CVT': 35000,
                'Filter Bensin': 20000,
                'Saringan Udara': 12000,
                'Oli Mesin':2000
            },
            'manual': {
                'Kampas Rem': 15000,
                'Rantai': 20000,
                'Gir Depan': 25000,
                'Gir Belakang': 25000,
                'Filter Udara': 10000,
                'Busi': 8000,
                'Kampas Kopling': 30000,
                'Kabel Kopling': 25000,
                'Filter Bensin': 20000,
                'Saringan Udara': 12000,
                'Seal Front Fork': 35000,
                'Bearing Roda': 40000,
                'Oli Mesin':2000
            }
        }
        
        # Preprocessing
        self.X = self.df[self.features].copy()
        self.y = self.df['Jns Service']
        
        # Encode categorical variables
        self.le_dict = {}
        categorical_columns = ['Kode Unit Motor', 'Kondisi_Oli', 'Kondisi_Rem']
        for column in categorical_columns:
            self.le_dict[column] = LabelEncoder()
            self.X[column] = self.le_dict[column].fit_transform(self.X[column])
        
        # Encode target variable
        self.le_target = LabelEncoder()
        self.y = self.le_target.fit_transform(self.y)
        
        # Latih model
        self.model = self.latih_model()

    def hitung_interval_terdekat(self, kilometer, interval):
        """
        Menghitung interval terdekat untuk penggantian part
        """
        jumlah_interval = kilometer // interval
        km_interval_terdekat = jumlah_interval * interval
        km_interval_berikutnya = (jumlah_interval + 1) * interval
        return km_interval_terdekat, km_interval_berikutnya

    def cek_kebutuhan_part(self, kilometer, transmisi):
        """
        Cek kebutuhan penggantian part berdasarkan kilometer dan transmisi
        """
        parts_status = {}
        parts = self.parts_intervals[transmisi]
        
        for part, interval in parts.items():
            km_terdekat, km_berikutnya = self.hitung_interval_terdekat(kilometer, interval)
            parts_status[part] = {
                'interval': interval,
                'km_terdekat': km_terdekat,
                'km_berikutnya': km_berikutnya,
                'status': 'Perlu Pengecekan' if kilometer >= km_terdekat else 'Belum Perlu'
            }
        
        return parts_status

    def latih_model(self):
        """
        Latih model Decision Tree
        """
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        dt = DecisionTreeClassifier(
            criterion='gini', 
            random_state=42,
            max_depth=7,
            min_samples_split=10
        )
        
        dt.fit(X_train, y_train)
        
        y_pred = dt.predict(X_test)
        akurasi = accuracy_score(y_test, y_pred)
        print(f"Akurasi Model: {akurasi:.2%}")
        
        return dt

    def prediksi_servis(self, data_motor):
        """
        Prediksi jenis servis untuk motor
        """
        input_data = pd.DataFrame([data_motor])
        
        for column, encoder in self.le_dict.items():
            if column in input_data.columns:
                input_data[column] = encoder.transform(input_data[column])
        
        prediksi = self.model.predict(input_data[self.features])
        jenis_servis = self.le_target.inverse_transform(prediksi)[0]
        
        return jenis_servis

def input_data_motor(spk_instance):
    """
    Fungsi untuk input data motor dari terminal
    """
    print("=== Sistem Pendukung Keputusan Servis Motor ===")
    data_motor = {}
    
    print("\n📋 Masukkan Detail Motor:")
    data_motor['Kode Unit Motor'] = input("Kode Unit Motor: ")
    
    # Input jenis transmisi
    while True:
        transmisi = input("Jenis Transmisi (matic/manual): ").lower()
        if transmisi in ['matic', 'manual']:
            break
        print("Mohon masukkan 'matic' atau 'manual'")
    
    data_motor['Tahun Kendaraan'] = int(input("Tahun Kendaraan: "))
    data_motor['Kilometer'] = int(input("Kilometer Tempuh: "))
    data_motor['Kondisi_Oli'] = input("Kondisi Oli (Baik/Perlu Ganti/Kritis): ")
    data_motor['Kondisi_Rem'] = input("Kondisi Rem (Baik/Perlu Penyetelan/Perlu Penggantian): ")
    data_motor['Bulan_Terakhir_Ganti_Oli'] = int(input("Bulan Terakhir Ganti Oli: "))
    data_motor['KM_Terakhir_Ganti_Oli'] = int(input("KM Terakhir Ganti Oli: "))
    data_motor['Usia_Kendaraan'] = int(input("Usia Kendaraan (Tahun): "))
    data_motor['KM_Per_Tahun'] = int(input("Kilometer Per Tahun: "))
    
    print("\n📝 Masukkan Informasi Keluhan:")
    data_motor['Jumlah_Keluhan'] = int(input("Jumlah keluhan yang dirasakan: "))
    
    # Cek part berdasarkan kilometer dan transmisi
    parts_status = spk_instance.cek_kebutuhan_part(data_motor['Kilometer'], transmisi)
    
    if parts_status:
        print("\n🔧 Pengecekan Part Berdasarkan Kilometer:")
        for part, info in parts_status.items():
            if info['status'] == 'Perlu Pengecekan':
                while True:
                    print(f"\nPart: {part}")
                    print(f"Kilometer saat ini: {data_motor['Kilometer']} km")
                    print(f"Interval penggantian: {info['interval']} km")
                    print(f"KM terdekat: {info['km_terdekat']} km")
                    print(f"KM berikutnya: {info['km_berikutnya']} km")
                    
                    sudah_ganti = input(f"Apakah sudah mengganti {part}? (y/n): ").lower()
                    if sudah_ganti in ['y', 'n']:
                        data_motor[f'Status_{part.replace(" ", "_")}'] = sudah_ganti
                        break
                    print("Mohon masukkan 'y' atau 'n'")
    
    return data_motor, parts_status, transmisi

def main():
    try:
        sistem_servis = SistemPendukungKeputusanServisMotor('indoperkasa2.csv')
        
        while True:
            # Input data motor
            data_motor, parts_status, transmisi = input_data_motor(sistem_servis)
            
            # Prediksi jenis servis
            jenis_servis = sistem_servis.prediksi_servis(data_motor)
            
            # Tampilkan hasil
            print("\n📋 Hasil Analisis:")
            print(f"Jenis Motor: {transmisi.upper()}")
            print(f"Jenis Servis yang Direkomendasikan: {jenis_servis}")
            
            # Tampilkan rekomendasi part
            print("\n🔧 Status Part berdasarkan Kilometer:")
            for part, info in parts_status.items():
                if info['status'] == 'Perlu Pengecekan':
                    status = data_motor.get(f'Status_{part.replace(" ", "_")}')
                    if status == 'n':
                        print(f"❗ {part}:")
                        print(f"   - Interval: {info['interval']} km")
                        print(f"   - KM Terdekat: {info['km_terdekat']} km")
                        print(f"   - KM Berikutnya: {info['km_berikutnya']} km")
                        print(f"   - Status: PERLU DIGANTI")
                    else:
                        print(f"✅ {part}: Sudah diganti")
                else:
                    print(f"🟢 {part}: {info['status']}")
            
            # Opsi lanjut
            lanjut = input("\nApakah ingin mengecek motor lain? (y/n): ").lower()
            if lanjut != 'y':
                break
        
        print("\nTerima kasih telah menggunakan sistem!")
    
    except FileNotFoundError:
        print("Error: File data motor tidak ditemukan!")
    except Exception as e:
        print(f"Terjadi kesalahan: {e}")

if __name__ == "__main__":
    main()