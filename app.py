from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import os

app = Flask(__name__)

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
                'Oli Mesin': 2000
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
                'Oli Mesin': 2000
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

    def cek_kebutuhan_part(self, kilometer, transmisi, km_terakhir_ganti_parts=None):
        """
        Cek kebutuhan penggantian part berdasarkan kilometer dan transmisi
        Juga memperhitungkan kapan terakhir part diganti
        """
        parts_status = {}
        parts = self.parts_intervals[transmisi]
        
        # Default jika tidak ada informasi tentang kapan terakhir kali part diganti
        if km_terakhir_ganti_parts is None:
            km_terakhir_ganti_parts = {}
        
        for part, interval in parts.items():
            # Jika part sudah pernah diganti, perhitungkan km sejak penggantian terakhir
            if part in km_terakhir_ganti_parts and km_terakhir_ganti_parts[part] > 0:
                km_sejak_ganti = kilometer - km_terakhir_ganti_parts[part]
                km_terdekat, km_berikutnya = self.hitung_interval_terdekat(km_sejak_ganti, interval)
                # Konversi ke km absolut
                km_terdekat = km_terakhir_ganti_parts[part] + km_terdekat
                km_berikutnya = km_terakhir_ganti_parts[part] + km_berikutnya
                # Status berdasarkan km sejak penggantian terakhir
                status = 'Perlu Pengecekan' if km_sejak_ganti >= interval else 'Belum Perlu'
            else:
                # Perhitungan normal untuk part yang belum pernah diganti
                km_terdekat, km_berikutnya = self.hitung_interval_terdekat(kilometer, interval)
                # Status hanya 'Perlu Pengecekan' jika kilometer saat ini melebihi atau sama dengan interval
                status = 'Perlu Pengecekan' if kilometer >= interval and kilometer >= km_terdekat else 'Belum Perlu'
            
            parts_status[part] = {
                'interval': interval,
                'km_terdekat': km_terdekat,
                'km_berikutnya': km_berikutnya,
                'status': status
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
                # Handle potential new categories by defaulting to the most common category
                try:
                    input_data[column] = encoder.transform(input_data[column])
                except ValueError:
                    most_common_idx = 0  # Default to first category
                    input_data[column] = most_common_idx
        
        prediksi = self.model.predict(input_data[self.features])
        jenis_servis = self.le_target.inverse_transform(prediksi)[0]
        
        return jenis_servis
    
    def tentukan_kondisi_oli(self, km_terakhir_ganti, kilometer):
        """
        Menentukan kondisi oli berdasarkan jarak tempuh sejak penggantian terakhir
        """
        interval_oli = 2000  # Interval untuk penggantian oli (km)
        jarak_tempuh = kilometer - km_terakhir_ganti
        
        if jarak_tempuh < interval_oli:
            return "Baik"
        elif jarak_tempuh < interval_oli * 1.5:
            return "Perlu Ganti"
        else:
            return "Kritis"
    
    def tentukan_kondisi_rem(self, keluhan_rem):
        """
        Menentukan kondisi rem berdasarkan keluhan
        """
        if keluhan_rem == "tidak_ada":
            return "Baik"
        elif keluhan_rem == "suara":
            return "Perlu Penyetelan"
        else:  # kurang_pakem atau kombinasi
            return "Perlu Penggantian"

# Definisikan fungsi untuk membuat sistem servis
def create_sistem_servis():
    csv_path = os.path.join(os.path.dirname(__file__), 'indoperkasa7.csv')
    print(f"Loading model from data: {csv_path}")
    return SistemPendukungKeputusanServisMotor(csv_path)

# Inisialisasi sistem servis
sistem_servis = create_sistem_servis()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    # Prepare input data
    data_motor = {
        'Kode Unit Motor': data['kode_motor'],
        'Tahun Kendaraan': int(data['tahun']),
        'Kilometer': int(data['kilometer']),
        'Bulan_Terakhir_Ganti_Oli': int(data['bulan_terakhir_ganti_oli']),
        'KM_Terakhir_Ganti_Oli': int(data['km_terakhir_ganti_oli']),
        'Usia_Kendaraan': int(data['usia_kendaraan']),
        'Jumlah_Keluhan': int(data['jumlah_keluhan'])
    }
    
    # Calculate KM_Per_Tahun as requested
    data_motor['KM_Per_Tahun'] = int(data_motor['Kilometer'] / data_motor['Usia_Kendaraan']) if data_motor['Usia_Kendaraan'] > 0 else data_motor['Kilometer']
    
    # Determine oil condition
    data_motor['Kondisi_Oli'] = sistem_servis.tentukan_kondisi_oli(
        data_motor['KM_Terakhir_Ganti_Oli'], 
        data_motor['Kilometer']
    )
    
    # Determine brake condition based on complaints
    data_motor['Kondisi_Rem'] = sistem_servis.tentukan_kondisi_rem(data['keluhan_rem'])
    
    # Get part status based on mileage and transmission
    # Check if user has provided part replacement data
    km_terakhir_ganti_parts = None
    if data.get('has_replaced_parts', False) and 'km_terakhir_ganti_parts' in data:
        km_terakhir_ganti_parts = data['km_terakhir_ganti_parts']
    
    # Use the part replacement data if available
    parts_status = sistem_servis.cek_kebutuhan_part(
        data_motor['Kilometer'], 
        data['transmisi'],
        km_terakhir_ganti_parts
    )
    
    # Make prediction
    jenis_servis = sistem_servis.prediksi_servis(data_motor)
    
    # Add part status information with flags for parts that need replacement
    parts_result = []
    for part, info in parts_status.items():
        parts_result.append({
            'name': part,
            'interval': info['interval'],
            'km_terdekat': info['km_terdekat'],
            'km_berikutnya': info['km_berikutnya'],
            'status': info['status'],
            'needs_replacement': info['status'] == 'Perlu Pengecekan'
        })
    
    result = {
        'jenis_servis': jenis_servis,
        'parts': parts_result,
        'kondisi_oli': data_motor['Kondisi_Oli'],
        'kondisi_rem': data_motor['Kondisi_Rem'],
        'km_per_tahun': data_motor['KM_Per_Tahun']
    }
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)