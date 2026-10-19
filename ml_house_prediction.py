import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# ==========================================
# 1. DATA COLLECTION (Simulasi Data Rumah)
# ==========================================
# Fitur: [Luas Bangunan (m2), Jumlah Kamar Tidur]
# Target: Harga Jual (Ratusan Juta Rupiah)
np.random.seed(42)
num_samples = 500
luas_bangunan = np.random.randint(30, 300, num_samples)
kamar_tidur = np.random.randint(1, 6, num_samples)

# Mensimulasikan harga: Rp. 5 Juta/m2 + Rp. 50 Juta/kamar + Noise
harga_jual = (luas_bangunan * 5) + (kamar_tidur * 50) + np.random.normal(0, 30, num_samples)

# Memasukkan ke DataFrame (Data Mentah)
df = pd.DataFrame({
    'Luas_m2': luas_bangunan,
    'Jumlah_Kamar': kamar_tidur,
    'Harga_Juta_Rp': harga_jual
})

print("Contoh 5 Data Rumah Pertama:")
print(df.head())

# ==========================================
# 2. SPLITTING DATA (Training & Testing)
# ==========================================
# Variabel X (Features/Atribut) dan Y (Target/Label)
X = df[['Luas_m2', 'Jumlah_Kamar']]
y = df['Harga_Juta_Rp']

# Membagi data: 80% untuk Mesin Belajar (Train), 20% untuk Ujian (Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nJumlah Data Latih: {len(X_train)} | Jumlah Data Uji: {len(X_test)}")

# ==========================================
# 3. TRAINING MODEL (Supervised Learning - Linear Regression)
# ==========================================
model = LinearRegression()
model.fit(X_train, y_train)

# ==========================================
# 4. EVALUATION & PREDICTION
# ==========================================
# Menguji kepintaran mesin menebak harga pada data yang belum pernah dilihat (X_test)
y_pred = model.predict(X_test)

# Menghitung Metrik Performa (Seberapa akurat prediksi mesin)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n=== HASIL EVALUASI MODEL ===")
print(f"R-squared (Tingkat Kecocokan) : {r2 * 100:.2f}% (Mendekati 100% sangat baik)")
print(f"Rata-rata Meleset (MAE)       : Rp. {mae:.2f} Juta")

# Mensimulasikan Input Manual dari Pengguna Baru
print("\n--- TEST PREDIKSI RUMAH BARU ---")
# Misalnya: Rumah ukuran 120m2 dengan 3 kamar tidur
input_baru = np.array([[120, 3]])
prediksi_baru = model.predict(input_baru)
print(f"Spesifikasi : 120m2, 3 Kamar Tidur")
print(f"Prediksi Harga AI : Rp. {prediksi_baru[0]:.2f} Juta")