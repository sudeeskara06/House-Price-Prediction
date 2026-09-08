import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Örnek Ev Fiyatı Veri Seti (California Housing Veri Seti)
url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
df = pd.read_csv(url)

# 2. Özellikler (X) ve Hedef Değişken (y - Ev Fiyatı)
# rm: Oda sayısı, lstat: Düşük gelirli nüfus oranı, crim: Suç oranı, age: Bina yaşı
X = df[['rm', 'lstat', 'crim', 'age', 'tax']]
y = df['medv']  # medv: Evin ortalama fiyatı ($1000 cinsinden)

# 3. Eğitim ve Test Setine Ayırma (%80 Eğitim, %20 Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Model Kurma (Random Forest Regressor)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Tahmin Yapma ve Başarıyı Ölçme
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("--- Ev Fiyatı Tahmin Modeli Sonuçları ---")
print(f"Ortalama Hata Miktarı (MAE): ${mae * 1000:.2f}")
print(f"Model Başarı Skoru (R² Score): %{r2 * 100:.2f}")

# 6. Örnek Bir Ev İçin Fiyat Tahmini Yapma
# [Oda Sayısı (rm), Düşük Gelir Oranı (lstat), Suç Oranı (crim), Yaş (age), Vergi (tax)]
ornek_ev = [[6.5, 9.5, 0.1, 45, 300]]
tahmini_fiyat = model.predict(ornek_ev)
print(f"\nÖrnek Evin Tahmini Fiyatı: ${tahmini_fiyat[0] * 1000:.2f}")