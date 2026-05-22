"""
train_model.py
──────────────
Bu script mushroom modelini eğitir ve joblib ile kaydeder.
Streamlit app'ini çalıştırmadan önce BİR KEZ çalıştırın:
    python train_model.py
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ── 1. Veriyi yükle ──────────────────────────────────────────────────
print("📂 Veri yükleniyor...")
CSV_PATH = "mushrooms (1).csv"

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV bulunamadı: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)
print(f"   ✅ {df.shape[0]:,} satır × {df.shape[1]} sütun yüklendi.")

# ── 2. Label Encoding ────────────────────────────────────────────────
print("\n🔢 Label Encoding uygulanıyor...")

encoders: dict[str, LabelEncoder] = {}  # Her sütun için ayrı encoder sakla
df_enc = df.copy()

for col in df_enc.columns:
    le = LabelEncoder()
    df_enc[col] = le.fit_transform(df_enc[col])
    encoders[col] = le                  # İleride app'te decode için sakla

print(f"   ✅ {len(encoders)} sütun encode edildi.")

# ── 3. Feature / Target ayrımı ───────────────────────────────────────
X = df_enc.drop(columns=["class"])
y = df_enc["class"]

# ── 4. Train / Test split ────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size    = 0.20,
    random_state = 42,
    stratify     = y
)
print(f"\n✂️  Split: {X_train.shape[0]:,} train / {X_test.shape[0]:,} test")

# ── 5. Model eğitimi ────────────────────────────────────────────────
print("\n🌲 Random Forest eğitiliyor...")
model = RandomForestClassifier(
    n_estimators = 200,    # Daha stabil tahminler için 200 ağaç
    max_depth    = None,
    random_state = 42,
    n_jobs       = -1
)
model.fit(X_train, y_train)

# ── 6. Değerlendirme ─────────────────────────────────────────────────
train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc  = accuracy_score(y_test,  model.predict(X_test))

print(f"\n🎯 SONUÇLAR")
print(f"   Train Accuracy : %{train_acc*100:.4f}")
print(f"   Test  Accuracy : %{test_acc*100:.4f}")
print()
print(classification_report(
    y_test, model.predict(X_test),
    target_names=["Edible (0)", "Poisonous (1)"]
))

# ── 7. Feature importance ─────────────────────────────────────────────
fi = pd.Series(model.feature_importances_, index=X.columns)\
       .sort_values(ascending=False)
print("🏆 Top 5 Feature:")
for feat, score in fi.head(5).items():
    print(f"   {feat:<35} {score*100:.2f}%")

# ── 8. Kaydet ────────────────────────────────────────────────────────
SAVE_PATH = "mushroom_model.pkl"

bundle = {
    "model"         : model,
    "encoders"      : encoders,       # {col: LabelEncoder}
    "feature_names" : list(X.columns),
    "feature_importances": fi.to_dict(),
    "test_accuracy" : test_acc,
}

joblib.dump(bundle, SAVE_PATH, compress=3)
print(f"\n💾 Model kaydedildi → {SAVE_PATH}")
print("✅ Tamamlandı! Şimdi: streamlit run app.py")
