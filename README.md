# Mantar Sınıflandırıcı

**Mantarın yenilebilir mi yoksa zehirli mi olduğunu 22 kategorik özellikle tahmin eden web uygulaması.**

---

## **İçindekiler**

- **Proje Hakkında**
- **Dataset**
- **Kullanılan Teknolojiler**
- **Kurulum ve Kullanım**
- **Proje Yapısı**
- **Demo**
---

## **Proje Hakkında**

Bu proje, **UCI Mushroom Dataset** kullanılarak mantarların yenilebilir (*edible*) mi yoksa zehirli (*poisonous*) mi olduğunu tahmin eden bir ML'dir.

**Proje kapsamı:**

| Aşama | İçerik |
|-------|--------|
|**EDA** | Keşifsel veri analizi, class distribution, feature dağılımları |
|**Preprocessing** | Label Encoding, Train/Test Split |
|**Modelleme** | Random Forest Classifier (n=200) |
|**Değerlendirme** | Accuracy, Confusion Matrix, Classification Report, Feature Importance |
|**Deployment** | Streamlit interaktif web uygulaması |

---

## **Dataset**

**UCI Mushroom Dataset**
https://www.kaggle.com/datasets/uciml/mushroom-classification

<details>
<summary>22 özelliğin tamamını görmek için tıklayın</summary>

|Özellik|
|--------|
|Şapka Şekli (Cap Shape)|
|Şapka Yüzeyi (Cap Surface)|
|Şapka Rengi (Cap Color)|
|Lekelenme (Bruises)|
|Kokusu (Odor)|
|Solungaç Bağlantısı (Gill Attachment)|
|Solungaç Sıklığı (Gill Spacing)|
| Solungaç Boyutu (Gill Size)|
|Solungaç Rengi (Gill Color)|
|Sap Şekli (Stalk Shape)|
|Sap Kökü (Stalk Root)|
| Halka Üstü Sap Yüzeyi (Stalk Surface Above Ring)|
|Halka Altı Sap Yüzeyi (Stalk Surface Below Ring)|
|Halka Üstü Sap Rengi (Stalk Color Above Ring)|
|Halka Altı Sap Rengi (Stalk Color Below Ring)|
|Peçe Tipi (Veil Type)|
|Peçe Rengi (Veil Color)|
| Halka Sayısı (Ring Number)|
|Halka Tipi (Ring Type)|
|Spor Baskı Rengi (Spore Print Color)|
|Popülasyon Sıklığı (Population)|
|Yetiştiği Alan (Habitat)|

</details>

---

## Kullanılan Teknolojiler
|Python 3.9+|
|----|
|**pandas**|        
|**numpy**|       
|**scikit-learn**|   
|**joblib**|
|**matplotlib**|    
|**seaborn**|      
|**streamlit**|       

---

## Kurulum ve Kullanım

```bash
git clone https://github.com/Nomisia/mushroom-classifier.git
cd "istediğiniz klasör"
```

```bash
pip install -r requirements.txt
```

Gereklilikler yüklendikten sonra modeli eğitmek için aşağıdakini yazabilirsiniz

```bash
python train_model.py
```
Uygulamayı başlatmak için 

```bash
streamlit run app.py
```
---

## Proje Yapısı


MushroomClassification/
- train_model.py                ← Model eğitme 

- app.py                        ← Uygulama

- requirements.txt              ← Python Gereklilikleri

- mushroom_model.pkl  ← Eğitilmiş model (train_model.py çalıştıktan sonra otomatik oluştu)

- mushrooms (1).csv  ← Ham dataset

---
## Demo
https://youtu.be/6X2NsGNC2RA
