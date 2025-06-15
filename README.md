# 🩺 Prediksi Risiko Diabetes dengan Machine Learning
Proyek ini bertujuan membangun model prediktif untuk mendeteksi risiko diabetes berdasarkan data medis sederhana. Menggunakan dataset dari Kaggle, dilakukan eksplorasi data, preprocessing, penyeimbangan kelas (SMOTE), dan modeling dengan tiga algoritma: K-Nearest Neighbors (KNN), Random Forest, dan Logistic Regression.
---
🔍 Model Terbaik:
<br>
✅ KNN dipilih karena memberikan akurasi tertinggi (99.45%) dan recall terbaik (99.47%), sangat ideal untuk deteksi dini.
---
🔧 Fitur Proyek:

- Preprocessing cermat: imputasi, normalisasi, outlier handling.
- SMOTE untuk menangani data imbalance.
- Hyperparameter tuning & evaluasi metrik: Accuracy, Precision, Recall, F1-score.
- Aplikasi deploy menggunakan Streamlit.
---
🌐 Link Aplikasi: https://predictive-analytic-diabetes.streamlit.app/
---
📂 Struktur Folder :
```
app/
├── app.py
├── requirements.txt
├── cleaned_dataset/Cleaned_Healthcare_Diabetes.csv
├── model/{rf_model.pkl, knn_model.pkl, lr_model.pkl, scaler.pkl}
```
---
📌 Cara Menjalankan:
python -m venv env
.\env\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
