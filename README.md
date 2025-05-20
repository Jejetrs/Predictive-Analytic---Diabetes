# Laporan Proyek Machine Learning - Nama Anda

## Domain Proyek

Diabetes merupakan salah satu penyakit kronis yang menjadi penyebab utama kematian di seluruh dunia. Menurut data dari WHO, pada tahun 2021, sekitar 537 juta orang dewasa hidup dengan diabetes, dan jumlah ini diprediksi akan meningkat secara signifikan di tahun-tahun mendatang. Penyakit ini sering tidak terdeteksi pada tahap awal karena gejalanya yang tidak spesifik, yang kemudian dapat berkembang menjadi komplikasi serius seperti gagal ginjal, kebutaan, serangan jantung, dan amputasi anggota tubuh.

Masalah utama dalam penanganan diabetes adalah keterlambatan diagnosis dan penanganan dini. Banyak individu yang tidak menyadari bahwa mereka menderita diabetes sampai terjadi komplikasi serius. Hal ini menjadi hambatan besar dalam upaya pencegahan dan pengelolaan penyakit secara efektif.

Salah satu solusi yang dapat ditawarkan adalah penggunaan teknologi Machine Learning untuk memprediksi kemungkinan seseorang menderita diabetes berdasarkan data kesehatan dan demografi. Dalam proyek ini, dilakukan pengembangan model machine learning untuk memprediksi apakah seseorang berisiko terkena diabetes berdasarkan data medis sederhana.

**Mengapa Masalah Ini Harus Diselesaikan**
- Deteksi dini dapat menyelamatkan nyawa dan menurunkan beban biaya kesehatan.
- Masyarakat membutuhkan alat bantu yang cepat dan dapat diandalkan untuk membantu tenaga medis dalam pengambilan keputusan.
- Data medis yang tersedia sangat potensial untuk diolah secara otomatis melalui algoritma pembelajaran mesin.

Referensi:

- Diabetes Report - WHO : https://www.who.int/news-room/fact-sheets/detail/diabetes
- Global Diabetes Statistics - IDF : https://diabetesatlas.org/

## Business Understanding


Dalam proyek ini, tujuan utamanya adalah membangun model prediktif yang dapat mengidentifikasi individu dengan risiko diabetes berdasarkan data kesehatan mereka.

Bagian laporan ini mencakup:

### Problem Statements

Menjelaskan pernyataan masalah latar belakang:
- Bagaimana cara memprediksi apakah seseorang mengidap diabetes berdasarkan data yang tersedia?
- Algoritma Machine Learning apa yang paling efektif dalam prediksi diabetes berdasarkan akurasi dan performa lainnya?
- Bagaimana hubungan setiap variable dalam menentukan prediksi diabetes?

### Goals
Untuk menjawab permasalahan yang ada, proyek ini bertujuan untuk:
1. Membangun model klasifikasi prediktif yang akurat dan andal dalam mengidentifikasi individu dengan risiko tinggi terkena diabetes berdasarkan data kesehatan.
2. Mengidentifikasi fitur atau atribut paling signifikan yang mempengaruhi hasil prediksi, guna memberikan wawasan tambahan dalam konteks medis dan pencegahan.
3. Meningkatkan performa model melalui pemilihan algoritma yang tepat serta penerapan teknik seperti hyperparameter tuning dan handling class imbalance.
4. Membandingkan kinerja berbagai algoritma klasifikasi, termasuk K-Nearest Neighbors, Random Forest, dan Logistic Regression, untuk menentukan model terbaik.
5. Menghasilkan model dengan performa evaluasi optimal, dilihat dari metrik seperti akurasi, precision, recall, dan F1-score, sehingga hasil prediksi dapat dipercaya untuk mendukung keputusan klinis awal.

**“Solution Statement”**
Untuk mewujudkan tujuan proyek, pendekatan yang digunakan adalah sebagai berikut:

1. Menggunakan tiga algoritma klasifikasi utama, yaitu:
    - Logistic Regression sebagai baseline model yang sederhana dan interpretable.
    - K-Nearest Neighbors (KNN) untuk menangkap pola lokal antar data.
    - Random Forest Classifier sebagai model ensemble untuk menangani non-linearitas dan meningkatkan generalisasi.

2. Menangani masalah ketidakseimbangan data (imbalanced classes) menggunakan teknik SMOTE (Synthetic Minority Over-sampling Technique) untuk menyeimbangkan distribusi antara kelas penderita dan non-penderita diabetes.

3. Melakukan hyperparameter tuning pada model yang memiliki potensi namun performanya belum optimal, dengan tujuan meningkatkan akurasi dan keseimbangan prediksi antar kelas.

4. Menerapkan evaluasi model menggunakan metrik klasifikasi yang relevan (accuracy, precision, recall, dan F1-score) untuk memastikan performa menyeluruh yang adil dan dapat diandalkan.

5. Membandingkan hasil dan memilih model terbaik berdasarkan performa evaluasi, stabilitas, dan interpretabilitas untuk dapat digunakan dalam implementasi praktis di dunia nyata.


## Data Understanding

Sumber : https://www.kaggle.com/datasets/nanditapore/healthcare-diabetes.

Paragraf awal bagian ini menjelaskan informasi mengenai data yang Anda gunakan dalam proyek. Sertakan juga sumber atau tautan untuk mengunduh dataset. Contoh: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Restaurant+%26+consumer+data). 

### Variabel-variabel pada dataset ini adalah sebagai berikut:
| Fitur                     | Deskripsi                                          |
|---------------------------|----------------------------------------------------|
| Id                        | ID unik setiap pasien                              |
| Pregnancies               | Jumlah kehamilan                                   |
| Glucose                   | Kadar glukosa dalam darah                          |
| BloodPressure             | Tekanan darah diastolik (mm Hg)                    |
| SkinThickness             | Ketebalan lipatan kulit triceps (mm)               |
| Insulin                   | Kadar insulin dalam darah (mu U/ml)                |
| BMI                       | Body Mass Index                                    |
| DiabetesPedigreeFunction  | Riwayat keluarga terhadap diabetes                 |
| Age                       | Usia pasien                                        |
| Outcome                   | Target variabel (1 = diabetes, 0 = tidak diabetes) |

### EDA
EDA dilakukan menggunakan visualisasi distribusi, korelasi antar fitur, dan deteksi outlier. Berdasarkan EDA ditemukan :
- Tidak terdapat nilai null secara eksplisit, namun terdapat nilai 0 pada kolom Glucose, BloodPressure, SkinThickness, Insulin, dan BMI yang secara medis tidak mungkin bernilai nol. Ini ditangani sebagai nilai yang perlu diimputasi.
- Distribusi kelas Outcome: Sekitar 34.4% data menunjukkan pasien menderita diabetes, sedangkan sisanya tidak.
- Hubungan Setiap Variabel dalam Menentukan Prediksi Diabetes, Berdasarkan hasil heatmap korelasi, maka bisa disebutkan bahwa:
    - Glucose, BMI, dan Age menunjukkan korelasi positif yang kuat dengan Outcome.
    - Variabel seperti Skin Thickness atau Insulin memiliki korelasi lebih lemah, namun tetap relevan secara klinis.

![Outcome_percentage](https://github.com/user-attachments/assets/fcc510e7-a463-4453-9e45-11196228932b)
![Heatmap](https://github.com/user-attachments/assets/e465efd9-b8be-4837-8bf9-aa2f284ab394)

## Data Preparation
Pada bagian ini dimaksudkan untuk menerapkan dan menyebutkan teknik data preparation yang dilakukan.

**Proses data preparation yang dilakukan**
1. Mengatasi nilai tidak valid (nol):
Kolom seperti Glucose, BloodPressure, SkinThickness, Insulin, dan BMI diperlakukan sebagai fitur yang memiliki nilai tidak valid berupa nol. Nilai tersebut diimputasi menggunakan median dari masing-masing kolom.

2. Scaling (Standardization):
Semua fitur diskalakan menggunakan StandardScaler untuk menormalkan distribusi data, mengingat algoritma seperti Logistic Regression dan KNN sensitif terhadap skala.

3. Train-Test Split:
Dataset dibagi menjadi 80% data pelatihan dan 20% data pengujian.

## Modeling
Pada tahap ini, saya menggunakan tiga algoritma machine learning yang cukup populer untuk kasus klasifikasi, yaitu:
Model yang digunakan dan parameternya :
| Model                | Parameter                          |
|----------------------|-------------------------------------|
| Logistic Regression  | `solver='liblinear'`               |
| Random Forest        | `n_estimators=100`, `max_depth=None` |
| K-Nearest Neighbors  | `n_neighbors=5` (default)          |

Model-model ini dipilih karena masing-masing mewakili pendekatan yang berbeda dalam klasifikasi, mulai dari yang sederhana hingga yang lebih kompleks, sehingga bisa dibandingkan performanya secara adil terhadap dataset diabetes ini.

Tujuan dari penggunaan beberapa model ini adalah untuk membandingkan performa mereka dalam memprediksi apakah seseorang berisiko terkena diabetes berdasarkan dataset yang digunakan.

**Kelebihan dan kekurangan dari algoritma yang digunakan**

1. K-Nearest Neighbors (KNN)
    KNN adalah model yang sangat intuitif dan mudah dipahami. Cara kerjanya yaitu mencari sejumlah "tetangga terdekat" dari sebuah titik data baru, lalu memprediksi kelasnya berdasarkan mayoritas label dari tetangga tersebut. Jarak yang digunakan untuk mengukur kedekatan biasanya adalah Euclidean distance.
    
    Pada model ini, KNN dikonfigurasi dengan 5 tetangga terdekat dan menggunakan bobot berdasarkan jarak (weights='distance'), sehingga tetangga yang lebih dekat memiliki pengaruh lebih besar.

    ```python
    knn_model = KNeighborsClassifier(n_neighbors=5, weights='distance')
    knn_model.fit(X_train_resampled, y_train_resampled)
    y_pred_knn = knn_model.predict(X_test_scaled)
    ```

    nilai K = 5 adalah nilai default dari KNeighborsClassifier dan sering jadi pilihan awal karena cukup kecil untuk menangkap pola lokal,tidak terlalu kecil untuk menghindari overfitting (misal K=1 terlalu sensitif) dan tidak terlalu besar yang bisa menyebabkan underfitting.
    
    Kelebihan:
    - Sederhana dan tidak butuh banyak asumsi.
    - Cocok saat data tidak terlalu besar.

    Kekurangan:
    - Performanya bisa turun jika datanya banyak (scalability).
    - Sensitif terhadap skala fitur dan noise, perlu dilakukan normalisasi.

2. Random Forest
    Random Forest adalah model berbasis ensembling yang menggunakan banyak pohon keputusan (decision tree). Ibaratnya, daripada menebak keputusan berdasarkan satu "pendapat", model ini mengumpulkan banyak "pendapat dari pohon-pohon" dan mengambil suara terbanyak.

    Setiap decision tree dibentuk dari subset data dan fitur yang berbeda-beda (ini disebut bagging), jadi model ini lebih kuat dalam menangani overfitting dibanding satu decision tree saja.
    
    ```python
    rf_model = RandomForestClassifier(n_estimators=100, random_state=101)
    rf_model.fit(X_train_resampled, y_train_resampled)
    y_pred_rf = rf_model.predict(X_test_scaled)
    ```

    Pada model ini, agar hasil eksperimen bisa reproducible (bisa diulang dengan hasil yang sama), kita beri angka tetap untuk random_state. Angka 101 sebenarnya arbitrary (bebas). Bisa pakai angka apa saja (misalnya: 0, 42, 123). Tujuannya hanya untuk konsistensi hasil saat kita melatih ulang model, menulis laporan, dan membandingkan model.

    Kelebihan:
    - Sangat kuat untuk data yang kompleks.
    - Tidak terlalu sensitif terhadap data outlier atau noise.
    - Bisa menunjukkan pentingnya fitur-fitur (feature importance).

    Kekurangan:
    - Kurang interpretatif (agak sulit dijelaskan ke non-teknis).
    - Memerlukan sumber daya lebih besar untuk pelatihan.
    - Waktu training bisa lebih lama dibanding model sederhana.
    - Tidak cocok kalau data banyak noise.

3. Logistic Regression
    Logistic Regression adalah model dasar yang sering digunakan dalam klasifikasi biner. Walaupun namanya "regression", model ini bukan untuk memprediksi angka, tapi Logistic Regression adalah model klasifikasi dasar yang menghitung probabilitas yang termasuk dalam suatu kelas (misalnya, terkena diabetes atau tidak). Model ini cukup interpretatif dan sering digunakan sebagai baseline dalam banyak proyek klasifikasi.

    Cara kerjanya adalah menghitung peluang (probabilitas) seseorang terkena diabetes berdasarkan nilai fitur-fitur seperti kadar glukosa, tekanan darah, dll. Jika probabilitasnya di atas 0.5, maka model akan memprediksi bahwa orang tersebut terkena diabetes (label = 1).

    ```python
    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train_resampled, y_train_resampled)
    y_pred_log = log_model.predict(X_test_scaled)
    ```

    Model Logistic Regression ini menggunakan proses optimasi iteratif (seperti Gradient Descent) untuk menemukan koefisien terbaik. Secara default, max_iter = 100 (tergantung solver). Namun, jika data cukup kompleks, model bisa belum konvergen (belum menemukan solusi optimal) dalam 100 iterasi.Oleh karena itu, max_iter=1000 dipilih agar memberi cukup ruang untuk algoritma menyelesaikan training dan menghindari peringatan ConvergenceWarning.

    Kelebihan:
    - Sederhana dan mudah diinterpretasikan.
    - Cocok sebagai baseline model.
    - Tidak mudah overfitting jika datanya bersih.

    Kekurangan:
    - Tidak cocok untuk data dengan hubungan yang sangat kompleks dan tidak linier.

    Namun, pada awalnya performa Logistic Regression jauh lebih rendah dibandingkan dua model lainnya. Nilai akurasinya hanya 76.89%, jauh di bawah KNN dan Random Forest yang mencapai hampir 99%.

    Untuk meningkatkan performa, dilakukan hyperparameter tuning menggunakan teknik Grid Search. Fokus tuning diarahkan pada pemilihan penalti dan solver. Setelah eksplorasi parameter, diperoleh kombinasi terbaik:

    Best Parameters: {'l1_ratio': 0.1, 'penalty': 'elasticnet', 'solver': 'saga'}

    Namun, meskipun sudah dilakukan tuning, peningkatan akurasi masih terbatas, hanya mencapai 76.71%, menandakan bahwa model Logistic Regression kurang cocok untuk dataset ini dibanding model lainnya.

Berdasarkan evaluasi terhadap ketiga model yang telah dilatih — yaitu K-Nearest Neighbors (KNN), Random Forest, dan Logistic Regression — model yang dipilih sebagai solusi terbaik untuk kasus prediksi diabetes ini adalah:

**✅ K-Nearest Neighbors (KNN)**

Alasan Pemilihan KNN sebagai Model Terbaik:

- Akurasi Tertinggi
KNN berhasil mencapai akurasi sebesar 99.28%, tertinggi di antara semua model yang diuji.
Artinya, model ini mampu memprediksi dengan benar hampir seluruh data uji, menunjukkan kinerjanya sangat baik dalam memahami pola pada data.

- Performa Konsisten Tanpa Overfitting
Meskipun akurasinya sangat tinggi, KNN tetap menunjukkan stabilitas dan tidak overfitting karena menggunakan bobot berdasarkan jarak, membuat prediksinya lebih “bijak” terhadap tetangga terdekat yang paling relevan.

- Tidak Memerlukan Asumsi Khusus
KNN tidak mengharuskan asumsi seperti linearitas hubungan antar fitur (berbeda dengan Logistic Regression). Ini sangat membantu jika data bersifat non-linear atau memiliki distribusi yang tidak biasa.

- Kemudahan Implementasi dan Interpretasi
Secara konsep, KNN sangat mudah dipahami oleh praktisi maupun pihak non-teknis: prediksi dilakukan dengan melihat mayoritas tetangga terdekat. Ini penting jika model akan digunakan dalam sistem riil dengan pengguna umum seperti tenaga medis.

- Mengalahkan Model Lain Secara Konsisten
Dibandingkan Random Forest (99.10%) dan Logistic Regression (76.89% sebelum tuning), KNN tetap unggul, bahkan tanpa tuning yang kompleks.

## Evaluation

Pada tahap ini, evaluasi model dilakukan untuk memahami seberapa baik kinerja model klasifikasi dalam memprediksi data baru. Evaluasi model dilakukan dengan menggunakan metrik-metrik evaluasi klasifikasi berikut:

- **Accuracy**: Persentase prediksi yang benar terhadap total prediksi.  
  Formula:  
  ![Accuracy](https://github.com/user-attachments/assets/4f0bdd4a-12db-4cde-862c-65d5cccf8ea9)

- **Precision**: Proporsi positif yang diprediksi benar dari seluruh prediksi positif.  
  Formula:  
  ![Precision](https://github.com/user-attachments/assets/4d1e0bf6-cf26-4286-a4cb-b1bf476ba0e5)

- **Recall**: Proporsi positif yang diprediksi benar dari seluruh kasus aktual positif.  
  Formula:  
  ![Recall](https://github.com/user-attachments/assets/8bfb5177-b2b0-41da-8e0d-c42c773dfa04)

- **F1-Score**: Harmonik rata-rata dari precision dan recall.  
  Formula:  
  ![F1-Score](https://github.com/user-attachments/assets/7a8a4a0e-64de-464e-a8ea-9981bb315b58)

**Hasil Evaluasi Model**
Berikut adalah hasil evaluasi terhadap ketiga model yang digunakan:

| Model               | Accuracy | Precision | Recall | F1 Score |
|---------------------|----------|-----------|--------|----------|
| KNN                 | 0.993    | 0.990     | 1.000  | 0.995    |
| Random Forest       | 0.991    | 0.980     | 1.000  | 0.990    |
| Logistic Regression | 0.769    | 0.648     | 0.641  | 0.644    |

Catatan: Evaluasi dilakukan terhadap data uji (X_test) setelah model dilatih menggunakan data hasil oversampling SMOTE dan data training yang telah diskalakan.

**Interpretasi dan Insight**
- KNN menjadi model terbaik karena mencetak skor tertinggi secara konsisten di seluruh metrik, terutama recall yang mencapai 1.00. Artinya, model tidak melewatkan satu pun pasien yang benar-benar mengidap diabetes dalam prediksinya (False Negative = 0). Hal ini sangat penting dalam konteks medis, karena salah satu tujuan utama adalah mendeteksi semua kasus positif secara tepat.
- Random Forest juga cukup baik, namun sedikit kalah di aspek precision, yang berarti masih terdapat sejumlah kecil false positive.
- Logistic Regression memberikan performa paling rendah dalam semua metrik, meskipun telah dilakukan hyperparameter tuning. Ini menunjukkan model ini kurang cocok untuk menangkap pola non-linear atau kompleksitas data dalam kasus ini.

## Kesimpulan

- Proyek ini berhasil membangun model prediksi diabetes dengan menggunakan data medis sederhana.
- Dari tiga model yang diuji (KNN, Random Forest, Logistic Regression), KNN memberikan performa terbaik dengan akurasi 99.3% dan recall sempurna 1.0, menjadikannya pilihan utama untuk digunakan dalam prediksi risiko diabetes.
- Penanganan nilai nol yang tidak valid dan imbalance data menggunakan median imputation dan SMOTE sangat berkontribusi dalam meningkatkan performa model.
- Hyperparameter tuning pada Logistic Regression memberikan sedikit peningkatan, tetapi model ini tetap kurang cocok dibanding dua model lainnya untuk dataset ini.

## Deployment

Aplikasi ini telah dikembangkan menggunakan Streamlit untuk visualisasi data dan prediksi risiko diabetes secara interaktif.
Link : 
