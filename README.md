# Laporan Proyek Machine Learning - Jessica Theresia
## Domain Proyek

Diabetes merupakan salah satu penyakit kronis paling serius yang menjadi penyebab utama kematian di seluruh dunia. Berdasarkan laporan International Diabetes Federation (IDF) Diabetes Atlas, pada tahun 2024 terdapat 589 juta orang dewasa (usia 20–79 tahun) yang hidup dengan diabetes, atau setara dengan 1 dari 9 orang dewasa di dunia. Jumlah ini diprediksi meningkat menjadi 853 juta pada tahun 2050. Diabetes juga menjadi penyebab 3,4 juta kematian di tahun 2024, atau sekitar 1 kematian setiap 9 detik. Selain itu, total pengeluaran untuk penanganan diabetes mencapai USD 1 triliun, meningkat 338% dibandingkan 17 tahun sebelumnya.

Yang lebih mengkhawatirkan, banyak kasus diabetes yang tidak terdiagnosis pada tahap awal. Gejala yang tidak spesifik membuat penderita sering kali tidak menyadari keberadaan penyakit ini hingga muncul komplikasi berat seperti gagal ginjal, kebutaan, penyakit jantung, bahkan amputasi. WHO mencatat bahwa 47% kematian akibat diabetes terjadi sebelum usia 70 tahun, dan lebih dari 530.000 kematian akibat penyakit ginjal juga disebabkan oleh diabetes.

Masalah utama dalam penanganan diabetes adalah keterlambatan diagnosis dan rendahnya cakupan pengobatan, terutama di negara berpenghasilan rendah dan menengah. Pada tahun 2022, 59% penderita diabetes usia di atas 30 tahun tidak menjalani pengobatan, menunjukkan adanya celah besar dalam sistem deteksi dan penanganan dini.

Salah satu pendekatan inovatif untuk menjawab tantangan ini adalah penggunaan algoritma Machine Learning (ML) untuk memprediksi risiko diabetes berdasarkan data kesehatan dan demografi. Dalam proyek ini, dikembangkan sebuah model prediktif yang mampu mengklasifikasikan apakah seseorang memiliki risiko menderita diabetes, menggunakan data medis yang relatif sederhana namun informatif. Model ini diharapkan dapat menjadi alat bantu yang cepat, akurat, dan hemat biaya untuk mendukung skrining awal.

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
5. Menghasilkan model dengan performa evaluasi optimal, dilihat dari metrik seperti akurasi, precision, recall, dan F1-score, sehingga hasil prediksi dapat dipercaya untuk mendukung keputusan awal.

#### **“Solution Statement”**
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

Dataset yang digunakan adalah Healthcare Diabetes Dataset dari Kaggle yang dapat diunduh di sini https://www.kaggle.com/datasets/nanditapore/healthcare-diabetes. Dataset erdiri dari 2768 baris dan 10 kolom fitur.

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

### EDA dan Visualisasi Data
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
1. Menghapus Kolom yang Tidak Relevan<br>
Kolom Id dihapus karena tidak berkontribusi dalam proses prediksi.

2. Mengatasi Nilai Tidak Valid (0)<br>
Kolom medis seperti Glucose, BloodPressure, SkinThickness, Insulin, dan BMI tidak seharusnya memiliki nilai nol. Nilai nol tersebut diganti dengan NaN, lalu diimputasi menggunakan median masing-masing kolom.

3. Handling Outliers<br>
Digunakan metode Winsorizing berbasis IQR untuk mengatasi nilai outlier. Nilai di bawah Q1 - 1.5 * IQR dan di atas Q3 + 1.5 * IQR dibatasi (clipped) ke batas bawah dan atas.

4. Split Data (Train-Test Split)<br>
Dataset dibagi menjadi 80% data pelatihan dan 20% data pengujian.

5. Normalisasi Data<br>
Menggunakan StandardScaler untuk menormalkan fitur karena model seperti KNN dan Logistic Regression sensitif terhadap skala fitur.
   
6. Handling Data Imbalance<br>
Menggunakan SMOTE untuk oversampling kelas minoritas sehingga distribusi kelas menjadi seimbang (1:1).

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

1. K-Nearest Neighbors (KNN)<br>
    KNN adalah model yang sangat intuitif dan mudah dipahami. Cara kerjanya yaitu mencari sejumlah "tetangga terdekat" dari sebuah titik data baru, lalu memprediksi kelasnya berdasarkan mayoritas label dari tetangga tersebut. Jarak yang digunakan untuk mengukur kedekatan biasanya adalah Euclidean distance.
    
    Pada model ini, KNN dikonfigurasi dengan 15 tetangga terdekat dan menggunakan bobot berdasarkan jarak (weights='distance'), sehingga tetangga yang lebih dekat memiliki pengaruh lebih besar.

    ```python
    knn_model = KNeighborsClassifier(n_neighbors=15, weights='distance')
    knn_model.fit(X_train_resampled, y_train_resampled)
    y_pred_knn = knn_model.predict(X_test_scaled)
    ```

    Pemilihan nilai k=15 bertujuan untuk menstabilkan prediksi. Nilai k yang terlalu kecil (misalnya 3 atau 5) dapat membuat model terlalu sensitif terhadap noise (overfitting), sementara nilai yang lebih besar seperti 15 memungkinkan model menangkap pola umum dengan lebih baik.
    
    Kelebihan:
    - Sederhana dan tidak butuh banyak asumsi.
    - Cocok untuk dataset berukuran kecil hingga sedang..

    Kekurangan:
    - Performanya bisa turun jika datanya banyak (scalability).
    - Sensitif terhadap skala fitur dan noise, perlu dilakukan normalisasi.

2. Random Forest<br>
    Random Forest adalah model berbasis ensembling yang menggunakan banyak pohon keputusan (decision tree). Ibaratnya, daripada menebak keputusan berdasarkan satu "pendapat", model ini mengumpulkan banyak "pendapat dari pohon-pohon" dan mengambil suara terbanyak.

    Setiap decision tree dibentuk dari subset data dan fitur yang berbeda-beda (ini disebut bagging), jadi model ini lebih kuat dalam menangani overfitting dibanding satu decision tree saja.
    
    ```python
    rf_model = RandomForestClassifier(n_estimators=100, random_state=101)
    rf_model.fit(X_train_resampled, y_train_resampled)
    y_pred_rf = rf_model.predict(X_test_scaled)
    ```

    Penggunaan random_state=101 bertujuan agar eksperimen bersifat reproducible (hasil konsisten saat dijalankan ulang), meskipun angkanya bebas (arbitrary).

    Kelebihan:
    - Sangat kuat untuk data yang kompleks dan interaksi non-linier.
    - Tidak terlalu sensitif terhadap data outlier atau noise.
    - Mampu mengukur pentingnya fitur (feature importance).

    Kekurangan:
    - Kurang interpretatif (agak sulit dijelaskan ke non-teknis).
    - Memerlukan sumber daya lebih besar untuk pelatihan.
    - Waktu training bisa lebih lama dibanding model sederhana.
    - Performa bisa menurun jika data terlalu noisy.

3. Logistic Regression<br>
    Logistic Regression adalah model dasar yang sering digunakan dalam klasifikasi biner. Walaupun namanya "regression", model ini bukan untuk memprediksi angka, tapi Logistic Regression adalah model klasifikasi dasar yang menghitung probabilitas yang termasuk dalam suatu kelas (misalnya, terkena diabetes atau tidak). Model ini cukup interpretatif dan sering digunakan sebagai baseline dalam banyak proyek klasifikasi.

    Cara kerjanya adalah menghitung peluang (probabilitas) seseorang terkena diabetes berdasarkan nilai fitur-fitur seperti kadar glukosa, tekanan darah, dll. Jika probabilitasnya di atas 0.5, maka model akan memprediksi bahwa orang tersebut terkena diabetes (label = 1).

    ```python
    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train_resampled, y_train_resampled)
    y_pred_log = log_model.predict(X_test_scaled)
    ```

    Parameter max_iter=1000 digunakan untuk memastikan proses optimasi konvergen, terutama jika dataset kompleks. Nilai default (100) sering kali tidak cukup dan menghasilkan peringatan ConvergenceWarning.

    Kelebihan:
    - Sederhana dan mudah diinterpretasikan.
    - Cocok sebagai baseline model.
    - Tidak mudah overfitting jika datanya bersih.

    Kekurangan:
    - Tidak cocok untuk data dengan hubungan yang sangat kompleks dan tidak linier.

    Namun, pada awalnya performa Logistic Regression jauh lebih rendah dibandingkan dua model lainnya. Nilai akurasinya hanya 76.89%, jauh di bawah KNN dan Random Forest yang mencapai hampir 99%.

    Untuk meningkatkan performa, dilakukan hyperparameter tuning menggunakan teknik Grid Search. Fokus tuning diarahkan pada pemilihan penalti dan solver. Setelah eksplorasi parameter, diperoleh kombinasi terbaik:
   
    ```python
    Best Parameters: {'l1_ratio': 0.1, 'penalty': 'elasticnet', 'solver': 'saga'}
    ```
    
    Namun, meskipun sudah dilakukan tuning, peningkatan akurasi masih terbatas, hanya mencapai 76.71%, menandakan bahwa model Logistic Regression kurang cocok untuk dataset ini dibanding model lainnya.
   

Berdasarkan evaluasi terhadap ketiga model yang telah dilatih yaitu K-Nearest Neighbors (KNN), Random Forest, dan Logistic Regression — model yang dipilih sebagai solusi terbaik untuk kasus prediksi diabetes ini adalah:

**✅ K-Nearest Neighbors (KNN)**

Alasan Pemilihan KNN sebagai Model Terbaik:

- Akurasi Tinggi<br>
KNN berhasil mencapai akurasi sebesar 98.91%. Artinya, model ini mampu memprediksi dengan benar hampir seluruh data uji, menunjukkan kinerjanya sangat baik dalam memahami pola pada data.

- Performa Konsisten Tanpa Overfitting<br>
Meskipun akurasinya sangat tinggi, KNN tetap menunjukkan stabilitas dan tidak overfitting karena menggunakan bobot berdasarkan jarak, membuat prediksinya lebih “bijak” terhadap tetangga terdekat yang paling relevan.

- Memprioritaskan recall<br>
Recall KNN 0.9948 artinya, hampir semua kasus positif (fraud/penyakit) berhasil dideteksi dengan benar. Ini sangat penting untuk skenario seperti deteksi penyakit, di mana tidak mendeteksi pasien yang sebenarnya sakit dapat berbahaya.
  
- Tidak Memerlukan Asumsi Khusus<br>
KNN tidak mengharuskan asumsi seperti linearitas hubungan antar fitur (berbeda dengan Logistic Regression). Ini sangat membantu jika data bersifat non-linear atau memiliki distribusi yang tidak biasa.

- Kemudahan Implementasi dan Interpretasi<br>
Secara konsep, KNN sangat mudah dipahami oleh praktisi maupun pihak non-teknis: prediksi dilakukan dengan melihat mayoritas tetangga terdekat. Ini penting jika model akan digunakan dalam sistem real dengan pengguna umum seperti tenaga medis.

## Evaluation

Pada tahap ini, evaluasi model dilakukan untuk memahami seberapa baik kinerja model klasifikasi dalam memprediksi data baru. Evaluasi model dilakukan dengan menggunakan metrik-metrik evaluasi klasifikasi berikut:

- **Accuracy**: Akurasi adalah proporsi jumlah prediksi yang benar (positif dan negatif) dibandingkan dengan total prediksi. Cocok digunakan ketika distribusi kelas seimbang.
  Formula:  
  ![Accuracy](https://github.com/user-attachments/assets/4f0bdd4a-12db-4cde-862c-65d5cccf8ea9)

- **Precision**: Presisi mengukur seberapa akurat prediksi positif dari model. Artinya, dari semua yang diprediksi sebagai positif, berapa banyak yang benar-benar positif. Cocok ketika false positive lebih berdampak besar, misalnya pada diagnosa penyakit.
  Formula:  
  ![Precision](https://github.com/user-attachments/assets/4d1e0bf6-cf26-4286-a4cb-b1bf476ba0e5)

- **Recall**: Recall menunjukkan seberapa banyak dari kasus positif yang berhasil dideteksi dengan benar oleh model. Cocok ketika false negative berbahaya, seperti gagal mendeteksi pasien sakit.
  Formula:  
  ![Recall](https://github.com/user-attachments/assets/8bfb5177-b2b0-41da-8e0d-c42c773dfa04)

- **F1-Score**: F1-Score adalah rata-rata harmonik dari Precision dan Recall. Digunakan saat membutuhkan keseimbangan antara presisi dan recall
  Formula:  
  ![F1-Score](https://github.com/user-attachments/assets/7a8a4a0e-64de-464e-a8ea-9981bb315b58)

  Keterangan:
    - TP = True Positive (prediksi positif yang benar)
    = TN = True Negative (prediksi negatif yang benar)
    - FP = False Positive (prediksi positif yang salah)
    - FN = False Negative (prediksi negatif yang salah)

### Kenapa Menggunakan Metrik Evaluasi ?
1. Accuracy saja tidak cukup.<br>
   Karena ini adalah kasus medis, konsekuensi dari kesalahan prediksi sangat penting:
    - False Positive (prediksi mengidap, padahal tidak) bisa menyebabkan kecemasan dan pengobatan yang tidak perlu.
    - False Negative (prediksi tidak mengidap, padahal mengidap) lebih berbahaya karena pasien tidak ditangani padahal perlu.
   Perlu diperhatikan juga:
    - Recall → Penting untuk memastikan pasien yang benar-benar sakit tidak terlewat.
    - Precision → Penting agar tidak terlalu banyak pasien yang sehat dikira sakit.
    - F1-Score → Dipakai untuk menyeimbangkan keduanya.

2. Model terbaik bukan hanya yang punya akurasi tinggi.<br>
   evaluasi multi-metrik (Accuracy, Precision, Recall, F1) membantu kita memilih model yang tidak hanya akurat secara keseluruhan, tetapi juga sensitif  terhadap pasien yang benar-benar mengidap diabetes.

3. Metrik evaluasi menjadi penilaian apakah perubahan/penyesuaian fitur tersebut meningkatkan performa model atau tidak.<br>
   Metrik tidak langsung menjawab hubungan antar fitur, tetapi membantu menilai dampak jika suatu fitur diubah, dihapus, atau diprioritaskan.

   
**Hasil Evaluasi Model**
Berikut adalah hasil evaluasi terhadap ketiga model yang digunakan:

| Model               | Accuracy | Precision | Recall | F1 Score |
|---------------------|----------|-----------|--------|----------|
| KNN                 | 0.98     | 0.97      | 0.99   | 0.98     |
| Random Forest       | 0.99     | 0.98      | 0.98   | 0.98     |
| Logistic Regression | 0.76     | 0.64      | 0.72   | 0.68     |

Catatan: Evaluasi dilakukan terhadap data uji (X_test) setelah model dilatih menggunakan data hasil oversampling SMOTE dan data training yang telah diskalakan.

**Interpretasi dan Insight**
- KNN menjadi model yang direkomendasikan dari ketiga model dengan memprioritaskan recall tertinggi yaitu 0.9948 artinya, hampir semua kasus positif (penyakit) berhasil dideteksi dengan benar. Artinya, model tidak melewatkan satu pun pasien yang benar-benar mengidap diabetes dalam prediksinya (False Negative = 0). Hal ini sangat penting dalam konteks medis, karena salah satu tujuan utama adalah mendeteksi semua kasus positif secara tepat.
- Random Forest juga cukup baik, namun sedikit kalah di aspek precision dan recall , yang berarti masih terdapat sejumlah kecil false positive.
- Logistic Regression memberikan performa paling rendah dalam semua metrik, meskipun telah dilakukan hyperparameter tuning. Ini menunjukkan model ini kurang cocok untuk menangkap pola non-linear atau kompleksitas data dalam kasus ini.

## Kesimpulan

- Proyek ini berhasil membangun model prediksi diabetes dengan menggunakan data medis sederhana.
- Dari tiga model yang diuji (KNN, Random Forest, Logistic Regression), KNN dijadikan model terbaik dengan akurasi 98.91% dan recall tertinggi 99.47%, menjadikannya pilihan utama untuk digunakan dalam prediksi risiko diabetes.
- Penanganan nilai nol yang tidak valid dan imbalance data menggunakan median imputation dan SMOTE sangat berkontribusi dalam meningkatkan performa model.
- Hyperparameter tuning pada Logistic Regression memberikan sedikit peningkatan, tetapi model ini tetap kurang cocok dibanding dua model lainnya untuk dataset ini.

## Deployment

Link deploy : https://predictive-analytic-diabetes.streamlit.app/
Aplikasi ini telah dikembangkan menggunakan Streamlit untuk memberikan antarmuka interaktif dalam memvisualisasikan data dan memprediksi risiko diabetes secara interaktif, user dapat melihat visualisasi data dan mencoba hasil prediksi dengan 3 skema model.
Aplikasi memerlukan library berikut untuk dijalankan:
- streamlit
- pandas
- numpy
- plotly
- scikit-learn
- joblib
- matplotlib

 ```
Struktur folder proyek yang kamu perlukan untuk menjalankan deployment secara lokal
predictive-analytic---diabetes/
├── app.py                              # File utama untuk menjalankan aplikasi Streamlit
├── requirements.txt                    # Daftar dependencies Python
├── diabetes-ribbonblue.jpg             # Gambar ikon atau header visual di halaman
├── cleaned_dataset/
│   └── Cleaned_Healthcare_Diabetes.csv # Dataset yang sudah dibersihkan untuk ditampilkan dan diprediksi
├── model/
│   ├── rf_model.pkl                    # Model Random Forest terlatih
│   ├── lr_model.pkl                    # Model Logistic Regression terlatih
│   ├── knn_model.pkl                   # Model K-Nearest Neighbors terlatih
│   └── scaler.pkl                      # Scaler (misal StandardScaler) yang digunakan sebelum prediksi
 ```

#####🛠️ Cara Menjalankan Aplikasi Secara Lokal

1. Pastikan file app.py ada di root folder
2. Aktifkan virtual environment (opsional tapi direkomendasikan)
3. Install semua dependency dengan : pip install -r requirements.txt
4. Jalankan aplikasi Streamlit di terminal code editor : streamlit run app.py
