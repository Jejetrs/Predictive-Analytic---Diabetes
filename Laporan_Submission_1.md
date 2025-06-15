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

Dalam proyek ini, tujuan utama adalah untuk membangun model prediktif yang dapat mengidentifikasi individu dengan risiko diabetes berdasarkan data kesehatan mereka. Model ini diharapkan dapat digunakan untuk membantu tenaga medis dalam melakukan skrining dini terhadap diabetes, yang berfokus pada pengolahan data medis yang sederhana namun memberikan hasil yang akurat dan dapat diandalkan.

Bagian laporan ini mencakup:

### Problem Statements

Menjelaskan pernyataan masalah latar belakang:
- Bagaimana cara memprediksi apakah seseorang mengidap diabetes berdasarkan data yang tersedia?
- Algoritma Machine Learning apa yang paling efektif dalam prediksi diabetes berdasarkan akurasi dan performa lainnya?
- Apa saja fitur kesehatan yang paling mempengaruhi klasifikasi risiko diabetes?

### Goals
Untuk menjawab permasalahan yang ada, proyek ini bertujuan untuk:
1. Mengembangkan model klasifikasi yang mampu memprediksi risiko diabetes dengan akurasi tinggi berdasarkan data medis pasien.
2. Membandingkan efektivitas dari tiga algoritma klasifikasi (Logistic Regression, KNN, Random Forest) ntuk menentukan model terbaik dalam memprediksi diabetes, sehingga hasil prediksi dapat dipercaya untuk mendukung keputusan awal.
3. Mengidentifikasi fitur-fitur yang mempengaruhi hasil prediksi untuk mendukung interpretasi medis yang lebih baik.

#### **“Solution Statement”**
Untuk mewujudkan tujuan proyek, pendekatan yang digunakan adalah sebagai berikut:

1. Menggunakan tiga algoritma klasifikasi utama, yaitu:
    - Logistic Regression sebagai baseline model yang sederhana dan interpretable.
    - K-Nearest Neighbors (KNN) untuk menangkap pola lokal antar data.
    - Random Forest Classifier sebagai model ensemble untuk menangani non-linearitas dan meningkatkan generalisasi.

2. Menangani masalah ketidakseimbangan data (imbalanced classes) dengan SMOTE (Synthetic Minority Over-sampling Technique) untuk menyeimbangkan distribusi antara kelas penderita dan non-penderita diabetes.

3. Melakukan Hyperparameter Tuning untuk Meningkatkan Performa, Prediksi, dan Penanganan Underfitting dan Overfitting.

4. Menerapkan evaluasi model menggunakan metrik klasifikasi yang relevan (accuracy, precision, recall, dan F1-score) untuk memastikan performa menyeluruh yang dapat diandalkan.

5. Membandingkan hasil dan memilih model terbaik berdasarkan performa evaluasi, stabilitas, dan interpretabilitas untuk dapat digunakan dalam implementasi praktis.

6. Menyediakan solusi prediktif yang tidak hanya akurat tetapi juga mudah diimplementasikan dalam praktik medis, dengan antarmuka web aplikasi yang ramah pengguna dan dapat diandalkan dalam deteksi dini diabetes.


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
| Model                              | Parameter                                                                                                                                                    |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| K-Nearest Neighbors (KNN) (awal)   | `n_neighbors=15`, `weights='distance'`                                                                                                                       |
| K-Nearest Neighbors (KNN) (tuning) | `n_neighbors=range(3, 21, 2)`, `weights=['uniform', 'distance']`, `metric=['euclidean', 'manhattan', 'minkowski']`                                           |
| Random Forest (awal)               | `n_estimators=100`, `random_state=101`                                                                                                                       |
| Random Forest (tuning)             | `n_estimators=[100, 200]`, `max_depth=[None, 10, 20]`, `min_samples_split=[2, 5]`, `min_samples_leaf=[1, 2]`, `max_features='sqrt'`, `random_state=42`       |
| Logistic Regression (awal)         | `max_iter=1000`                                                                                                                                              |
| Logistic Regression (tuning)       | `penalty='elasticnet'`, `solver='saga'`, `l1_ratio=[0.1, 0.5, 0.9]`, `max_iter=1000`, `random_state=42`                                                     |


Model-model ini dipilih karena masing-masing mewakili pendekatan yang berbeda dalam klasifikasi, mulai dari yang sederhana hingga yang lebih kompleks, sehingga bisa dibandingkan performanya secara adil terhadap dataset diabetes ini.

Tujuan dari penggunaan beberapa model ini adalah untuk membandingkan performa mereka dalam memprediksi apakah seseorang berisiko terkena diabetes berdasarkan dataset yang digunakan.

**Kelebihan dan kekurangan dari algoritma yang digunakan**

1. K-Nearest Neighbors (KNN)<br>
    KNN adalah model yang sangat intuitif dan mudah dipahami. Cara kerjanya yaitu mencari sejumlah "tetangga terdekat" dari sebuah titik data baru, lalu memprediksi kelasnya berdasarkan mayoritas label dari tetangga tersebut. Jarak yang digunakan untuk mengukur kedekatan biasanya adalah Euclidean distance.

    Kelebihan:
    - Sederhana dan tidak butuh banyak asumsi.
    - Cocok untuk dataset berukuran kecil hingga sedang..

    Kekurangan:
    - Performanya bisa turun jika datanya banyak (scalability).
    - Sensitif terhadap skala fitur dan noise, perlu dilakukan normalisasi.
    
    Pada model ini, KNN dikonfigurasi dengan 15 tetangga terdekat dan menggunakan bobot berdasarkan jarak (weights='distance'), sehingga tetangga yang lebih dekat memiliki pengaruh lebih besar.

    ```python
    knn_model = KNeighborsClassifier(n_neighbors=15, weights='distance')
    knn_model.fit(X_train_resampled, y_train_resampled)
    y_pred_knn = knn_model.predict(X_test_scaled)
    ```

    Pemilihan nilai k=15 bertujuan untuk menstabilkan prediksi. Nilai k yang terlalu kecil (misalnya 3 atau 5) dapat membuat model terlalu sensitif terhadap noise (overfitting), sementara nilai yang lebih besar seperti 15 memungkinkan model menangkap pola umum dengan lebih baik.

    Tuning KNN:
    Dikarenakan performa KNN sangat bergantung pada nilai k dan cara menghitung jarak walaupun akurasi sudah sangat baik, hypertuning diperlukan untuk menemukan kombinasi parameter terbaik agar model tidak overfit dan dapat menangkap pola data dengan optimal.

    Dengan menggunakan GridSearchCV, pada model ini mencoba berbagai kombinasi parameter:
    - n_neighbors ganjil antara 3 hingga 19,
    - weights antara 'uniform' dan 'distance',
    - metric termasuk 'euclidean', 'manhattan', dan 'minkowski'.

    ```python
    param_grid_knn = {
    'n_neighbors': list(range(3, 21, 2)),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']}
    ```

    Setelah tuning, ditemukan parameter terbaik yang meningkatkan performa model:
    ```python
    Best parameters for KNN: {'metric': 'euclidean', 'n_neighbors': 17, 'weights': 'distance'}
    ```

    Model KNN terbaik dari tuning ini kemudian digunakan untuk prediksi dan menunjukkan peningkatan skor F1 Score dibanding model awal.
    

2. Random Forest<br>
    Random Forest adalah model berbasis ensembling yang menggunakan banyak pohon keputusan (decision tree). Ibaratnya, daripada menebak keputusan berdasarkan satu "pendapat", model ini mengumpulkan banyak "pendapat dari pohon-pohon" dan mengambil suara terbanyak.

    Kelebihan:
    - Sangat kuat untuk data yang kompleks dan interaksi non-linier.
    - Tidak terlalu sensitif terhadap data outlier atau noise.
    - Mampu mengukur pentingnya fitur (feature importance).

    Kekurangan:
    - Kurang interpretatif (agak sulit dijelaskan ke non-teknis).
    - Memerlukan sumber daya lebih besar untuk pelatihan.
    - Waktu training bisa lebih lama dibanding model sederhana.
    - Performa bisa menurun jika data terlalu noisy.

    Setiap decision tree dibentuk dari subset data dan fitur yang berbeda-beda (ini disebut bagging), jadi model ini lebih kuat dalam menangani overfitting dibanding satu decision tree saja.
    
    ```python
    rf_model = RandomForestClassifier(n_estimators=100, random_state=101)
    rf_model.fit(X_train_resampled, y_train_resampled)
    y_pred_rf = rf_model.predict(X_test_scaled)
    ```

    Penggunaan random_state=101 bertujuan agar eksperimen bersifat reproducible (hasil konsisten saat dijalankan ulang), meskipun angkanya bebas (arbitrary).

    Tuning Random Forest:
    Dikarenakan Random Forest memiliki banyak parameter yang memengaruhi kompleksitas dan generalisasi, hypertuning penting untuk menghindari underfitting atau overfitting, serta mencari konfigurasi terbaik untuk performa optimal guna menjaga keseimbangan antara bias dan variance.

    Dengan GridSearchCV, model ini dilakukan eksplorasi kombinasi parameter:
    - n_estimators (jumlah pohon): 100, 200
    - max_depth: None, 10, 20
    - min_samples_split: 2, 5
    - min_samples_leaf: 1, 2
    - max_features: 'sqrt'

    Setelah tuning, ditemukan parameter terbaik yang meningkatkan performa model:
    ```python
    Best Parameters RF: {'max_depth': 20, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 100}
    ```
    Model hasil tuning ini kemudian diuji dan menunjukkan peningkatan akurasi dan f1 score dibanding model dasar.

3. Logistic Regression<br>
    Logistic Regression adalah model dasar yang sering digunakan dalam klasifikasi biner. Walaupun namanya "regression", model ini bukan untuk memprediksi angka, tapi Logistic Regression adalah model klasifikasi dasar yang menghitung probabilitas yang termasuk dalam suatu kelas (misalnya, terkena diabetes atau tidak). Model ini cukup interpretatif dan sering digunakan sebagai baseline dalam banyak proyek klasifikasi.

    Cara kerjanya adalah menghitung peluang (probabilitas) seseorang terkena diabetes berdasarkan nilai fitur-fitur seperti kadar glukosa, tekanan darah, dll. Jika probabilitasnya di atas 0.5, maka model akan memprediksi bahwa orang tersebut terkena diabetes (label = 1).

    Kelebihan:
    - Sederhana dan mudah diinterpretasikan.
    - Cocok sebagai baseline model.
    - Tidak mudah overfitting jika datanya bersih.

    Kekurangan:
    - Tidak cocok untuk data dengan hubungan yang sangat kompleks dan tidak linier.

    ```python
    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train_resampled, y_train_resampled)
    y_pred_log = log_model.predict(X_test_scaled)
    ```

    Parameter max_iter=1000 digunakan untuk memastikan proses optimasi konvergen, terutama jika dataset kompleks. Nilai default (100) sering kali tidak cukup dan menghasilkan peringatan ConvergenceWarning.

    Namun model pertama pada ini (max_iter=1000) memiliki nilai akurasi hanya 76.89% jauh di bawah KNN dan Random Forest yang mencapai hampir 99%.

    Tuning Parameter Logistic Regression:
    Hypertuning diperlukan untuk menemukan regularisasi dan solver terbaik agar model tidak overfit dan tetap stabil serta diharapkan dapat meningkatkan akurasi model, terutama pada data dengan fitur multikolinearitas, namun pada kasus ini peningkatannya terbatas.

    Dengan GridSearchCV, pada model ini dicoba parameter berikut untuk meningkatkan performa:
    - penalty: elasticnet
    - solver: saga
    - l1_ratio: 0.1, 0.5, 0.9
   
    Setelah tuning, ditemukan parameter terbaik yang meningkatkan performa model:
    ```python
    Best Parameters: {'l1_ratio': 0.1, 'penalty': 'elasticnet', 'solver': 'saga'}
    ```

    Namun peningkatan performa setelah tuning masih terbatas, dengan akurasi hanya mencapai 76.71% dan F1 score di bawah KNN dan Random Forest. Ini menunjukkan Logistic Regression kurang cocok untuk dataset ini.

Berdasarkan evaluasi terhadap ketiga model yang telah dilatih yaitu K-Nearest Neighbors (KNN), Random Forest, dan Logistic Regression — model yang dipilih sebagai solusi terbaik untuk kasus prediksi diabetes ini adalah:

**✅ K-Nearest Neighbors (KNN)**

Alasan Pemilihan KNN sebagai Model Terbaik:

- Akurasi Tinggi<br>
KNN berhasil mencapai akurasi sebesar 99.45% setelah tuning. Artinya, model ini mampu memprediksi dengan benar hampir seluruh data uji, menunjukkan kinerjanya sangat baik dalam memahami pola pada data.

- Performa Konsisten Tanpa Overfitting<br>
Meskipun akurasinya sangat tinggi, KNN tetap menunjukkan stabilitas dan tidak overfitting karena menggunakan bobot berdasarkan jarak, membuat prediksinya lebih “bijak” terhadap tetangga terdekat yang paling relevan.

- Akurasi Metrik Evaluasi<br>
enghasilkan nilai akurasi, precision, recall, dan F1-score terbaik serta paling seimbang di antara ketiganya,terutama pada Recall KNN 99.47% dan F1 Score 99.21% artinya, hampir semua kasus positif (penyakit) berhasil dideteksi dengan benar. Ini sangat penting untuk skenario seperti deteksi penyakit, di mana tidak mendeteksi pasien yang sebenarnya sakit dapat berbahaya.
  
- Tidak Memerlukan Asumsi Khusus<br>
KNN tidak mengharuskan asumsi seperti linearitas hubungan antar fitur (berbeda dengan Logistic Regression). Ini sangat membantu jika data bersifat non-linear atau memiliki distribusi yang tidak biasa.

- Kemudahan Implementasi dan Interpretasi<br>
Secara konsep, KNN sangat mudah dipahami oleh praktisi maupun pihak non-teknis: prediksi dilakukan dengan melihat mayoritas tetangga terdekat. Ini penting jika model akan digunakan dalam sistem real dengan pengguna umum seperti tenaga medis.

## Evaluation

Evaluasi model ini bertujuan untuk mengukur seberapa baik kinerja model dalam memprediksi risiko diabetes pada data baru, serta untuk memastikan bahwa model yang dibangun memenuhi tujuan yang lebih luas sesuai dengan Business Understanding. Evaluasi ini melibatkan analisis hasil model berdasarkan metrik-metrik evaluasi yang relevan dalam konteks medis, di mana kesalahan prediksi dapat memiliki dampak signifikan.

### Metrik Evaluasi <br>
  Metrik evaluasi digunakan untuk mengukur dan menilai performa model dalam konteks klasifikasi. Dalam kasus ini, saya menggunakan beberapa metrik untuk mengevaluasi model secara komprehensif:

  - **Accuracy**: Akurasi adalah proporsi jumlah prediksi yang benar (positif dan negatif) dibandingkan dengan total prediksi.
    Formula:  
    ![Accuracy](https://github.com/user-attachments/assets/4f0bdd4a-12db-4cde-862c-65d5cccf8ea9)

    Dalam kasus ini, akurasi model KNN mencapai 99.45%, yang menunjukkan bahwa model ini sangat baik dalam memberikan prediksi yang benar untuk mayoritas data.

  - **Precision**: Presisi mengukur seberapa akurat prediksi positif dari model. Artinya, dari semua yang diprediksi sebagai positif, berapa banyak yang benar-benar positif. Cocok ketika false positive lebih berdampak besar, misalnya pada diagnosa penyakit.
    Formula:  
    ![Precision](https://github.com/user-attachments/assets/4d1e0bf6-cf26-4286-a4cb-b1bf476ba0e5)

    Dalam konteks prediksi diabetes, presisi penting untuk memastikan bahwa pasien yang diprediksi mengidap diabetes benar-benar mengidap penyakit tersebut. Model KNN menunjukkan presisi 98%, yang berarti hampir semua prediksi positif yang dihasilkan oleh model benar adanya.

    Hal ini sangat penting dalam konteks medis, di mana kegagalan untuk mendeteksi pasien yang benar-benar mengidap diabetes (false negative) bisa berbahaya. Model KNN memberikan recall 99.47%, yang menunjukkan bahwa hampir semua pasien yang mengidap diabetes berhasil terdeteksi oleh model, tanpa ada yang terlewat (false negative = 0).

  - **Recall**: Recall menunjukkan seberapa banyak dari kasus positif yang berhasil dideteksi dengan benar oleh model. Cocok ketika false negative berbahaya, seperti gagal mendeteksi pasien sakit.
    Formula:  
    ![Recall](https://github.com/user-attachments/assets/8bfb5177-b2b0-41da-8e0d-c42c773dfa04)

    Model KNN mendapatkan F1-score 99.21%, yang menunjukkan bahwa model ini tidak hanya akurat, tetapi juga sensitif terhadap kasus positif yang harus terdeteksi.

  - **F1-Score**: F1-Score adalah rata-rata harmonik dari Precision dan Recall. Digunakan saat membutuhkan keseimbangan antara presisi dan recall
    Formula:  
    ![F1-Score](https://github.com/user-attachments/assets/7a8a4a0e-64de-464e-a8ea-9981bb315b58)

    Keterangan:
      - TP = True Positive (prediksi positif yang benar)
      = TN = True Negative (prediksi negatif yang benar)
      - FP = False Positive (prediksi positif yang salah)
      - FN = False Negative (prediksi negatif yang salah)

  #### Kenapa Menggunakan Metrik Evaluasi ?
  1. Accuracy saja tidak cukup.<br>
      Karena ini adalah kasus medis, konsekuensi dari kesalahan prediksi sangat penting:
        - False Positive (prediksi mengidap, padahal tidak) bisa menyebabkan kecemasan dan pengobatan yang tidak perlu.
        - False Negative (prediksi tidak mengidap, padahal mengidap) lebih berbahaya karena pasien tidak ditangani padahal perlu.
      Perlu diperhatikan juga:
        - Recall → Penting untuk memastikan pasien yang benar-benar sakit tidak terlewat.
        - Precision → Penting agar tidak terlalu banyak pasien yang sehat dikira sakit.
        - F1-Score → Dipakai untuk menyeimbangkan keduanya.

  2. Model terbaik bukan hanya yang punya akurasi tinggi.<br>
      Evaluasi multi-metrik (Accuracy, Precision, Recall, F1) membantu kita memilih model yang tidak hanya akurat secara keseluruhan, tetapi juga sensitif  terhadap pasien yang benar-benar mengidap diabetes.

  3. Metrik evaluasi menjadi penilaian apakah perubahan/penyesuaian fitur tersebut meningkatkan performa model atau tidak.<br>
      Metrik tidak langsung menjawab hubungan antar fitur, tetapi membantu menilai dampak jika suatu fitur diubah, dihapus, atau diprioritaskan.

    
  **Hasil Evaluasi Model**
  Berikut adalah hasil evaluasi terhadap ketiga model yang digunakan:

  | Model               | Accuracy | Precision | Recall | F1 Score |
  |---------------------|----------|-----------|--------|----------|
  | KNN                 | 0.99     | 0.98      | 0.99   | 0.99     |
  | Random Forest       | 0.98     | 0.98      | 0.97   | 0.98     |
  | Logistic Regression | 0.76     | 0.64      | 0.72   | 0.68     |

  Catatan: Evaluasi dilakukan terhadap data uji (X_test) setelah model dilatih menggunakan data hasil oversampling SMOTE (X_train_resampled) dan data training yang telah diskalakan.

  **Interpretasi dan Insight**
  Berdasarkan hasil evaluasi, 
  - KNN menjadi model yang direkomendasikan karena memiliki akurasi dan recall yang sangat tinggi (99.45% dan 99.47%), yang menunjukkan bahwa model ini mampu mendeteksi hampir semua kasus positif secara akurat, dengan false negative yang sangat rendah (0). Hal ini sangat penting dalam konteks medis, di mana tidak mendeteksi pasien yang benar-benar mengidap diabetes dapat menyebabkan konsekuensi serius.
  - Random Forest juga memberikan hasil yang baik dengan akurasi 98%, namun masih sedikit kalah dalam hal recall dan F1-score dibandingkan KNN, yang menunjukkan bahwa meskipun model ini cukup stabil, ada sedikit trade-off dalam performa yang lebih rendah pada deteksi kasus positif.
  - Di sisi lain, Logistic Regression memberikan hasil yang paling rendah dalam semua metrik, dengan akurasi hanya 76.89%. Hasil ini menunjukkan bahwa Logistic Regression tidak mampu menangkap kompleksitas data secara optimal, terutama ketika hubungan antar fitur bersifat non-linear, yang merupakan karakteristik dari dataset diabetes ini. Meskipun sudah dilakukan hyperparameter tuning, performa Logistic Regression masih terbatas.

### Hubungan dengan Business Understanding<br>
Proyek ini bertujuan untuk membangun model prediktif yang dapat membantu mendeteksi risiko diabetes menggunakan data medis yang sederhana dan terjangkau. Tujuan utama adalah menyediakan alat bantu yang cepat, akurat, dan hemat biaya untuk mendukung keputusan medis, khususnya di negara berpenghasilan rendah dan menengah yang sering kali kekurangan fasilitas dan tenaga medis terlatih. Evaluasi ini bertujuan untuk memastikan bahwa model yang dibangun tidak hanya efektif dalam memprediksi risiko diabetes, tetapi juga relevan dengan masalah yang ada, serta mendukung tujuan utama proyek untuk mendeteksi diabetes lebih awal.

Evaluasi juga berfokus pada bagaimana model ini dapat digunakan untuk menjawab problem statement yang telah ditetapkan, yaitu memprediksi apakah seseorang berisiko diabetes, memilih algoritma yang paling efektif, dan mengidentifikasi fitur-fitur medis yang paling berpengaruh dalam prediksi tersebut.

**Menjawab Problem Statement**
  - Bagaimana cara memprediksi apakah seseorang mengidap diabetes berdasarkan data yang tersedia?<br>
    Model yang dibangun menggunakan tiga algoritma klasifikasi utama—KNN, Random Forest, dan Logistic Regression—berhasil memberikan prediksi yang sangat akurat tentang risiko diabetes. Model KNN, dengan akurasi 99.45%, mampu mendeteksi hampir seluruh kasus positif (Recall 99.47%), yang berarti model ini sangat baik dalam mendeteksi pasien yang benar-benar mengidap diabetes.

  - Algoritma Machine Learning apa yang paling efektif dalam prediksi diabetes?<br>
    Berdasarkan evaluasi metrik performa, KNN terbukti menjadi algoritma yang paling efektif dalam memprediksi diabetes. KNN memberikan hasil terbaik pada semua metrik evaluasi, terutama dalam hal recall, yang penting untuk mendeteksi semua pasien yang benar-benar sakit.

  - Apa saja fitur kesehatan yang paling memengaruhi klasifikasi risiko diabetes?<br>
    Fitur-fitur seperti Glucose, BMI, dan Age menunjukkan korelasi yang sangat kuat dengan Outcome (status diabetes). Fitur-fitur ini memainkan peran kunci dalam membentuk prediksi model, yang dapat memberikan wawasan penting untuk pengambilan keputusan medis. Model KNN yang menunjukkan kinerja terbaik, didorong oleh fitur-fitur ini yang berpengaruh besar dalam mengklasifikasikan seseorang berisiko diabetes atau tidak.

### Mencapai Goals yang Diharapkan<br>
  - Goal 1: Mengembangkan model klasifikasi dengan akurasi tinggi untuk memprediksi risiko diabetes.<br>
      KNN berhasil mencapai akurasi 99.45%, yang sangat tinggi, dan memenuhi tujuan utama untuk membangun model yang dapat diandalkan dalam memprediksi risiko diabetes. Model ini sangat cocok untuk diterapkan dalam konteks medis karena kemampuannya dalam mendeteksi pasien yang benar-benar sakit.

  - Goal 2: Membandingkan efektivitas tiga algoritma klasifikasi (Logistic Regression, KNN, Random Forest).<br>
      Evaluasi ini berhasil membandingkan ketiga algoritma secara objektif, dengan KNN menjadi yang terbaik berdasarkan akurasi, recall, precision, dan F1-score. Hasil ini memberikan bukti bahwa KNN adalah algoritma yang lebih efektif dan stabil dibandingkan dengan model lainnya.

  - Goal 3: Mengidentifikasi fitur-fitur kunci yang memengaruhi hasil prediksi.<br>
      Proses analisis korelasi berhasil mengidentifikasi fitur-fitur utama yang paling mempengaruhi prediksi risiko diabetes. Fitur-fitur seperti Glucose, BMI, dan Age memainkan peran penting dalam membentuk prediksi, memberikan wawasan yang sangat berguna bagi tenaga medis dalam pengambilan keputusan. Heatmap  dan Aplikasi yang dibangun memungkinkan visualisasi data dan hubungan antar fitur ini untuk memberikan pemahaman yang lebih baik dalam konteks klinis.

#### Dampak Solusi Statement<br>
Untuk mewujudkan tujuan proyek dan memberikan solusi yang efektif dalam deteksi dini diabetes, pendekatan yang digunakan adalah sebagai berikut:

  - Menggunakan Tiga Algoritma Klasifikasi Utama<br>
    Pendekatan ini bertujuan untuk membandingkan dan memilih model yang paling efektif dalam memprediksi risiko diabetes. Tiga algoritma yang digunakan adalah:

      1. K-Nearest Neighbors (KNN) unggul dalam menangkap pola lokal antar data, memberikan hasil yang sangat baik dalam mengidentifikasi individu yang berisiko diabetes.

      2. Random Forest menggabungkan banyak pohon keputusan untuk menangani kompleksitas data yang tidak linear, sehingga memberikan prediksi yang lebih stabil dan tepat dalam konteks risiko diabetes.

      3. Logistic Regression memberikan dasar yang mudah diinterpretasi, yang membantu tenaga medis memahami bagaimana faktor-faktor kesehatan memengaruhi prediksi diabetes.

      Dampak penggunaan ketiga algoritma ini adalah kemampuan untuk memilih model yang paling akurat dan dapat diandalkan, yang sangat penting untuk deteksi dini diabetes dalam praktek medis.

  - Menangani Masalah Ketidakseimbangan Data (Imbalanced Classes) Menggunakan SMOTE untuk menyeimbangkan distribusi antara kelas penderita dan non-penderita diabetes.<br>
    Dataset yang digunakan memiliki distribusi kelas yang tidak seimbang, di mana penderita diabetes jauh lebih sedikit dibandingkan dengan yang tidak. Ketidakseimbangan ini dapat menyebabkan model menjadi bias, lebih memprediksi kelas mayoritas dan kurang memperhatikan kelas minoritas (penderita diabetes). Dengan menggunakan teknik SMOTE (Synthetic Minority Over-sampling Technique), distribusi kelas dapat diseimbangkan, memberikan lebih banyak contoh pada kelas minoritas sehingga model dapat belajar dengan lebih baik untuk mendeteksi risiko diabetes.

      Dampak dari penggunaan SMOTE adalah peningkatan kemampuan model dalam mendeteksi diabetes, mengurangi kemungkinan bias yang dapat menyebabkan kesalahan dalam diagnosa dan meminimalkan false negatives, di mana penderita diabetes tidak terdeteksi oleh model.

  - Melakukan Hyperparameter Tuning untuk Meningkatkan Performa, Prediksi, dan Penanganan Underfitting dan Overfitting<br>
    Hyperparameter tuning memastikan bahwa model yang digunakan memiliki performa yang optimal, baik dalam hal akurasi maupun keseimbangan prediksi antar kelas. Dengan melakukan penyetelan parameter yang tepat, model dapat dihindari dari masalah overfitting atau underfitting, yang dapat mengurangi akurasi prediksi pada data baru.

      Dampak dari tuning ini adalah peningkatan stabilitas dan keakuratan model, yang menjadikannya lebih andal saat digunakan di dunia nyata, khususnya dalam deteksi dini diabetes, di mana setiap keputusan medis harus didasarkan pada hasil yang sangat akurat.

  - Menerapkan Evaluasi Model Menggunakan Metrik Klasifikasi yang Relevan (Accuracy, Precision, Recall, dan F1-score)<br>
    Model dievaluasi dengan metrik-metrik seperti accuracy, precision, recall, dan F1-score. Evaluasi ini memberikan gambaran yang lebih jelas tentang kinerja model, terutama dalam konteks medis di mana kesalahan prediksi bisa berbahaya. Dengan memperhatikan metrik seperti recall, dimaksudkan memastikan bahwa model dapat mendeteksi sebanyak mungkin pasien yang benar-benar sakit, yang sangat penting untuk mencegah false negative (gagal mendeteksi diabetes).

      Dampak dari evaluasi ini dapat dilihat dari evaluasi metrik masing - masing model menunjukkan adanya peningkatan keandalan model dalam membuat prediksi yang tepat dengan pencegahan overfitting, hal ini dapat membantu dalam mengambil keputusan yang lebih tepat dan aman.

  - Membandingkan Hasil dan Memilih Model Terbaik Berdasarkan Performa Evaluasi, Stabilitas, dan Interpretabilitas<br>
    Kriteria utama dalam pemilihan model adalah stabilitas, akurasi prediksi, dan kemampuan untuk diinterpretasi dalam konteks medis. Ini memastikan bahwa model yang dipilih dapat diandalkan untuk penggunaan praktis dan memudahkan dalam memahami hasil prediksi serta mengambil keputusan berdasarkan data.

      Dampak dari memilih model terbaik adalah memberikan solusi yang optimal yang dapat langsung digunakan untuk skrining risiko diabetes, dengan hasil yang mudah dipahami dan diterjemahkan ke dalam langkah-langkah medis yang tepat.

  - Menyediakan solusi prediktif yang tidak hanya akurat tetapi juga mudah diimplementasikan dalam praktik medis, dengan antarmuka web aplikasi yang ramah pengguna dan dapat diandalkan dalam deteksi dini diabetes.<br>
    Aplikasi berbasis Streamlit yang dikembangkan dalam proyek ini memberikan antarmuka pengguna yang intuitif dan ramah pengguna, sehingga memudahkan tenaga medis atau siapa pun untuk melakukan prediksi risiko diabetes hanya dengan memasukkan data kesehatan pasien. Aplikasi ini memungkinkan pengguna untuk memilih model prediksi yang diinginkan (KNN, Random Forest, atau Logistic Regression) dan memberikan hasil yang cepat serta mudah dimengerti.

      Dampak dari aplikasi ini adalah efisiensi waktu dan penghematan biaya, terutama jika sumber daya medis terbatas. Aplikasi ini memberikan akses mudah dan cepat untuk melakukan skrining, yang dapat mempercepat proses diagnosa dan mengurangi waktu tunggu untuk deteksi risiko diabetes.

## Kesimpulan

Berdasarkan hasil evaluasi ini, proyek ini berhasil membangun model prediktif yang efektif dalam mendeteksi risiko diabetes menggunakan data medis yang relatif sederhana. KNN muncul sebagai model terbaik dengan akurasi dan recall yang tinggi, sehingga sangat direkomendasikan untuk implementasi lebih lanjut dalam skrining risiko diabetes. Penggunaan teknik seperti SMOTE untuk mengatasi ketidakseimbangan data dan hyperparameter tuning yang diterapkan pada KNN dan Random Forest juga berkontribusi besar terhadap peningkatan performa model. Model ini diharapkan dapat memberikan dampak yang signifikan dalam mendeteksi dini diabetes dan mendukung keputusan medis yang lebih cepat dan akurat. 

Evaluasi ini juga mengonfirmasi bahwa tujuan utama dari proyek ini, yaitu untuk mengembangkan alat prediktif yang dapat diandalkan, telah tercapai, dengan model yang tidak hanya akurat tetapi juga dapat diinterpretasikan oleh para tenaga medis untuk keputusan klinis yang lebih baik.

## Testing / Deployment

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
Struktur folder proyek yang diperlukan untuk menjalankan deployment secara lokal

app/
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
2. Aktifkan virtual environment (opsional tapi direkomendasikan) :  python -m venv myenv  ===> (aktifkan) .\myenv\Scripts\activate ===> (nonaktifkan) deactivate
3. Install semua dependency dengan : pip install -r requirements.txt
4. Jika perlu upgrade pip (opsional): python.exe -m pip install --upgrade pip 
4. Jalankan aplikasi Streamlit di terminal code editor : streamlit run app.py