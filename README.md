# Laporan Proyek Machine Learning - Muhammad Haikal

## Project Overview

Di era digital saat ini, akses terhadap informasi kesehatan menjadi semakin mudah. Banyak individu yang mencoba melakukan diagnosis awal berdasarkan gejala yang mereka alami melalui pencarian di internet. Fenomena ini, meskipun didorong oleh keinginan untuk proaktif terhadap kesehatan, seringkali berujung pada kebingungan atau kecemasan akibat informasi yang tidak terstruktur, berlebihan, atau bahkan salah. Keterlambatan dalam mengenali pola gejala yang signifikan dan mendapatkan penanganan yang tepat dapat memperburuk kondisi kesehatan. Oleh karena itu, dibutuhkan sebuah sistem yang dapat membantu pengguna secara lebih terarah.

Proyek ini bertujuan untuk mengembangkan sistem rekomendasi penyakit berbasis gejala yang tidak hanya memberikan informasi awal mengenai kemungkinan kondisi medis, tetapi juga menyajikan informasi obat-obatan yang umumnya terkait dengan penyakit tersebut. Sistem ini diharapkan dapat menjadi panduan awal yang informatif, mendorong pengguna untuk melakukan konsultasi lebih lanjut dengan tenaga medis profesional, dan bukan sebagai pengganti diagnosis medis. Dengan memanfaatkan dataset gejala-penyakit dan dataset penyakit-obat, model *content-based filtering* akan dibangun untuk memberikan rekomendasi yang relevan.

Dataset yang digunakan dalam proyek ini bersumber dari Kaggle: [Medicine Recommendation System Dataset](https://www.kaggle.com/datasets/noorsaeed/medicine-recommendation-system-dataset/data).

**Referensi Tambahan**:
Komal Kumar, N., & Vigneswari, D. (2019, December). A drug recommendation system for multi-disease in health care using machine learning. In International Conference on Advanced Communication and Computational Technology (pp. 1-12). Singapore: Springer Nature Singapore. (Referensi ini menunjukkan adanya penelitian serupa dalam domain rekomendasi kesehatan).

## Business Understanding

### Problem Statements
Permasalahan utama yang ingin diselesaikan adalah bagaimana memberikan panduan awal yang lebih terstruktur kepada individu yang mengalami gejala penyakit tertentu.
1.  Bagaimana cara mengembangkan sistem yang mampu merekomendasikan daftar penyakit yang paling mungkin diderita pengguna berdasarkan sekumpulan gejala yang mereka inputkan?
2.  Bagaimana cara mengintegrasikan informasi pengobatan sehingga sistem tidak hanya merekomendasikan penyakit, tetapi juga memberikan informasi mengenai obat-obatan yang umum digunakan untuk penyakit yang direkomendasikan tersebut?
3.  Pendekatan atau algoritma sistem rekomendasi manakah yang paling sesuai untuk membangun sistem rekomendasi penyakit berdasarkan deskripsi tekstual dari gejala?

### Goals
Tujuan utama dari proyek ini adalah untuk menciptakan sebuah sistem rekomendasi yang informatif dan mudah digunakan.
1.  Mengembangkan model *content-based filtering* yang mampu menerima input berupa satu atau lebih gejala dari pengguna dan menghasilkan daftar top-N penyakit yang paling relevan berdasarkan kemiripan gejala.
2.  Mengintegrasikan dataset informasi obat untuk menampilkan daftar obat-obatan yang umum dikaitkan dengan setiap penyakit yang direkomendasikan oleh sistem.
3.  Memastikan sistem memberikan output yang mudah dipahami oleh pengguna, mencakup nama penyakit yang direkomendasikan, skor kesamaan gejala (sebagai indikator relevansi), dan informasi obat terkait.

### Solution Statements
Untuk mencapai tujuan-tujuan tersebut, solusi yang diajukan adalah sebagai berikut:
1.  **Pra-pemrosesan Data**: Melakukan pembersihan dan transformasi pada kedua dataset (gejala-penyakit dan penyakit-obat) untuk memastikan konsistensi data dan kesiapannya untuk pemodelan. Ini termasuk penanganan nilai hilang, normalisasi teks (lowercase, penghapusan spasi ekstra), dan pembuatan profil gejala unik untuk setiap penyakit.
2.  **Pemodelan Rekomendasi Penyakit**: Menerapkan teknik *content-based filtering*. Profil gejala tekstual dari setiap penyakit akan diubah menjadi representasi vektor numerik menggunakan TF-IDF (*Term Frequency-Inverse Document Frequency*). Kesamaan antara vektor gejala input pengguna dan vektor profil penyakit akan dihitung menggunakan *Cosine Similarity*.
3.  **Integrasi Informasi Obat**: Setelah mendapatkan daftar penyakit yang direkomendasikan, sistem akan melakukan pencarian (lookup) pada dataset penyakit-obat yang telah diproses untuk mengambil dan menampilkan daftar obat yang sesuai.
4.  **Evaluasi Model**: Melakukan evaluasi kualitatif dan kuantitatif sederhana (Precision@k pada sampel data) untuk menilai relevansi rekomendasi penyakit yang dihasilkan oleh model.

## Data Understanding

Proyek ini memanfaatkan dua dataset utama yang bersumber dari Kaggle:
1.  **`symtoms_df.csv`**: Dataset ini berisi informasi mengenai berbagai penyakit dan gejala-gejala yang terkait dengannya. Setiap baris data merepresentasikan suatu penyakit beserta hingga empat gejala yang sering menyertainya.
    * Jumlah baris: 4920
    * Jumlah kolom: 6
2.  **`medications.csv`**: Dataset ini berisi daftar penyakit dan obat-obatan yang umum digunakan untuk pengobatannya.
    * Jumlah baris: 41
    * Jumlah kolom: 2

Tautan sumber dataset: [Medicine Recommendation System Dataset](https://www.kaggle.com/datasets/noorsaeed/medicine-recommendation-system-dataset/data)

### Variabel-variabel pada `symtoms_df.csv`:
-   **Unnamed: 0**: Indeks atau ID baris.
-   **Disease**: Nama penyakit (target rekomendasi).
-   **Symptom_1**: Gejala pertama yang terkait dengan penyakit.
-   **Symptom_2**: Gejala kedua yang terkait dengan penyakit.
-   **Symptom_3**: Gejala ketiga yang terkait dengan penyakit.
-   **Symptom_4**: Gejala keempat yang terkait dengan penyakit (memiliki beberapa nilai NaN).

### Variabel-variabel pada `medications.csv`:
-   **Disease**: Nama penyakit.
-   **Medication**: Daftar obat yang umum digunakan untuk penyakit tersebut, dalam format string yang merepresentasikan list.

### Exploratory Data Analysis (EDA)
Analisis data eksploratif dilakukan untuk memahami karakteristik dan kualitas data:
1.  **Informasi Umum Data**: `df_symptoms.info()` dan `df_medicine.info()` digunakan untuk melihat tipe data setiap kolom dan jumlah nilai non-null.
2.  **Statistik Deskriptif**: `df_symptoms.describe(include='all')` dan `df_medicine.describe(include='all')` digunakan untuk melihat statistik dasar, termasuk jumlah nilai unik dan nilai yang paling sering muncul untuk fitur kategorikal/objek.
3.  **Identifikasi Nilai Unik**:
    -   Pada `symtoms_df.csv`, teridentifikasi 41 penyakit unik. Setelah menggabungkan semua gejala dari `Symptom_1` hingga `Symptom_4` dan melakukan pembersihan (lowercase, strip), terdapat 86 gejala unik.
    -   Pada `medications.csv`, terdapat 41 penyakit unik, yang cocok dengan jumlah penyakit unik di dataset gejala.
4.  **Pemeriksaan Konsistensi Nama Penyakit**: Dilakukan pemeriksaan (walaupun tidak eksplisit divisualisasikan di notebook awal) untuk memastikan bahwa nama-nama penyakit di kedua dataset dapat dipetakan dengan benar setelah pembersihan (misalnya, dengan mengubah semua nama menjadi huruf kecil dan menghilangkan spasi berlebih).
5.  **Format Kolom Gejala dan Obat**: Ditemukan bahwa kolom gejala pada `symtoms_df.csv` mengandung spasi di awal atau akhir string. Kolom `Medication` pada `medications.csv` adalah string yang perlu di-parsing menjadi list.

## Data Preparation
Tahapan persiapan data yang dilakukan adalah sebagai berikut, sesuai urutan implementasi pada *notebook*:

**1. Dataset Gejala-Penyakit (`df_symptoms_processed`)**:
    * **Penghapusan Kolom Indeks**: Kolom `Unnamed: 0` dihapus karena tidak memberikan informasi prediktif untuk model.
        * *Alasan*: Kolom indeks hanya penanda baris dan tidak relevan untuk analisis konten gejala.
    * **Penanganan Nilai Hilang (NaN)**: Nilai NaN pada kolom `Symptom_1`, `Symptom_2`, `Symptom_3`, dan `Symptom_4` diisi dengan string kosong (`''`).
        * *Alasan*: Ini memastikan bahwa semua sel memiliki tipe data string dan dapat diproses saat penggabungan gejala, serta mengindikasikan tidak adanya gejala tambahan yang tercatat untuk slot tersebut.
    * **Normalisasi Teks (Gejala dan Penyakit)**:
        * Semua teks pada kolom gejala (`Symptom_1` s/d `Symptom_4`) dan kolom `Disease` dibersihkan dari spasi di awal/akhir menggunakan `.str.strip()`.
        * Semua teks pada kolom gejala dan `Disease` diubah menjadi huruf kecil menggunakan `.str.lower()`.
        * *Alasan*: Standardisasi ini penting untuk konsistensi data, menghindari duplikasi akibat perbedaan minor dalam penulisan (misal, "Fever" vs " fever"), dan memastikan pencocokan yang akurat.
    * **Pembuatan Profil Gejala Agregat (`disease_profile_df`)**:
        * Dibuat daftar gejala untuk setiap baris data.
        * Dilakukan pengelompokan berdasarkan nama penyakit (`Disease`). Untuk setiap penyakit unik, semua gejala yang terkait dari berbagai baris dikumpulkan menjadi satu set gejala unik.
        * Set gejala unik ini kemudian diubah menjadi satu string tunggal (`Symptoms_Profile`), di mana setiap gejala diurutkan secara alfabetis dan dipisahkan oleh spasi.
        * *Alasan*: Ini menciptakan "dokumen" profil gejala yang komprehensif dan unik untuk setiap penyakit. Pengurutan memastikan konsistensi representasi vektor oleh TF-IDF. Dataframe baru (`disease_profile_df`) dibuat hanya dengan kolom `Disease` dan `Symptoms_Profile`.

**2. Dataset Penyakit-Obat (`df_medicine_processed`)**:
    * **Penghapusan Kolom Indeks (jika ada)**: Kolom `Unnamed: 0` dihapus dari `df_medicine` jika ada.
    * **Normalisasi Nama Penyakit**: Nama penyakit pada kolom `Disease` dibersihkan dari spasi dan diubah menjadi huruf kecil, serupa dengan yang dilakukan pada dataset gejala.
        * *Alasan*: Memastikan konsistensi dengan nama penyakit di `disease_profile_df` untuk proses *lookup* informasi obat.
    * **Penanganan Missing Values pada `Medication`**: Kolom `Medication` diisi dengan string `"['Informasi tidak tersedia']"` jika terdapat nilai NaN.
        * *Alasan*: Memberikan placeholder yang jelas jika informasi obat tidak ada.
    * **Parsing Kolom `Medication`**: Kolom `Medication` yang berisi string representasi list (misalnya, `"['Obat A', 'Obat B']"`) diubah menjadi list Python aktual menggunakan fungsi `ast.literal_eval`. Setiap nama obat dalam list juga dibersihkan dari spasi ekstra.
        * *Alasan*: Ini mengubah data string menjadi struktur list yang lebih mudah dimanipulasi dan ditampilkan.
    * **Penghapusan Duplikat Penyakit**: Baris duplikat berdasarkan kolom `Disease` dihapus, dengan hanya mempertahankan entri pertama yang muncul.
        * *Alasan*: Memastikan bahwa setiap penyakit memiliki satu daftar obat unik yang diasosiasikan dengannya dalam sistem.

Proses persiapan data ini menghasilkan dua dataframe utama yang bersih dan siap untuk tahap pemodelan: `disease_profile_df` (berisi penyakit dan profil gejalanya) dan `df_medicine_processed` (berisi penyakit dan daftar obatnya).

## Modeling and Result

Pendekatan yang digunakan untuk membangun sistem rekomendasi penyakit adalah **Content-Based Filtering**. Model ini merekomendasikan penyakit berdasarkan kemiripan konten (dalam hal ini, gejala) antara input pengguna dan profil penyakit yang ada dalam dataset.

**Tahapan Pemodelan**:
1.  **Representasi Fitur (Vektorisasi Gejala)**:
    * Untuk mengubah data tekstual gejala menjadi format numerik yang dapat diproses oleh algoritma kesamaan, digunakan **TF-IDF (Term Frequency-Inverse Document Frequency) Vectorizer** dari `sklearn.feature_extraction.text`.
    * `TfidfVectorizer` di-*fit* dan di-*transform* pada kolom `Symptoms_Profile` dari `disease_profile_df`. Proses ini menghasilkan matriks TF-IDF (`tfidf_matrix_symptoms`), di mana setiap baris merepresentasikan sebuah penyakit dan setiap kolom merepresentasikan sebuah gejala unik. Nilai dalam matriks menunjukkan bobot TF-IDF dari setiap gejala untuk setiap penyakit.
    * *Kelebihan TF-IDF*: Memberikan bobot yang lebih tinggi pada gejala yang sering muncul pada penyakit tertentu tetapi jarang muncul pada penyakit lain (menunjukkan spesifisitas), dan mengurangi bobot gejala yang sangat umum di banyak penyakit (misalnya, "demam").

2.  **Penghitungan Kesamaan (Similarity Calculation)**:
    * **Cosine Similarity** dari `sklearn.metrics.pairwise` digunakan untuk menghitung kesamaan antara vektor TF-IDF.
    * Pertama, kesamaan kosinus dihitung antar semua vektor penyakit dalam `tfidf_matrix_symptoms` (menghasilkan `cosine_sim_symptoms`), meskipun matriks ini tidak secara langsung digunakan dalam fungsi rekomendasi akhir yang membandingkan input pengguna dengan semua penyakit.
    * Yang lebih penting, saat pengguna memasukkan gejala, gejala tersebut diubah menjadi vektor TF-IDF, dan kemudian kesamaan kosinus dihitung antara vektor pengguna ini dan semua vektor penyakit dalam `tfidf_matrix_symptoms`.
    * *Kelebihan Cosine Similarity*: Mengukur kesamaan berdasarkan orientasi (sudut) antar vektor, efektif untuk data tekstual berdimensi tinggi, dan tidak terlalu dipengaruhi oleh panjang "dokumen" gejala.

3.  **Pengembangan Fungsi Rekomendasi**:
    * Fungsi `recommend_diseases_and_meds` dikembangkan untuk menerima input string gejala dari pengguna (dipisahkan koma), jumlah top-N rekomendasi yang diinginkan, serta data profil penyakit, data obat, vectorizer TF-IDF, dan matriks TF-IDF gejala.
    * **Input Pengguna**: Gejala input dari pengguna dibersihkan (spasi, lowercase) dan diubah menjadi format string yang diurutkan, serupa dengan `Symptoms_Profile`.
    * **Transformasi Input**: Gejala pengguna yang telah diproses diubah menjadi vektor TF-IDF menggunakan `tfidf_vectorizer.transform()`.
    * **Skoring Kesamaan**: Kesamaan kosinus dihitung antara vektor TF-IDF pengguna dan semua vektor TF-IDF penyakit.
    * **Pengurutan dan Seleksi Top-N**: Penyakit diurutkan berdasarkan skor kesamaan tertinggi. Top-N penyakit dengan skor kesamaan lebih besar dari 0 dipilih.
    * **Integrasi Informasi Obat**: Untuk setiap penyakit yang direkomendasikan, fungsi `get_medications_for_disease_lookup` mencari daftar obat terkait dari `df_medicine_processed` yang telah dipersiapkan.

**Hasil (Result)**:
Sistem menghasilkan daftar top-N penyakit yang paling sesuai dengan gejala input pengguna. Untuk setiap penyakit yang direkomendasikan, ditampilkan skor kesamaan gejala dan daftar obat yang umumnya digunakan.

*Contoh Output untuk input gejala "itching, skin_rash, nodal_skin_eruptions" (Top-3):*

| No. | Penyakit (Skor Kesamaan Gejala)           | Obat Umum                                                                                                        |
|-----|------------------------------------------|-------------------------------------------------------------------------------------------------------|
| 1.  | Fungal Infection (0.70)                  | Antifungal Cream, Fluconazole, Terbinafine, Clotrimazole, Ketoconazole                                |
| 2.  | Chicken Pox (0.46)                       | Antiviral drugs, Pain relievers, IV fluids, Blood transfusions, Platelet transfusions                  |
| 3.  | Drug Reaction (0.31)                     | Antihistamines, Epinephrine, Corticosteroids, Antibiotics, Antifungal Cream                           |



Sistem ini mampu memberikan rekomendasi yang relevan berdasarkan input gejala. Jika tidak ada gejala yang cocok atau input kosong, sistem akan memberikan pesan yang sesuai.

## Evaluation

Evaluasi sistem rekomendasi berbasis konten ini, terutama tanpa dataset *ground truth* eksternal yang terverifikasi (misalnya, diagnosis medis aktual untuk serangkaian gejala yang dilaporkan pengguna), dilakukan secara kualitatif dan dengan metrik kuantitatif sederhana pada sampel data.

Metrik evaluasi utama yang digunakan adalah **Precision@k**.
* **Formula Precision@k**:
    $$\text{Precision@k} = \frac{\text{Jumlah rekomendasi relevan dalam k rekomendasi teratas}}{k}$$
* **Cara Kerja Metrik dalam Konteks Ini**:
    Metrik ini bertujuan untuk mengukur seberapa akurat rekomendasi yang diberikan oleh sistem dalam daftar *k* teratas. Untuk simulasi evaluasi ini:
    1.  Sebuah sampel acak (10 penyakit) diambil dari `df_symptoms_processed` (setelah penghapusan duplikat penyakit untuk memastikan setiap penyakit diuji sekali sebagai *ground truth*).
    2.  Untuk setiap penyakit dalam sampel, hingga 3 gejala non-kosong diambil dari data aslinya untuk dijadikan input ke sistem rekomendasi.
    3.  Sistem kemudian menghasilkan daftar top-k penyakit yang direkomendasikan (dalam kasus ini, k=1 dan k=3).
    4.  Sebuah rekomendasi dianggap *relevan* atau *hit* jika penyakit asli dari sampel uji muncul dalam daftar top-k penyakit yang direkomendasikan.
    5.  Precision@k dihitung sebagai total *hits* dibagi dengan jumlah total sampel uji (yaitu, 10).

**Hasil Proyek Berdasarkan Metrik Evaluasi (pada 10 sampel acak)**:
* **Precision@1**: 0.80
    * Ini berarti bahwa untuk 80% dari kasus uji dalam sampel, penyakit yang benar (penyakit asli dari sampel) adalah rekomendasi pertama yang diberikan oleh sistem.
* **Precision@3**: 1.00
    * Ini berarti bahwa untuk semua (100%) kasus uji dalam sampel, penyakit yang benar muncul dalam 3 rekomendasi teratas yang diberikan oleh sistem.

**Interpretasi dan Keterbatasan**:
* Hasil Precision@k yang tinggi (terutama Precision@3 = 1.00 pada sampel ini) menunjukkan bahwa sistem cukup baik dalam mengidentifikasi penyakit yang benar berdasarkan sebagian gejalanya, setidaknya untuk data yang ada dalam datasetnya.
* **Evaluasi Kualitatif**: Berdasarkan contoh-contoh penggunaan yang dijalankan (*notebook* sel 72), rekomendasi penyakit yang diberikan tampak relevan dengan gejala input. Misalnya, untuk gejala "itching, skin_rash, nodal_skin_eruptions", penyakit "Fungal Infection" direkomendasikan dengan skor tertinggi, yang mana sesuai dengan data. Informasi obat yang ditampilkan juga sesuai dengan yang terdaftar untuk penyakit tersebut di `df_medicine.csv`.
* **Keterbatasan**:
    * Evaluasi kuantitatif ini dilakukan pada sampel data yang sangat kecil (10) dan menggunakan gejala dari dataset yang sama.
    * Tidak ada validasi dari ahli medis.
    * Kualitas rekomendasi obat sangat bergantung pada kelengkapan dan keakuratan dataset `medicine.csv`.

Meskipun ada keterbatasan, evaluasi awal ini memberikan indikasi bahwa pendekatan *content-based filtering* dengan TF-IDF dan Cosine Similarity cukup efektif untuk tugas rekomendasi penyakit berdasarkan gejala pada dataset yang digunakan. Untuk implementasi di dunia nyata, validasi lebih lanjut dengan dataset uji yang lebih besar, beragam, dan terverifikasi secara medis sangat diperlukan.

---
**Akhir Laporan**
