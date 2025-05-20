# Laporan Proyek Machine Learning - Nurtawinda

## Domain Proyek

Tempat tinggal menjadi salah satu kebutuhan primer untuk setiap individu (Ridho et al., 2022). Pembelian tempat tinggal atau rumah merupakan suatu keputusan besar yang memerlukan banyak pertimbangan. Harga rumah dapat dipengaruhi berbagai factor, mulai dari kondisi lokasi, fasilitas sekitar, dan karakteristik dari rumah itu sendiri(Hidayah et al., n.d.). Selain itu, harga rumah juga dapat dipengaruhi dari variabel lain yang perlu dipertimbangkan seperti alamat dan lokasi rumah (Cahyani Putri & Arianto, 2024). Penentuan harga rumah baik untuk penjual maupun pembeli harus dipertimbangkan dengan baik (Mu’tashim et al., 2021).

Maka dari itu, diperlukan sebuah metode yang mampu membantu dalam menentukan harga rumah yang sesuai. Sistem prediksi harga rumah bisa menjadi salah satu solusi untuk membantu dan mempermudah proses pengambilan keputusan untuk membeli rumah (Ridho et al., 2022). Saat ini berbagai metode Machine Learning telah banyak digunakan diberbagai bidang termasuk prediksi harga rumah, diantaranya regresi linear, support vector regression, decision tree, dan random forest. Setiap metode tentunya memiliki keunggulan dan kekurangan masing-masing tergantung dari jenis data yang digunakan dan kompleksitas masalah yang akan diselesaikan. (Hidayah et al., n.d.; Ridho et al., 2022).

Pada penelitian-penelitian sebelumnya, beberapa teknik analisis data digunakan untuk meningkatkan akurasi prediksi harga rumah. Salah satunya adalah Information Gain, yang digunakan untuk memilih fitur terbaik dalam model prediksi harga rumah (Putri & Arianto, 2024). Metode ini membantu dalam mengidentifikasi informasi yang paling relevan untuk memprediksi harga rumah secara lebih efektif.

Penelitian ini bertujuan untuk mengembangkan model prediksi harga rumah dengan menggunakan berbagai teknik machine learning dan analisis data untuk meningkatkan akurasi prediksi. Dengan demikian, model prediksi yang dihasilkan diharapkan dapat membantu konsumen dalam membuat keputusan yang lebih tepat dalam membeli rumah, sesuai dengan kebutuhan dan anggaran yang dimiliki.

Secara keseluruhan, prediksi harga rumah menggunakan metode machine learning menawarkan potensi besar untuk memperbaiki proses pengambilan keputusan di pasar properti. Penelitian ini akan terus mengembangkan dan menguji metode-metode ini untuk menciptakan solusi yang lebih akurat dan dapat diandalkan dalam memprediksi harga rumah di masa depan.

Referensi:

Putri, N. A. C., & Arianto, D. B. (2024). Komparasi Penggunaan Information Gain Pada Machine Learning untuk Memprediksi Harga Rumah di Jabodetabek. Jurnal Sains dan Teknologi, 5(3), 756-762.

Hidayah, F., Angesti, S. J., & Widyastuti, Y. P. Prediksi Harga Rumah di Boston Menggunakan Metode Linear Regression, SVR, Decision Tree dan Random Forest Regression. ()

Mu'tashim, M. L., Muhayat, T., Damayanti, S. A., Zaki, H. N., & Wirawan, R. (2021). Analisis prediksi harga rumah sesuai spesifikasi menggunakan multiple linear regression. Informatik: Jurnal Ilmu Komputer, 17(3), 238-245.

Ridho, I. I., Mahalisa, G., Sari, D. R., & Fikri, I. (2022). Metode Neural Network Untuk Penentuan Akurasi Prediksi Harga Rumah. Technologia: Jurnal Ilmiah, 13(1), 56-58.


## Business Understanding

Rumah merupakan salah satu kebutuhan primer yang setiap tahun harganya mengalami kenaikan. Terdapat banyak faktor yang mempengaruhi kenaikan harga tersebut. Tanpa pemahaman yang baik mengenai faktor tersebut, pembeli maupun penjual rumah beresiko mendapatkan harga rumah yang tidak optimal. Maka dari itu, diperlukan sistem prediksi harga rumah yang dapat membantu pembeli maupun penjual dalam menentukan harga rumah yang lebih tepat.

### Problem Statements

- Faktor apa saja yang memiliki pengaruh besar terhadap harga rumah?
- Model regresi mana yang paling efektif dalam memprediksi harga rumah dengan mempertimbangkan faktor-faktor yang ada?

### Goals

Menjelaskan tujuan dari pernyataan masalah:
- Mengidentifikasi faktor yang memiliki pengaruh besar terhadap harga rumah.
- Mengidentifikasi model regresi yang paling efektif dalam memprediksi harga rumah dengan mempertimbangkan faktor-faktor yang ada.

### Solution statements
- Menyusun model regresi untuk mengetahui interaksi antara faktor-faktor dan dampaknya terhadap penentuan harga rumah.
- Membangun beberapa model regresi seperti K-Nearest Neighbor, Random Forest Regression, dan Gradient Boosting regression.
- Melakukan evalusi menggunakan metrik evaluasi Mean Absolute Error (MAE) untuk melihat, membandingkan, dan menentukan performa paling baik.

## Data Understanding
Data yang digunakan pada proyek ini adalah [House Price Prediction Dataset](https://www.kaggle.com/datasets/zafarali27/house-price-prediction-dataset) yang diperoleh dari Kaggle. Dataset yang digunakan terdiri dari 10 kolom dan 2000 baris.

### Variabel-variabel pada House Price Prediction Dataset adalah sebagai berikut:
- Id: Berisi identifikasi unik untuk setiap entitas secara individual.
- Area (Luas): Luas rumah dalam satuan kaki persegi, yang umumnya merupakan salah satu faktor terpenting dalam menentukan harga.
- Bedrooms & Bathrooms (Kamar Tidur & Kamar Mandi): Jumlah kamar dalam sebuah rumah secara signifikan memengaruhi nilainya. Rumah dengan lebih banyak kamar cenderung dihargai lebih tinggi.
- Floors (Lantai): Jumlah lantai dalam sebuah rumah dapat menandakan rumah yang lebih besar dan mewah, yang berpotensi meningkatkan harganya.
- Year Built (Tahun Pembangunan): Usia rumah dapat memengaruhi kondisinya dan nilainya. Rumah yang baru dibangun umumnya lebih mahal daripada yang lebih tua.
- Location (Lokasi): Rumah di lokasi yang diminati seperti pusat kota atau kawasan perkotaan cenderung dihargai lebih tinggi daripada yang berada di kawasan pinggiran atau pedesaan.
- Condition (Kondisi): Kondisi saat ini dari rumah sangat penting, karena rumah yang terawat baik (dalam kondisi 'Sangat Baik' atau 'Baik') akan menarik harga lebih tinggi dibandingkan rumah dalam kondisi 'Cukup' atau 'Buruk'.
- Garage (Garasi): Ketersediaan garasi dapat meningkatkan harga karena menambah kenyamanan dan ruang.
- Price (Harga): Variabel target, mewakili harga jual rumah, digunakan untuk melatih model pembelajaran mesin dalam memprediksi harga rumah berdasarkan fitur-fitur lainnya.

Dari 10 kolom tersebut, kolom ID tidak digunakan karena tidak memiliki pengaruh untuk sistem prediksi yang akan dibangun.

**Explanatory Data Analysis - Univariate Analysis**:
Berikut beberapa tahapan yang digunakan pada tahap ini:

1. Melakukan pengecekan terhadap data yang outlier pada feature-feature numerik.

- ![Plot Persentase Area](https://github.com/Nurtawinda/House-Price-Prediction/blob/main/Plot%20outliers/Area.png?raw=true)

- ![Plot Outlier Bathrooms](https://github.com/Nurtawinda/House-Price-Prediction/blob/main/Plot%20outliers/Bathrooms.png?raw=true)

- ![Plot Outlier Bedrooms](https://github.com/Nurtawinda/House-Price-Prediction/blob/main/Plot%20outliers/Bedrooms.png?raw=true)

- ![Plot Outlier Floors](https://github.com/Nurtawinda/House-Price-Prediction/blob/main/Plot%20outliers/Floors.png?raw=true)

- ![Plot Outlier Price](https://github.com/Nurtawinda/House-Price-Prediction/blob/main/Plot%20outliers/Price.png?raw=true)

- ![Plot Outlier Year Built](https://github.com/Nurtawinda/House-Price-Prediction/blob/main/Plot%20outliers/YearBuilt.png?raw=true)

Plot diatas memperlihatkan bahwa tidak terdapat outlier pada data yang digunakan.

2. Melihat presentase Fitur
- ![Plot Persentase Condition](https://github.com/Nurtawinda/House-Price-Prediction/blob/main/Persentase%20feature/Condition.png?raw=true)

- ![Plot Persentase Garage](https://github.com/Nurtawinda/House-Price-Prediction/blob/main/Persentase%20feature/Garage.png?raw=true)

- ![Plot Persentase Location](https://github.com/Nurtawinda/House-Price-Prediction/blob/main/Persentase%20feature/Location.png?raw=true)

Terdapat 3 fitur kategorikal yaitu:
1. Condition (Kondisi). Pada plot *Condition* terdapat 4 kategori yaitu *Excellent, Fair, Good,* dan *Poor*. Dari 4 kategori tersebut, rumah paling banyak berada pada kondisi Fair (cukup), dilanjutkan dengan excellent (Sangat baik), poor (Buruk), dan terakhir adalah good (baik).
2. Garage (garasi). Pada plot yang ditampilkan, terdapat lebih dari 1000 rumah tidak memiliki garasi atau dengan kata lain rumah yang memiliki garasi lebih sedikit dibandingkan dengan rumah yang tidak memiliki garasi.
3. Location (Lokasi). Berdasarkan plot yang ditampilkan, terdapat empat kategori lokasi yaitu *Downtown, Urban, Suburban* dan *Rural*. Dari 4 lokasi tersebut, rumah paling banyak berada di Kota Downtown, kemudian Urban, Suburban dan terakhir adalah Rural.

- ![Plot Persentase Fitur Numerik ](https://github.com/Nurtawinda/House-Price-Prediction/blob/main/Persentase%20feature/fitur%20numerik.png?raw=true)

Terdapat 6 fitur numerik yaitu:
1. Area (Area). 
2. Bedrooms (Kamar Tidur).
3. Bathrooms (Kamar Mandi).
4. Floors (Lantai).
5. Year Built (Tahun Dibangun).
6. Price (Harga).

Dari plot diatas kita dapar melihat bagaimana penyebaran data khususnya data Numerik. Plot area menunjukkan penyebaran luas area rumah, dimana pada plot tersebut 5000 meter persegi merupakan area rumah paling luas. Selanjutnya Bedrooms dan Bathrooms, melalui plot yang tersaji kita dapat melihat dan mengetahui jumlah rumah untuk masing-masing jumlah kamar. Pada plot Floors sendiri mengacu pada jumlah tingkat rumah, dari plot tersebut diketahui bahwa pada data yang digunakan rumah bertingkat 2 memiliki jumah yang lebih banyak dibandingkan dengan rumah bertingkat 1 dan 3. Kemudian ada plot Year Built yang menunjukkan tahun rumah dibangun dan dapat diketahui bahwa rumah-rumah tersebut dibangun pada rentang tahun 1900 sampai sekitar tahun 2020an. Terakhir ada plot Price yang menunjukkan rentang harga rumah.

**Explanatory Data Analysis - Multivariate Analysis**:
Tahap ini dilakukan untuk melihat rata-rata antar fitur
1. Rata-rata 'Area' Relatife terhadap - Location
![Rata-rata 'Price' Relatife terhadap - Location](https://github.com/Nurtawinda/House-Price-Prediction/blob/main/Rata-rata%20antar%20fitur/1.png?raw=true) Pada plot tersebut dapat diketahui bahwa harga rumah di 4 lokasi memiliki harga yang hampir sama yaitu diatas 500000, yang artinya tiap lokasi memiliki pengaruh yang sama terhadap harga.
2. Rata-rata 'Price' Relatife terhadap - Condition
![Rata-rata 'Price' Relatife terhadap - Condition](https://github.com/Nurtawinda/House-Price-Prediction/blob/main/Rata-rata%20antar%20fitur/2.png?raw=true) Pada plot ini dapat diketahui bahwa tiap kondisi memiliki rata-rata yang hampir sama satu sama lain. Sehingga tiap Kondisi memiliki pengaruh yang sama terhadap harga.
3. Rata-rata 'Price' Relatife terhadap - Garage
![Rata-rata 'Price' Relatife terhadap - Garage](https://github.com/Nurtawinda/House-Price-Prediction/blob/main/Rata-rata%20antar%20fitur/3.png?raw=true) Pada plot yang ditampilkan, dapat diketahu bahwa rata-rata harga rumah pada dua kondisi tersebut memiliki rata-rata yang sama. Sehingga dapat dikatakan bahwa rumah dengan garasi dan tanpa garasi memiliki pengaruh yang sama terhadap harga.
4. Hubungan antar fitur numerik
![hubungan antar fitur numerik](https://github.com/Nurtawinda/House-Price-Prediction/blob/main/Hubungan%20antar%20fitur%20numerik.png?raw=true)
- Terdapat indikasi korelasi positif antara Area dengan Price, Bedrooms, dan Bathrooms. Demikian pula, Bedrooms dan Bathrooms berkorelasi positif satu sama lain dan dengan Price. Ini berarti, secara umum, rumah dengan area yang lebih besar, lebih banyak kamar tidur dan mandi cenderung memiliki harga yang lebih tinggi.
- Hubungan antara Floors dan YearBuilt dengan fitur lainnya tidak menunjukkan pola linear yang kuat secara visual. Ini mengindikasikan bahwa jumlah lantai dan tahun pembangunan mungkin tidak menjadi faktor penentu harga secara langsung atau memiliki hubungan yang lebih kompleks dengan fitur lainnya.
- Distribusi setiap fitur bervariasi. Area dan Price menunjukkan skewness positif (ekor panjang ke kanan). Bedrooms, Bathrooms, dan Floors cenderung memiliki distribusi diskrit dengan beberapa puncak. YearBuilt tersebar di berbagai rentang tahun.

Dari Explanatory data yang dilakukan, semua fitur memiliki kolerasi dalam penentuan harga, namun fitur yang memiliki korelasi besar ialah jumlah kamar tidur, kamar mandi, jumlah lantai, kemudian luas area dan tahun dibangunnya rumah.

## Data Preparation
Berikut beberapa tahapan yang dilakukan pada bagian data Preparation.
- Data menunjukkan tidak adanya outlier, nilai kosong, dan data yang terduplikat. Maka dari itu dataset yang digunakan dapat dikatakan memiliki data yang bersih.

- Encoding fitur kategori. Pada tahap ini, fitur kategori akan diubah kedalam format biner menggunakan OneHotEncoder. Setelah melakukan tahap Encoding, setiap kategori pada fitur akan diubah menjadi kolom yang bernilai True dan False atau 1 atau 0. Melalui tahap ini, algoritma Machine Learning akan dengan mudah memproses data karena telah bernilai numerik.

- Reduksi dimensi dengan Principal Component Analysis (PCA). Selanjutnya adalah melakukan tahap PCA yang bertujuan untuk mengetahu proporsi informasi dari fitur yang linear. Dari data yng digunakan, fitur yang memiliki korelasi linear paling tinggi yangitu Bedrooms, Bathrooms, dan Floors. 

- Pembagian dataset. Tahap ini dilakukan dengan menggunakan fungsi train_test_split dari library sklearn. Proporsi pembagian data latih dan uji adalah 90:10. Sehingga data latih berjumlah 1800 data dan 200 sisanya merupakan data uji.

- Standarisasi. Tahap ini dilakukan untuk membantu fitur data menjadi lebih mudah untuk diolah oleh algoritma. Setelah mdistandarisasi maka data akan berada pada rentang antara -1 dan 1. Proses standarisasi mengubah nilai rata-rata (mean) menjadi 0 dan nilai standar deviasi menjadi 1.

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

Terdapat 3 model yang digunakan pada proyek ini, yaitu KNN (K-Nearest Neighbor), Random Forest, dan Boosting Gradien.

Model KNN digunakan dengan parameter n_neighbors=10 dan parameter lainnya bernilai default. KNN sangat sensitif terhadap skala fitur dan kurang efisien pada dataset besar. Namun, KNN merupakan model sederhana dan tidak membutuhkan proses pelatihan yang kompleks. 

Model Random Forest digunakan dengan parameter n_estimators=50, max_depth=16, random_state=55, n_jobs=-1 dan lainnya bernilai default. Tiap tree akan dilatih secara acak dan hasil akar akan diambil dari rata-rata tree. Random Forest efektif untuk dataset besar dengan banyak fitur dan cenderung robust terhadap outlier serta tidak memerlukan banyak tuning parameter. Namun, model ini sulit diinterpretasikan dibandingkan model linear atau decision tree tunggal dan dapat menjadi lambat untuk dataset yang sangat besar atau real-time prediction.

Model Boosting Algorithm digunakan dengan parameter learning_rate=0.05, random_state=55 dan untuk parameter lainnya bernilai default. Boosting Algorithm cenderung menghasilkan model yang sangat akurat dengan menggabungkan beberapa model lemah secara adaptif. Tetapi model Boosting rentan terhadap overfitting jika data terlalu kompleks atau jumlah iterasi terlalu banyak, dan proses pelatihannya bisa lebih lama dibandingkan algoritma lain. 

## Evaluation
Metrik evaluasi yang digunakan pada proyek ini adalah MSE atau *Mean Squared Error*. MSE cocok digunakan untuk kasus-kasus regresi dimana MSE mengukur rata-rata kuadrat perbedaan antara nilai yang diprediksi dan nilai yang sebenarnya dalam suatu model. Semakin rendah nilai MSE, semakin baik performa model, karena menunjukkan bahwa prediksi model lebih dekat dengan nilai aktual.

berikut adalah nilai dari MSE masing-masing model:
| Model        | Train MSE | Test MSE |
|--------------|-----------|----------|
| KNN          | 68428494.059473 | 78608018.195175 |
| RandomForest | 17205492.559097 | 82208912.479786 |
| Boosting     | 75091854.999492 | 75483398.289809 |

Dari nilai tersebut, didapatkan plot sebagai berikut:

![Plot MSE](https://github.com/Nurtawinda/House-Price-Prediction/blob/main/Plot%20MSE.png?raw=true)

Dari Plot MSE tersebut dapat diketahui:
- Untuk Boosting, nilai MSE pada data train dan test relatif lebih dekat dibandingkan dua algoritma lainnya, menunjukkan generalisasi yang lebih baik meskipun nilainya secara keseluruhan cukup tinggi.
- Untuk KNN (K-Nearest Neighbors), MSE pada data test sedikit lebih tinggi daripada pada data train, menunjukkan potensi generalisasi yang kurang baik dibandingkan Boosting.
- Untuk RF (Random Forest), MSE pada data train jauh lebih rendah daripada pada data test, mengindikasikan overfitting yang signifikan.

Setelah mengetahui nilai MSE pada pelatihan dan pengujian, maka selanjutnya dilakukan evaluasi lanjutan yaitu evaluasi prediksi. Pada evaluasi ini, nilai aktual (y_true) akan dibandingkan dengan hasil prediksi model.

| y_true | prediksi_KNN | prediksi_RF | prediksi_Boosting |
|--------|--------------|-------------|-------------------|
| 481613 | 518565.6     | 600042.9    | 499871.8          |
| 419470 | 346496.9     | 329385.6    | 491057.2          |
| 395590 | 479941.7     | 577325.3    | 531676.9          |
| 443781 | 586938.5     | 495399.0    | 547519.1          |
| 352468 | 554449.5     | 543529.4    | 540519.4          |

Kesimpulan:
model Boosting menunjukkan kinerja yang paling baik dalam hal generalisasi dibandingkan KNN dan RandomForest. Meskipun nilai MSE Boosting pada data train tidak serendah RandomForest, perbedaan antara MSE pada data train dan test untuk Boosting jauh lebih kecil. Ini mengindikasikan bahwa model Boosting mampu mempertahankan performanya dengan lebih baik ketika dihadapkan pada data yang belum pernah dilihat sebelumnya (data test).
