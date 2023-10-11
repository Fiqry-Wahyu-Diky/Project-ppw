import streamlit as st
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import altair as alt
from streamlit_option_menu import option_menu
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import requests
from bs4 import BeautifulSoup
import csv
import matplotlib.pyplot as plt
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import nltk

# binary
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer

showWarningOnDirectExecution = False

with st.sidebar:
    selected = option_menu(
        menu_title="MENU",
        options=["HOME", "PROJECT"],
    )
# ====================== Home ====================
if selected == "HOME":
    st.markdown('<h1 style = "text-align: center;"> Text Processing </h1>', unsafe_allow_html=True)
    # gambar = Image.open("nlp.jpg")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write('')
    with col2:
        st.image("nlp.jpg", width=250)
    with col3:
        st.write(' ')

    st.write(' ')
    st.markdown(
        '<p style = "text-align: justify;"> <b> Text processing </b> adalah proses manipulasi atau analisis teks menggunakan komputer atau alat otomatis lainnya. Tujuannya dapat bervariasi, termasuk pengolahan teks untuk ekstraksi informasi, pemrosesan bahasa alami, pengenalan pola teks, pemfilteran teks, atau tugas lainnya yang melibatkan teks. Ada beberapa tahapan dalam text processing sebagai berikut:</p>',
        unsafe_allow_html=True)
    st.write('- Crawling Data <br> <p style = "text-align: justify;"><b>Crawling</b> merupakan alat otomatis yang mengumpulkan beragam informasi dengan menjelajahi berbagai halaman web. Proses ini mencakup identifikasi serta ekstraksi elemen-elemen seperti teks, gambar, dan unsur lainnya, sehingga membentuk pemahaman menyeluruh tentang konten yang tersebar di internet.</p>',unsafe_allow_html=True)
    st.write(
        '- Normalisasi Text <br> <p style = "text-align: justify;"> <b>Normalisasi Text</b> merupakan suatu proses mengubah data teks menjadi bentuk standar, sehingga dapat digunakan dalam pengolahan lebih lanjut.</p>',
        unsafe_allow_html=True)
    st.write(
        '- Reduksi Dimensi <br> <p style = "text-align: justify;"> <b>Reduksi dimensi Text</b> adalah proses mengurangi jumlah atribut (fitur) dalam suatu dataset dengan tetap mempertahankan informasi yang signifikan. Tujuan utama dari reduksi dimensi adalah untuk mengatasi masalah "kutukan dimensi" (curse of dimensionality), di mana dataset dengan banyak fitur dapat mengakibatkan masalah komputasi yang mahal dan pemodelan yang kurang akurat. Reduksi dimensi juga dapat membantu dalam memahami struktur data, menghilangkan atribut yang tidak relevan, dan memungkinkan visualisasi yang lebih baik dari data yang kompleks.</p>',
        unsafe_allow_html=True)

# ====================== Project ====================
else:
    st.markdown('<h1 style = "text-align: center;">Text Processing</h1>', unsafe_allow_html=True)
    st.write("Oleh | FIQRY WAHYU DIKY W | 200411100125")
    crawling, preprocessing, lda, modelling = st.tabs(
["Crawling", "Preprocessing", "Reduksi LDA", "Modeling"])

# ====================== Crawling ====================
    with crawling:
        dataset, keterangan = st.tabs(["Link data", "Keterangan"])
        with dataset:
            st.warning("##### Sebelum melakukan crawling, disarankan melihat keterangan")
            # Membuat variabel place dan number untuk kata kunci tujuan
            place = st.text_input("Masukkan kata kunci tujuan:")
            number = st.number_input("Masukkan key number link:",step=1)
            k = st.number_input("Masukkan banyak pages:",step=1)
            # Menyusun URL berdasarkan variabel place dan number
            url = f'https://pta.trunojoyo.ac.id/c_search/{place}/{number}/'

            # Tombol untuk mulai crawling
            if st.button("Mulai Crawling"):
                datas = []

                for ipages in range(1, k+1):
                    response = requests.get(url + str(ipages))
                    soup = BeautifulSoup(response.text, 'html.parser')
                    pages = soup.findAll('li', {'data-id': 'id-1'})

                    if not pages:
                        break  # Menghentikan iterasi jika tidak ada lagi data yang ditemukan

                    for items in pages:
                        # Mengambil abstrak dari teks di dalam tag <p> dengan atribut 'align' = 'justify'
                        # ==== mencari abstrak ====
                        button_abstrak_pages = items.find('a', 'gray button').get(
                            'href')  # setiap iterasi list pages mencari <a> ambil link href
                        response_abstrak_pages = requests.get(
                            button_abstrak_pages)  # meminta HTTP dari setiap link yang ada di variabel button_abstrak_pages

                        soup_abstrak = BeautifulSoup(response_abstrak_pages.text,
                                                     'html.parser')  # Isi teks dari respons HTTP yang diterima dari server web
                        abstrak = soup_abstrak.find('p', {'align': 'justify'}).text.replace('ABSTRAK', '')

                        datas.append({
                            'Judul': items.find('a', 'title').text,
                            'Nama Penulis': items.find_all('span')[0].text.replace('Penulis :', ''),
                            'Pembimbing I': items.find_all('span')[1].text.replace('Dosen Pembimbing I :', ''),
                            'Pembimbing II': items.find_all('span')[2].text.replace('Dosen Pembimbing II :', ''),
                            'Abstrak': abstrak
                        })

# ================== Menampilkan hasil crawling ====================
                if datas:
                    df_datas = pd.DataFrame(datas)
                    st.success("Data telah selesai dicrawling")
                    st.warning("Hasil Crawling:")
                    csv_file = df_datas.to_csv(index=False)
                    csv_bytes = csv_file.encode()
                    st.download_button(
                        label="Unduh Data sebagai CSV",
                        data=csv_bytes,
                        file_name="data_crawling.csv",
                        mime="text/csv",
                    )
                    st.table(df_datas)

                else:
                    st.warning("Tidak ada Hasil Crawling")

# ================== Tab keterangan ====================
        with keterangan:
            columns = ["Prodi","Kata Kunci", "Number Link"]
            data = [
                ["Hukum","byprod", 1],
                ["Magister Ilmu Hukum", "byprod", 24],

                ["Teknologi Industri Pertanian", "byprod", 2],
                ["Agribisnis", "byprod", 3],
                ["Agroteknologi", "byprod", 4],
                ["Ilmu Kelautan", "byprod", 5],

                ["Teknik Informatika", "byprod", 10],
                ["Sistem Informasi", "byprod", 31],
                ["Teknik Industri", "byprod", 9],

            ]
            df_link = pd.DataFrame(data=data, columns=columns)

            st.write("Berikut keterangan dalam Portal Artikel Tugas Akhir :")
            st.info("#### Fakultas Hukum:")
            st.table(df_link.iloc[0:2])

            st.info("#### Fakultas Pertanian:")
            st.table(df_link.iloc[2:6])

            st.info("#### Fakultas Teknik:")
            st.table(df_link.iloc[6:9])


# ======================= Preprocessing =================================
    with preprocessing:
        st.write("# Normalisasi")
        data, cleaned, vsm= st.tabs(
            ["data", "Clean data", "VSM"])

        # Tombol untuk mengunduh data
        with data:
            st.write(
                "Sebelum melakukan modelling data harus diprocessing terdapat Normalisasi Text merupakan suatu proses mengubah data teks menjadi bentuk standar, sehingga dapat digunakan dalam pengolahan lebih lanjut. <br> Data sudah disajikan dan diberi label mohon undah data di bawah ini",
                unsafe_allow_html=True)
            # Mendefinisikan URL data dari Google Drive
            st.write("##### Data")
            data = pd.read_csv("data_crawling_pta_labels.csv")
            # jumlah_data = (data['humidity']).count()
            # st.success(jumlah_data)
            banyak_data = len(data)
            st.success(f"#### Banyak data yang digunakan sejumlah : {banyak_data}")
            st.write(data)

            if st.button("cek data kosong"):
                data_kosong = data.isna().sum()
                st.write(data_kosong)
                st.warning("###### terdapat data kosong pada masing-masing kolom, harus clean data!!!")

            # if st.button('Unduh Data'):
            #     st.text('Sedang mengunduh data...')
            #
            #     # Mengunduh data dari Google Drive
            #     output = gdown.download(url, quiet=False)
            #
            #     st.text('Data telah berhasil diunduh!')
            #
            #     # Membaca data dari file yang diunduh dan menyimpannya dalam DataFrame
            #     data = pd.read_csv(output)
            #     # Menampilkan data
            #     st.header('Data yang Telah Diunduh')
            #     st.write(data)
            #
            # #     ============ download
            #     data_label = pd.DataFrame(data)
            #     csv_file = data_label.to_csv(index=False)
            #     csv_bytes = csv_file.encode()
            #     st.download_button(
            #         label="Unduh Data sebagai CSV",
            #         data=csv_bytes,
            #         file_name="data_crawling_labels.csv",
            #         mime="text/csv",
            #     )

        with cleaned:
        # Tombol untuk menghapus data NaN dari kolom "Abstrak"
            st.info("#### Data sudah dibersihkan")
            # Menghapus semua kolom yang memiliki setidaknya satu nilai NaN
            data.dropna(subset=["Abstrak","Dosen Pembimbing II"], inplace=True)
            # st.write(data.isna().sum())

            if st.button("Cek data"):
                # Menampilkan hasil periksa data kosong atau null dalam bentuk tabel
                st.write("Jumlah Data Kosong atau Null dalam Setiap Kolom:")
                data_kosong = data.isna().sum()
                st.write(data_kosong)

 # ======== cleanned ===================
# ============ Punctuation ===========
            st.info("#### Punctuation")
            data['clean_abstrak'] = data['Abstrak'].str.replace(r'[^\w\s]', '', regex=True).str.lower()
            # membersihkan angka
            data['clean_abstrak'] = data['clean_abstrak'].str.replace('\d+', '', regex=True)
            # st.write(string.punctuation)
            st.write(data)

# =========== stop word ========================
            st.info("#### Stop word")
            nltk.download('punkt')
            # Download kamus stop words
            nltk.download('stopwords')
            # Inisialisasi kamus stop words dari NLTK
            stop_words = set(stopwords.words('indonesian'))  # Inisialisasi kamus stop words
            # Menghapus stop words dari kolom 'Abstrak'
            for stop_word in stop_words:
                data['abstrak_stopword'] = data['clean_abstrak'].str.replace(rf'\b{stop_word}\b', '',
                                                                             regex=True)  # rf untuk formating string
            st.write(data)

            # ====== tokenizing
            st.info("#### Tokenizing")
            data['tokens'] = data['abstrak_stopword'].apply(word_tokenize)
            st.write(data)

# ================ stemming ========================
            st.warning("Apakah ingin melakukan stemming? Jika iya maka centang")
            st.write("mungkin memebutuhkan waktu beberapa menit!!!")
            stemming_ck = st.checkbox("Stemming data")
            if stemming_ck:
                st.info("#### Stemming")

                factory = StemmerFactory()
                stemmer = factory.create_stemmer()

                # Membuat kolom baru untuk hasil stemming
                data['stemmed_tokens'] = None

                # Melakukan stemming pada setiap elemen dalam kolom 'tokens' tanpa fungsi atau list comprehension
                for index, row in data.iterrows():  # perulangan indeks, row dari data
                    stemmed_tokens = []
                    for word in row['tokens']:
                        # print(word)
                        stemmed_word = stemmer.stem(word)
                        # print(stemmed_word)
                        stemmed_tokens.append(stemmed_word)
                    # print(stemmed_tokens)
                    data.at[index, 'stemmed_tokens'] = stemmed_tokens
                st.info("#### Stemming sudah dilakukan")

# =============== merge ==================
            # menggabungkan kata
            st.info("#### Menggabungkan kata")
            if stemming_ck:
                data['final_abstrak'] = data['stemmed_tokens'].apply(lambda x: ' '.join(x))
            else:
                data['final_abstrak'] = data['tokens'].apply(lambda x: ' '.join(x))
            st.write(data)

# ============ VSM ============
        with vsm:
            st.warning("Anda harus memilih satu untuk proses LDA")
            binary_ck = st.checkbox("Binary")
            tf_ck = st.checkbox("Term Frequensi")
            log_tf_ck = st.checkbox("Log Term Frequensi")
            tfidf_ck = st.checkbox("TF-IDF")

# ============================ binary ====================
            # Inisialisasi OneHotEncoder
            encoder = OneHotEncoder()

            # Inisialisasi CountVectorizer
            vectorizer = CountVectorizer()  # mengubah setiap dokumen teks menjadi vektor

            # Melakukan one-hot encoding pada kolom 'final_abstrak'
            one_hot_encoded = vectorizer.fit_transform(data['final_abstrak'])

            # Mendapatkan nama fitur (kolom)
            fitur_names = vectorizer.get_feature_names_out()

            # Membuat DataFrame dari hasil one-hot encoding
            one_hot_df = pd.DataFrame(one_hot_encoded.toarray(), columns=fitur_names)

# ============================ TF ====================
            # Inisialisasi DataFrame untuk Term Frequency (TF)
            df = pd.DataFrame(data)

            # Inisialisasi CountVectorizer
            vectorizer = CountVectorizer()

            # Melakukan transformasi TF pada kolom 'final_abstrak'
            tf_matrix = vectorizer.fit_transform(df['final_abstrak'])

            # Membuat DataFrame dari hasil TF
            tf_df = pd.DataFrame(tf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

# ============================ Log-TF ====================
            # Inisialisasi CountVectorizer
            vectorizer = CountVectorizer()

            # Melakukan transformasi TF pada kolom 'final_abstrak'
            tf_matrix = vectorizer.fit_transform(df['final_abstrak'])

            # Menghitung log-TF dengan logaritma natural (ln)
            log_tf_matrix = np.log1p(tf_matrix)

            # Membuat DataFrame dari hasil log-TF
            log_tf_df = pd.DataFrame(log_tf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

# ============================ TFIDF ====================
            from sklearn.feature_extraction.text import TfidfVectorizer
            # Inisialisasi TfidfVectorizer
            tfidf_vectorizer = TfidfVectorizer()

            # Melakukan transformasi TF-IDF pada kolom 'final_abstrak'
            tfidf_matrix = tfidf_vectorizer.fit_transform(df['final_abstrak'])

            # Membuat DataFrame dari hasil TF-IDF
            tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())


            # =========== memilih ===============

            if binary_ck:
                st.info("#### Binary")
                # Cetak DataFrame one-hot encoded
                st.write(one_hot_df)
                dataLDA = one_hot_df
            elif tf_ck:
                st.info("#### Term Frequensi")
                st.write(tf_df)
                dataLDA = tf_df
            elif log_tf_ck:
                st.info("#### Log Term Frequensi")
                st.write(log_tf_df)
                dataLDA = log_tf_df
            elif tfidf_ck:
                st.info("#### TFIDF")
                st.write(tfidf_df)
                dataLDA = tfidf_df
            else:
                st.warning("Anda belum memilih VSM")
            # ============================ binary ====================





# =========================== LDA ===============================
    with lda:
        # requirements
        from sklearn.decomposition import LatentDirichletAllocation

        st.write("# LDA")
        st.write("Merupakan salah satu metode untuk mereduksi dimensi")
        if binary_ck:
            st.info("#### Data yang digunakan data VSM Binary")
            st.write(dataLDA)
        elif tf_ck:
            st.info("#### Data yang digunakan data VSM Term Frequensi")
            st.write(dataLDA)
        elif log_tf_ck:
            st.info("#### Data yang digunakan data VSM Log Term Frequensi")
            st.write(dataLDA)
        elif tfidf_ck:
            st.info("#### Data yang digunakan data VSM TFIDF")
            st.write(dataLDA)
        else:
            st.warning("Anda belum memilih VSM")


# ==================== proses LDA ====================
        # membuat variable k, alpha dan beta untuk proses LDA

        nk =  st.number_input("Masukkan nilai K untuk banyak topik:",step=1,value=1)
        alpha = st.number_input("Masukkan nilai alfa :",value=0.1)
        beta = st.number_input("Masukkan nilai beta :",value=0.2)

        # ========== kondisi data
        k = 0
        a = 0
        b = 0

        hasil_proporsi_td = st.checkbox("Tampilkan Hasil Proporsi Topik pada Dokumen")
        if hasil_proporsi_td:
            k=nk
            a=alpha
            b=beta

            # --------------------------------------
            lda_model = LatentDirichletAllocation(n_components=k, doc_topic_prior=a, topic_word_prior=b)
            # Proporsi topik pada dokumen
            proporsi_topik_dokumen = lda_model.fit_transform(dataLDA)

            # simpan kolom
            topik_kolom = []

            for i in range(1, k + 1):
                topik_kolom.append(f'Topik {i}')

            proporsi_topik_dokumen_df = pd.DataFrame(proporsi_topik_dokumen, columns=topik_kolom)

# ============= gabungkan label data ===========
            data_label = data['Label']
            proporsi_topik_dokumen_df = pd.concat([proporsi_topik_dokumen_df, data_label], axis=1)

            # hapus data kosong
            proporsi_topik_dokumen_df.dropna(inplace=True)


            st.info("Hasil Proporsi Topik Dokumen")
            st.write(proporsi_topik_dokumen_df)


# =========================== Modelling ==================================
    with modelling:
        modelApp, akurasiApp, evaluasi = st.tabs(["Modelling APP", "Akurasi Visual","Evaluasi"])
        with modelApp:
# ================= create model ==============
            # Loop untuk setiap iterasi topik
            # Data yang Anda miliki
            X = proporsi_topik_dokumen_df.iloc[:, :k]  # Mengambil hanya kolom-kolom topik pertama hingga ke-k
            y = proporsi_topik_dokumen_df['Label']

            nb_ck = st.checkbox("Naive Bayes")
            knn_ck = st.checkbox("KNN")
            rf_ck = st.checkbox("Random Forest")
            # ==================== function ==================
            def train_and_evaluate_model(X, y, k, model_type):
                accuracies = []
                reports = []

                for i in range(1, k + 1):
                    X_train, X_test, y_train, y_test = train_test_split(X.iloc[:, :i], y, test_size=0.3, random_state=0)
                    if model_type == 'Naive Bayes':
                        model = GaussianNB()
                    elif model_type == 'KNN':
                        model = KNeighborsClassifier()
                    else:
                        model = RandomForestClassifier()

                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)
                    accuracies.append(accuracy)

                    # Hitung laporan evaluasi
                    report = classification_report(y_test, y_pred)
                    reports.append(report)

                max_acc = max(accuracies)
                ind_max_acc = np.argmax(accuracies)

                return max_acc, topik_kolom[ind_max_acc],accuracies,reports
            # ==================== naive bayes ================
            if nb_ck:
                max_acc_nb, best_topic_nb,accuracies_nb,eval_nb = train_and_evaluate_model(X, y, k, 'Naive Bayes')
                st.info("###### Dengan menggunakan metode Naive Bayes akurasi tertinggi didapatkan sebesar:")
                st.info(f"Akurasi : {max_acc_nb}%, Pada {best_topic_nb}")

            # ==================== KNN ================
            if knn_ck:
                max_acc_knn, best_topic_knn,accuracies_knn,eval_knn = train_and_evaluate_model(X, y, k, 'KNN')
                st.warning("###### Dengan menggunakan metode KNN akurasi tertinggi didapatkan sebesar:")
                st.warning(f"Akurasi : {max_acc_knn}%, Pada {best_topic_knn}")

            # ================ Random Forest =========
            if rf_ck:    # Inisialisasi array untuk menyimpan akurasi
                max_acc_rf, best_topic_rf,accuracies_rf,eval_rf = train_and_evaluate_model(X, y, k, 'Random Forest')
                st.success("###### Dengan menggunakan metode Random Forest akurasi tertinggi didapatkan sebesar:")
                st.success(f"Akurasi : {max_acc_rf}%, Pada {best_topic_rf}")


#   ================= Grafik ============
        with akurasiApp:
            #   ======= function plot ========
            def plot_data(data_accuracy, topics, title):
                num_data = len(data_accuracy) # Menghitung jumlah data
                scale_factor = num_data / 10 # Tentukan faktor skala untuk figsize berdasarkan jumlah data

                fig, ax = plt.subplots(figsize=(8 * scale_factor, len(data_accuracy) / 2)) # Membuat plot dengan figsize yang disesuaikan
                ax.plot(topics, data_accuracy, color='b', marker='o', linestyle='-')  # Menambahkan garis yang menghubungkan titik-titik
                ax.scatter(topics, data_accuracy, color='b', marker='o')
                ax.set_title(title)
                ax.set_xlabel("\n Keterangan Topik")
                ax.set_ylabel("Nilai")
                ax.grid(True)

                for i in range(len(data_accuracy)):
                    ax.text(topics[i], data_accuracy[i], f"{data_accuracy[i]:.2f}", ha='center', va='bottom')

                return fig
# ================ Nb ===========
            if nb_ck:
                # Visualisasi NB
                data_accuracy_nb = accuracies_nb
                topics = topik_kolom
                fig_nb = plot_data(data_accuracy_nb, topics, "Visualisasi Data dengan Titik-Titik dan Keterangan Topik (Naive Bayes)")
                st.pyplot(fig_nb)

# ================ KNN ===========
            if knn_ck:
                # Visualisasi KNN
                data_accuracy_knn = accuracies_knn
                topics = topik_kolom
                fig_knn = plot_data(data_accuracy_knn, topics,
                                    "Visualisasi Data dengan Titik-Titik dan Keterangan Topik (KNN)")
                st.pyplot(fig_knn)

# ================ Random Forest ===========
            if rf_ck:
                # Visualisasi KNN
                data_accuracy_rf = accuracies_rf
                topics = topik_kolom
                fig_rf = plot_data(data_accuracy_rf, topics,
                                    "Visualisasi Data dengan Titik-Titik dan Keterangan Topik (Random Forest)")
                st.pyplot(fig_rf)

        with evaluasi:
            st.write("# Evaluasi")
            col1, col2, col3 = st.columns(3)
    #         ==================== nb ==============
            with col1:
                if nb_ck:
                    eval_nb_df = pd.DataFrame(eval_nb)
                    st.info("##### Evaluasi Naive Bayes")
                    st.write(eval_nb_df)

            with col2:
                if knn_ck:
                    eval_knn_df = pd.DataFrame(eval_knn)
                    st.info("##### Evaluasi KNN")
                    st.write(eval_knn_df)

            with col3:
                if knn_ck:
                    eval_rf_df = pd.DataFrame(eval_knn)
                    st.info("##### Evaluasi Random Forest")
                    st.write(eval_rf_df)



    # # =========================== Implementasi ===============================
    # with implementasi:
    #     st.write("# Implementasi")
    #     st.info(
    #         "Dalam melakukan pengecekan tingkat stres harus menggunakan 3 fitur data yang didapatkan dari melakukan beberapa aktivitas dan diukur menggunakan sebuah sensor. Aktivitas:")
    #     col1, col2, col3 = st.columns(3)
    #     with col1:
    #         st.write("")
    #     with col2:
    #         st.image("aktivitas.png", use_column_width="auto")
    #     with col3:
    #         st.write("")
    #     st.markdown("---")
    #     st.write("##### Input fitur")
    #     name = st.text_input("Masukkan nama anda")
    #     col1, col2, col3 = st.columns(3)
    #     with col1:
    #         humidity_mean = st.number_input("Masukkan Rata-rata Kelembaban (10.00 - 30.00)", min_value=10.00,
    #                                         max_value=30.00)
    #     with col2:
    #         temperature_mean = st.number_input("Masukkan rata-rata Suhu (79.00 - 99.00) Fahrenheit", min_value=79.00,
    #                                            max_value=99.00)
    #     with col3:
    #         step_count_mean = st.number_input("Masukkan rata-rata hitungan langkah (0.00 - 200.00)", min_value=0.00,
    #                                           max_value=200.00)
    #
    #     cek_hasil = st.button("Cek Prediksi")
    #     # ============================ Mengambil akurasi tertinggi ===========================
    #     if knn_accuracy > gauss_accuracy and knn_accuracy > decission3_accuracy:
    #         use_model = knn
    #         metode = "KNN"
    #     elif gauss_accuracy > knn_accuracy and gauss_accuracy > decission3_accuracy:
    #         use_model = gaussian
    #         metode = "Naive-Bayes Gaussian"
    #     else:
    #         use_model = decission3
    #         metode = "Decission Tree"
    #     # ============================ Normalisasi inputan =============================
    #     inputan = [[humidity_mean, temperature_mean, step_count_mean]]
    #     inputan_norm = scaler.transform(inputan)
    #     # inputan
    #     # inputan_norm
    #     FIRST_IDX = 0
    #     if cek_hasil:
    #         hasil_prediksi = use_model.predict(inputan_norm)[FIRST_IDX]
    #         if hasil_prediksi == 0:
    #             st.success(
    #                 f"{name} Terdeteksi tingkat stress tergolong Rendah, dengan tingkat = {hasil_prediksi} Berdasarkan metode {metode}")
    #         elif hasil_prediksi == 1:
    #             st.warning(
    #                 f"{name} Terdeteksi tingkat stress tergolong Normal, dengan tingkat = {hasil_prediksi} Berdasarkan metode {metode}")
    #         else:
    #             st.error(
    #                 f"{name} Terdeteksi tingkat stress tergolong Tinggi, dengan tingkat = {hasil_prediksi} Berdasarkan metode {metode}")
