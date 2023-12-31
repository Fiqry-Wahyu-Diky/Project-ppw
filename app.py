import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

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
from sklearn.feature_extraction.text import TfidfVectorizer

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
    crawling, preprocessing, lda, modelling, implementasi = st.tabs(
["Crawling", "Preprocessing", "Reduksi LDA", "Modeling","Implementasi"])


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

    # ============ button crawling =====================
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
        data, cleaned, vsm = st.tabs(
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

# ======== cleanned ===================
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


# ================== tokenizing =================
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



# ============ VSM =========================
        with vsm:
            # Membuat DataFrame
            # ============================ BINARY ====================
            st.success("### Binary")
            binary = pd.read_csv("binary_matrix.csv")
            st.dataframe(binary)

            # ============================ TF ====================
            st.success("### Term Frequensi")
            tf = pd.read_csv("tf_matrix.csv")
            st.dataframe(tf)

            # ============================ LOG-TF ====================
            st.success("### Log-Term Frequensi")
            Logtf = pd.read_csv("log_tf_matrix.csv")
            st.dataframe(Logtf)

            # ============================ TFIDF ====================
            st.success("### TFIDF")
            tfidf = pd.read_csv("tfidf_matrix.csv")
            st.dataframe(tfidf)


# =========================== LDA ===============================
    with lda:
        # requirements
        from sklearn.decomposition import LatentDirichletAllocation
        st.write("## LDA")

        st.write("Data yang digunakan untuk proses LDA adalah data TFIDF ")

        st.write("Hasil LDA menggunakan k = 20, alpha = 0.1, dan beta = 0.2")
        st.success("##### Proporsi topik pada dokumen")

        st.warning("###### Train")
        ptd_train = pd.read_csv("proporsi_topik_dokumen_df_train_final.csv")
        st.dataframe(ptd_train)

        st.warning("###### Test")
        ptd_test = pd.read_csv("proporsi_topik_dokumen_df_test_final.csv")
        st.dataframe(ptd_test)

        st.success("##### Proporsi kata pada topik")

        st.warning("###### Test")
        ptk_train = pd.read_csv("ProporsiKataTopik_df_train.csv")
        st.dataframe(ptk_train)

        st.warning("###### Test")
        ptk_test = pd.read_csv("ProporsiKataTopik_df_test.csv")
        st.dataframe(ptk_test)



# =========================== Modelling ==================================
    with modelling:
        modelApp, akurasiApp= st.tabs(["Modelling APP", "Akurasi Visual"])
        with modelApp:
            # ========== nb ==========
            acc_nb = pd.read_csv("accuracies_nb_df.csv")
            max_index_acc = acc_nb["Akurasi"].idxmax()
            max_acc_nb = acc_nb.loc[max_index_acc]["Akurasi"]
            best_topik_nb = acc_nb.loc[max_index_acc]["Topik"]
            st.warning("###### Dengan menggunakan metode Naive Bayes akurasi tertinggi didapatkan sebesar:")
            st.warning(f"Akurasi : {max_acc_nb}%, Pada Topik- {best_topik_nb} ")

            # ========== knn ==========
            acc_knn = pd.read_csv("accuracies_knn_df.csv")
            max_index_acc = acc_knn["Akurasi"].idxmax()
            max_acc_knn = acc_knn.loc[max_index_acc]["Akurasi"]
            best_topik_knn = acc_knn.loc[max_index_acc]["Topik"]
            st.error("###### Dengan menggunakan metode KNN akurasi tertinggi didapatkan sebesar:")
            st.error(f"Akurasi : {max_acc_knn}%, Pada Topik- {best_topik_knn}")

            # ================ Random Forest =========
            acc_rf = pd.read_csv("accuracies_rf_df.csv")
            max_index_acc = acc_rf["Akurasi"].idxmax()
            max_acc_rf = acc_rf.loc[max_index_acc]["Akurasi"]
            best_topik_rf = acc_rf.loc[max_index_acc]["Topik"]
            st.success("###### Dengan menggunakan metode Random Forest akurasi tertinggi didapatkan sebesar:")
            st.success(f"Akurasi : {max_acc_rf}%, Pada Topik- {best_topik_rf}")


#   ================= Grafik ============
        with akurasiApp:
            st.write("Akurasi")
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

                # Mengatur nilai-nilai pada sumbu x
                ax.set_xticks(range(1, len(data_accuracy) + 1))

                return fig
# ================ Nb ===========
            # Visualisasi NB
            fig_nb = plot_data(acc_nb["Akurasi"], acc_nb["Topik"], "Visualisasi Data (Naive Bayes)")
            st.pyplot(fig_nb)

# ================ KNN ===========
            # Visualisasi knn
            fig_knn = plot_data(acc_knn["Akurasi"], acc_knn["Topik"], "Visualisasi Data (KNN)")
            st.pyplot(fig_knn)
#
# ================ Random Forest ===========
            # Visualisasi rf
            fig_rf = plot_data(acc_rf["Akurasi"], acc_rf["Topik"], "Visualisasi Data (Random Forest)")
            st.pyplot(fig_rf)


    # =========================== Implementasi ===============================
    with implementasi:
        st.write("# Implementasi")
        st.info(f"Dalam implementasi akan digunakan metode yang paling tinggi akurasinya (dalam evaluasi) yaitu: metode Random Forest dan menggunakan {best_topik_rf} Topik")
        inp = st.text_input("Masukkan abstrak")
        st.warning(f"VSM yang digunakan yaitu TFIDF")

        if st.button("Calculate"):

    # ============== tf-idf ============
            # Melakukan transformasi TF-IDF pada kolom
            tfidf_vectorizer = TfidfVectorizer()
            tfidf_matrix_inp = tfidf_vectorizer.fit_transform([inp])
            # Membuat DataFrame dari hasil TF-IDF
            tfidf_inp_df = pd.DataFrame(tfidf_matrix_inp.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
            st.info("Vectorizer TFIDF")
            st.dataframe(tfidf_inp_df)

    #   ============== LDA inp =========
            lda_model_inp = joblib.load("lda_model.pkl")
            # Proporsi topik pada dokumen
            proporsi_topik_dokumen_inp = lda_model_inp.fit_transform(tfidf_inp_df)

            # simpan kolom
            topik_kolom_inp = []

            for i in range(1, 8+1):
                topik_kolom_inp.append(f'Topik {i}')
            # st.write("kolom input")
            # st.write(topik_kolom_inp)
    # ====================== topik pd dokumen =========
            inp_proporsi_topik_dokumen_df = pd.DataFrame(proporsi_topik_dokumen_inp, columns=topik_kolom_inp)
            # proporsi_topik_dokumen_df.insert(0,'stemmed_tokens', abstrak)
            st.info("Proporsi Topik pada Dokumen")
            st.write(inp_proporsi_topik_dokumen_df)

    #     # ===================== kata pada topik ==========
        # Proporsi kata pada topik
            inp_fitur = tfidf_inp_df.columns.tolist()
            ProporsiKataTopik_inp = lda_model_inp.components_
            inp_ProporsiKataTopik_df = pd.DataFrame(ProporsiKataTopik_inp, columns=inp_fitur)
            inp_ProporsiKataTopik_df.insert(0, 'Topik', topik_kolom_inp)
            st.warning("Proporsi kata pada Topik")
            st.write(inp_ProporsiKataTopik_df)

    # ===========  predict =========

            # Mendapatkan model terbaik dari kamus models
            best_model = joblib.load("rf_model_imp.pkl")
            # st.write(best_model)
            # Lakukan prediksi
            predict_inp = best_model.predict(inp_proporsi_topik_dokumen_df)
            # st.write(predict_inp)
            if predict_inp == "KK":
                st.success("Hasil Prediksi dari Kalimat yang diinputkan menunjukkan : ")
                st.success("KK")
            else:
                st.success("Hasil Prediksi dari Kalimat yang diinputkan menunjukkan : ")
                st.success("RPL")


