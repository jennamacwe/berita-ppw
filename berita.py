import streamlit as st
import pickle
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# stopwords
import nltk
from nltk.corpus import stopwords

# Streamlit bagian aplikasi
st.title("Pencarian dan Penambangan WEB")
st.write("Nama  : Jennatul Macwe ")
st.write("Nim   : 210411100151 ")
st.write("Kelas : Pencarian dan Penambangan WEB B ")

tab1, = st.tabs(["Implementasi"])

# Inisialisasi stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Unduh stopwords bahasa Indonesia
nltk.download('stopwords')
stop_words = stopwords.words('indonesian')

# Fungsi preprocessing
def remove_symbols(data_berita):
    data_berita = re.sub(r'[^a-zA-Z0-9\s]', '', data_berita)  # Menghapus karakter
    return data_berita

def case_folding(text):
    if isinstance(text, str):
        lowercase_text = text.lower()
        return lowercase_text
    else:
        return text

def tokenize(text):
    tokens = text.split()
    return tokens

def remove_stopwords(text):
    return [word for word in text if word not in stop_words]

def stemming(text):
    return [stemmer.stem(word) for word in text]

def preprocessing(berita):
    berita = remove_symbols(berita)
    berita = case_folding(berita)
    tokens = tokenize(berita)
    tokens = remove_stopwords(tokens)
    stemming_result = stemming(tokens)
    return ' '.join(stemming_result)

# Fungsi untuk memuat model dan vectorizer dari file lokal
def load_model_and_vectorizer():
    with open('logistic_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer

# Fungsi prediksi kategori berita
def predict_category(berita_input, model, vectorizer):
    # Preprocessing hanya untuk input user
    preprocessed_berita = preprocessing(berita_input)
    berita_vectorized = vectorizer.transform([preprocessed_berita])
    prediction = model.predict(berita_vectorized)
    return 'Politik' if prediction == 0 else 'Gaya Hidup'

with tab1:
    st.title("Prediksi Kategori Berita")
    user_input = st.text_area("Isi Berita", height=200)

    if st.button("Prediksi"):
        if user_input:
            # Memuat model dan vectorizer yang sudah disimpan
            model, vectorizer = load_model_and_vectorizer()
            predicted_category = predict_category(user_input, model, vectorizer)
            st.write(f"**Kategori berita:** {predicted_category}")
        else:
            st.write("Harap masukkan isi berita.")
