import pandas as pd
import tensorflow as tf
# Menonaktifkan pesan warning
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

# Load model
model = load_model("D:/Walter/Semester 7/NLP/analisis_sentimen_EdLink - Copy/model/saved_model.keras")  # Ganti 

# Preprocessing teks pada data pelatihan
data = pd.read_csv("D:/Walter/Semester 7/NLP/analisis_sentimen_EdLink - Copy/data_cleaned.csv")
max_features = 1000  # Jumlah kata unik yang digunakan saat tokenisasi
tokenizer = Tokenizer(num_words=max_features, split=" ")
tokenizer.fit_on_texts(data["txt"].values)
maxlen = max(len(seq) for seq in tokenizer.texts_to_sequences(data['txt'].values))

# Loop untuk prediksi teks
print("Ketik 'exit' untuk keluar.")
while True:
    # Input kalimat yang ingin diprediksi sentimennya
    text_to_predict = input("Silahkan Masukan kalimat yang akan diprediksi: ")
    
    # Berhenti jika input adalah 'exit'
    if text_to_predict.lower() == 'exit':
        print("Keluar dari program. Terima kasih!")
        break

    # Buat DataFrame untuk kalimat yang ingin diprediksi
    new_df = pd.DataFrame({'txt': [text_to_predict]})
    
    # Transformasi teks baru menjadi sekuen angka
    X_new = tokenizer.texts_to_sequences(new_df['txt'].values)
    X_new = pad_sequences(X_new, maxlen=maxlen)

    # Lakukan prediksi pada data baru
    prediction = model.predict(X_new)

    # Tampilkan hasil prediksi
    if prediction[0][0] > 0.5:
        print("Sentimen: Positif")
    else:
        print("Sentimen: Negatif")
