import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import os

# Load the pre-trained model
model = tf.keras.models.load_model('batik_model.h5')

# Class names dictionary
class_names = {
    0: 'Batik Cap Asem Arang',
    1: 'Batik Cap Asem Sinom',
    2: 'Batik Cap Asem Warak',
    3: 'Batik Cap Blekok',
    4: 'Batik Cap Blekok Warak',
    5: 'Batik Cap Gambang Semarangan',
    6: 'Batik Cap Kembang Sepatu',
    7: 'Batik Cap Semarangan',
    8: 'Batik Cap Tugu Muda',
    9: 'Batik Cap Warak Beras Utah'
}

# Batik descriptions and characteristics with images
batik_details = {
    'Batik Cap Asem Arang': {
        'ciri': 'Warna latar belakang kain adalah Hijau Tua. Pola yang terlihat terdiri dari elemen-elemen berbentuk cabang-cabang berwarna coklat keemasan. Terdapat beberapa elemen berbentuk lonjong dengan warna biru, hijau, dan hitam, serta beberapa elemen bulat dengan warna merah dan putih di dalam pola cabang-cabang tersebut. Motif-motif ini tersusun secara simetris dan berulang di seluruh permukaan kain. Pola dan warna yang digunakan mengingatkan pada gaya batik tradisional, dengan kombinasi warna dan bentuk yang khas.',
        'gambar': 'asem_arang.jpg'
    },
    'Batik Cap Asem Sinom': {
        'ciri': 'Warna latar belakang kain adalah Hijau toska (turquoise green). Terdapat gambar-gambar menyerupai biji atau kacang polong berwarna merah dan oranye muda. Disertai dengan daun-daun kecil berwarna putih, dan pola dedaunan yang berwarna hijau muda yang tersebar di seluruh kain. Pola dedaunan disusun secara acak dengan berbagai ukuran dan orientasi, memberikan kesan alami dan dinamis. Pola tersebut tampak seperti motif batik atau kain tradisional yang sering ditemukan pada tekstil Indonesia atau Asia Tenggara.',
        'gambar': 'asem_sinom.jpg'
    },
    'Batik Cap Asem Warak': {
        'ciri': 'Warna latar belakang kain adalah Oren. Gambar menyerupai binatang berwarna merah, hijau, dan oranye dengan pola geometris yang unik pada tubuhnya. Di sepanjang sisi kain, terdapat pola geometris berwarna krem dengan aksen merah dan hijau, yang menyerupai motif tradisional atau etnik. Binatang tersebut memiliki bentuk yang menyerupai hewan fiksi atau karakter kartun dengan pola yang mencolok di tubuhnya, termasuk titik-titik dan lingkaran. Pola tersebut tampak seperti motif kain tradisional atau etnik yang berfokus pada elemen hewan dan desain geometris yang kaya akan warna dan detail.',
        'gambar': 'asem_warak.jpg'
    },
    'Batik Cap Blekok': {
        'ciri': 'Warna latar belakang kain adalah Hijau tua. Gambar burung berwarna kuning dengan detail garis putih, menyerupai burung bangau atau burung lain yang sedang terbang dengan sayap yang terbuka lebar. Terdapat elemen dedaunan dan bunga berwarna kuning dengan detail putih yang mengelilingi burung tersebut. Pola ini diulang secara teratur di seluruh kain, memberikan kesan simetris dan teratur. Setiap burung dikelilingi oleh dedaunan yang menyerupai flora alami. Motif ini tampak seperti motif batik atau kain tradisional, dengan fokus pada elemen alam dan fauna yang sering ditemukan dalam seni tekstil Indonesia atau Asia Tenggara.',
        'gambar': 'blekok.jpg'
    },
    'Batik Cap Blekok Warak': {
        'ciri': 'Warna latar belakang kain adalah Merah marun. Gambar hewan menyerupai kuda dengan warna oranye dan putih. Hewan ini memiliki detail pola di tubuhnya yang menyerupai garis-garis. Gambar burung berwarna putih dengan aksen oranye, mirip dengan bangau atau burung berkaki panjang. Daun-daun kecil berwarna putih yang tersebar di seluruh permukaan kain. Tepi kain dihiasi dengan pola geometris yang mengelilingi keseluruhan kain, berwarna merah dan putih.',
        'gambar': 'blekok_warak.jpg'
    },
    'Batik Cap Gambang Semarangan': {
        'ciri': 'Warna latar belakang kain adalah Hitam. Elemen berbentuk cabang-cabang atau sulur berwarna coklat keemasan yang tersebar merata di seluruh kain. Elemen berbentuk lonjong dengan warna biru, hijau, dan hitam. Elemen berbentuk bulat dengan warna merah dan putih. Pola tersusun secara simetris dan berulang di seluruh permukaan kain. Kombinasi warna dan bentuk yang khas, mengingatkan pada gaya batik tradisional. Ada beberapa bentuk geometris seperti persegi yang mengelilingi elemen-elemen utama. Elemen-elemen utama memiliki detail garis-garis yang menambah kekayaan visual pola kain.',
        'gambar': 'gambang_semarangan.jpg'
    },
    'Batik Cap Kembang Sepatu': {
        'ciri': 'Warna latar belakang kain adalah Merah marun. Motif bunga besar yang tampak seperti bunga sepatu atau bunga serupa, dengan warna putih dan oranye. Daun-daun yang mengelilingi bunga tersebut, dengan warna oranye dan putih yang serasi. Motif bunga dan daun tersusun secara simetris dan berulang di seluruh permukaan kain. Pola bunga dan daun ditampilkan dengan detail garis yang menonjol, memberikan kesan artistik dan tradisional. Motif bunga dan daun memiliki variasi bentuk dan ukuran, namun tetap konsisten dalam penggunaan warna dan gaya garis.',
        'gambar': 'kembang_sepatu.jpg'
    },
    'Batik Cap Semarangan': {
        'ciri': 'Warna dominan pada kain ini adalah Hitam, dengan pola yang menggunakan warna putih dan merah. Terdapat beberapa elemen arsitektur yang khas, termasuk bangunan dengan kubah dan menara, yang kemungkinan besar adalah representasi bangunan ikonik dari suatu daerah. Pola juga melibatkan elemen-elemen flora seperti batang-batang panjang yang mungkin mewakili bambu atau tanaman lainnya, serta fauna yang menyerupai burung. Pola ini diulang secara terus menerus, menciptakan efek visual yang kaya dan detail. Warna merah pada beberapa bagian memberikan kontras yang mencolok terhadap latar belakang hitam dan pola putih.',
        'gambar': 'semarangan.jpg'
    },
    'Batik Cap Tugu Muda': {
        'ciri': 'Warna dasar kain ini adalah Biru, dengan pola berwarna putih, merah, hijau, dan sedikit ungu. Salah satu motif utama yang jelas terlihat adalah representasi dari Tugu Muda, sebuah monumen terkenal di Semarang, yang digambarkan dengan warna merah dan putih. Terdapat elemen-elemen floral dengan daun dan batang yang berkelok-kelok, menggunakan warna putih, hijau, dan ungu. Pola ini juga mengandung elemen-elemen abstrak yang memberikan kesan dinamis dan penuh gerak. Motif Tugu Muda dan elemen floral diulang secara teratur di seluruh kain, menciptakan pola yang konsisten. Penggunaan warna merah pada motif Tugu Muda memberikan kontras yang kuat terhadap latar belakang biru, membuatnya sangat menonjol.',
        'gambar': 'tugu_muda.jpg'
    },
    'Batik Cap Warak Beras Utah': {
        'ciri': 'Warna dasar kain ini adalah Merah cerah, dengan pola yang menggunakan warna kuning, biru, hijau, dan putih. Motif utama pada kain ini adalah gambar hewan yang dikenal sebagai Warak Ngendog, yang merupakan simbol budaya dari Kota Semarang. Warak Ngendog digambarkan dengan berbagai warna seperti biru, merah, dan hijau. Ada pola gelombang atau garis melengkung berwarna kuning yang mengisi latar belakang kain, memberikan kesan dinamis dan ritmis. Motif Warak Ngendog dan pola gelombang diulang secara teratur di seluruh kain, menciptakan pola yang konsisten. Kombinasi warna cerah seperti merah, kuning, biru, dan hijau menciptakan kontras yang kuat, membuat motif hewan dan pola latar belakang sangat menonjol.',
        'gambar': 'warak_beras_utah.jpg'
    }
}

# Function to predict batik name
# def predict_batik(image):
#     image = image.resize((256, 256))  # Resize the image to the expected input size
#     image_array = np.array(image) / 255.0
#     image_array = np.expand_dims(image_array, axis=0)
#     predictions = model.predict(image_array)
#     print(predictions)
#     max_prediction = np.max(predictions)
#     if max_prediction < 0.95:  # Threshold for prediction confidence
#         return "Tidak dapat diprediksi"
#     predicted_class = class_names[np.argmax(predictions)]
#     return predicted_class

def predict_batik(image):
    image = image.resize((256, 256))  # Resize the image to the expected input size
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    predictions = model.predict(image_array)
    print(predictions)
    max_prediction = np.max(predictions)
    if max_prediction < 0.95:  # Threshold for prediction confidence
        return "Tidak dapat diprediksi"
    predicted_class = class_names[np.argmax(predictions)]
    return predicted_class

# Function to process the image from the camera
# def process_image(img_pil):
#     prediction = predict_batik(img_pil)
#     return prediction

# Custom CSS
st.markdown("""
    <style>
    .header-font {
        font-size: 30px !important;
        font-weight: bold;
        color: #FF6347;
        text-align: center;
    }
    .description-font {
        font-size: 18px !important;
        text-align: justify;
    }
    .prediction {
        font-size: 24px !important;
        font-weight: bold;
        color: #32CD32;
        text-align: center;
    }
    .sidebar .sidebar-content {
        padding: 20px;
    }
    .nav-button {
        background-color: #FF6347;
        color: white;
        border: none;
        padding: 10px;
        font-size: 18px;
        font-weight: bold;
        text-align: center;
        margin: 10px 0;
        border-radius: 5px;
        cursor: pointer;
        width: 100%;
        display: block;
    }
    .nav-button:hover {
        background-color: #FF4500;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar styling with buttons
st.sidebar.title("Navigasi")
menu = ["Beranda", "Jenis Batik", "Upload Gambar", "Kamera", "Tentang"]
choice = st.sidebar.radio("Pilih halaman", menu)

# Beranda
if choice == "Beranda":
    st.markdown('<p class="header-font">Selamat Datang di Web Batik Semarang</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown('<p class="description-font">Web Streamlit ini membantu Anda mengenali berbagai jenis batik Semarang melalui gambar yang diunggah atau menggunakan kamera web (desktop).</p>', unsafe_allow_html=True)
        st.markdown('<p class="description-font">Mari eksplorasi keindahan dan makna dari setiap motif batik Semarang.</p>', unsafe_allow_html=True)
    
    with col2:
        st.image("logo.jpg", caption='Gambar Batik Semarang', use_column_width=True)

# Jenis Batik
elif choice == "Jenis Batik":
    st.markdown('<p class="header-font">Jenis Batik Semarang</p>', unsafe_allow_html=True)
    for batik_name, details in batik_details.items():
        with st.expander(batik_name):
            image_path = os.path.join('images', details['gambar'])
            if os.path.exists(image_path):
                image = Image.open(image_path)
                resized_image = image.resize((200, 200))  # Resize the image to be smaller
                st.image(resized_image, use_column_width=False)  # Set use_column_width to False to control size
            else:
                st.warning(f"Gambar untuk {batik_name} tidak ditemukan.")
            st.markdown(f"**Ciri-Ciri:** {details['ciri']}")

# Upload Gambar
elif choice == "Upload Gambar":
    st.markdown('<p class="header-font">Upload Gambar</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Pilih file gambar", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Gambar yang diunggah', use_column_width=True)
        st.write("Memprediksi...")

        predicted_class = predict_batik(image)
        st.markdown(f'<p class="prediction">Nama Batik: {predicted_class}</p>', unsafe_allow_html=True)

# Kamera
elif choice == "Kamera":
    st.markdown('<p class="header-font">Kamera</p>', unsafe_allow_html=True)
    st.markdown('<p class="description-font">Hanya dapat diakses atau digunakan dengan kamera webcam (desktop).</p>', unsafe_allow_html=True)

    # class VideoProcessor(VideoProcessorBase):
    #     def __init__(self):
    #         self.prediction = ""

    #     def transform(self, frame):
    #         img = frame.to_ndarray(format="bgr24")
    #         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #         img_pil = Image.fromarray(img_rgb)
    #         self.prediction = predict_batik(img_pil)
    #         font = cv2.FONT_HERSHEY_SIMPLEX
    #         cv2.putText(img, self.prediction, (50, 50), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
    #         return img

    #     def get_prediction(self):
    #         return self.prediction

    # ctx = webrtc_streamer(key="example", video_processor_factory=VideoProcessor)
    img_file_buffer = st.camera_input("Take a picture")
    if img_file_buffer is not None:
        # To read image file buffer with PIL:
        img_pil = Image.open(img_file_buffer)
        
        # Process the image
        prediction = predict_batik(img_pil)
        
        # Display the prediction
        st.write(f"Prediction: {prediction}")
        
        # Display the image with the prediction text
        img_rgb = np.array(img_pil)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img_bgr, prediction, (50, 50), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
        
        st.image(img_bgr, channels="BGR")


#Tentang
elif choice == "Tentang":
    st.markdown('<p class="header-font">Tentang Kami</p>', unsafe_allow_html=True)
    
    # Center the university logo below the heading
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("gunadarma_logo.png", caption='Logo Universitas', use_column_width=True)  # Mengubah use_column_width menjadi True

    st.markdown('<p class="description-font">Web Streamlit Batik Semarang dibuat untuk mengenali dan melestarikan berbagai motif batik dari Semarang. Jika ada pertanyaan dan saran, silakan hubungi kami.</p>', unsafe_allow_html=True)
    st.markdown('<p class="description-font">Kontak:</p>', unsafe_allow_html=True)
    st.markdown('<ul class="description-font"><li>Email: diastialfiana08@gmail.com</li><li>Telp: +628-577-682-344-99</li></ul>', unsafe_allow_html=True)
