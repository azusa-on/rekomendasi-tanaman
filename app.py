import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

# == Konfigurasi halaman ==
st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌾",
    layout="wide"
)

# == Load model & objek ==
@st.cache_resource
def load_models():
    rf       = joblib.load('model_rf.pkl')
    knn      = joblib.load('model_knn.pkl')
    kmeans   = joblib.load('model_kmeans.pkl')
    scaler   = joblib.load('scaler.pkl')
    le       = joblib.load('label_encoder.pkl')
    return rf, knn, kmeans, scaler, le

rf_model, knn_model, kmeans_model, scaler, le = load_models()

# == Load dataset ==
@st.cache_data
def load_data():
    return pd.read_csv('Crop_recommendation.csv')

df = load_data()

# == Nama cluster ==
cluster_names = {
    0: 'Lahan Kering Basa',
    1: 'Lahan Basah Subur',
    2: 'Lahan Nitrogen Tinggi',
    3: 'Lahan Subtropis Mineral Tinggi',
    4: 'Lahan Lembab Tropis',
    5: 'Lahan Kering Sejuk',
    6: 'Lahan Panas Kering Asam',
}

# Tambahkan ini — generate kolom cluster dari model yang sudah diload
features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
X = df[features]
X_scaled = scaler.transform(X)
df['cluster'] = kmeans_model.predict(X_scaled)
df['cluster_name'] = df['cluster'].map(cluster_names)

# == Sidebar navigasi ==
st.sidebar.title("🌾 Crop Recommendation")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigasi",
    ["🏠 Home", "📊 Dataset Overview", "🔍 Analysis", "📈 Visualization", "ℹ️ About"]
)

# == Page Home ==
def show_home():
    # Header
    st.title("🌾 Sistem Rekomendasi Tanaman Berbasis Data Mining")
    st.markdown("---")

    # Deskripsi singkat
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Tentang Sistem Ini
        Sistem ini membantu petani dalam menentukan jenis tanaman yang paling optimal 
        berdasarkan kondisi lahan dan iklim menggunakan pendekatan **Data Mining**.
        
        Permasalahan yang diselesaikan:
        - 🌱 Petani tidak tahu **tipe lahannya** termasuk kategori apa
        - 🌦️ Petani kesulitan menentukan **tanaman yang cocok** berdasarkan kondisi cuaca dan tanah
        - 📉 Pemilihan tanaman yang salah menyebabkan **hasil panen tidak optimal**
        
        ### Solusi yang Ditawarkan
        Dengan memasukkan **7 parameter kondisi lahan**, sistem akan memberikan:
        - ✅ **Rekomendasi tanaman** yang paling optimal
        - ✅ **Tipe lahan** berdasarkan karakteristik kondisi tanah
        """)
    
    with col2:
        st.info("""
        **Dataset**
        Crop Recommendation Dataset
        
        **Jumlah Data**
        2.200 records
        
        **Jumlah Tanaman**
        22 jenis tanaman
        
        **Metode**
        - K-Means Clustering
        - Random Forest
        - K-Nearest Neighbors
        """)

    st.markdown("---")

    # Fitur parameter
    st.markdown("### 📋 Parameter yang Digunakan")
    col1, col2, col3, col4 = st.columns(4)
    
    params = [
        ("🌿", "Nitrogen (N)", "Kandungan nitrogen dalam tanah"),
        ("🌿", "Fosfor (P)", "Kandungan fosfor dalam tanah"),
        ("🌿", "Kalium (K)", "Kandungan kalium dalam tanah"),
        ("🌡️", "Suhu", "Suhu rata-rata lingkungan (°C)"),
        ("💧", "Kelembaban", "Kelembaban udara (%)"),
        ("⚗️", "pH Tanah", "Tingkat keasaman tanah"),
        ("🌧️", "Curah Hujan", "Curah hujan rata-rata (mm)"),
    ]
    
    cols = [col1, col2, col3, col4]
    for i, (icon, name, desc) in enumerate(params):
        with cols[i % 4]:
            st.metric(label=f"{icon} {name}", value="", delta=desc)

    st.markdown("---")

    # Cara penggunaan
    st.markdown("### 🚀 Cara Menggunakan Sistem")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.success("**Step 1 — Input Data**\n\nMasukkan 7 parameter kondisi lahan kamu di halaman Analysis")
    with col2:
        st.warning("**Step 2 — Proses**\n\nSistem memproses data menggunakan model Machine Learning yang telah dilatih")
    with col3:
        st.info("**Step 3 — Hasil**\n\nDapatkan rekomendasi tanaman dan tipe lahan secara otomatis")

    st.markdown("---")
    st.markdown("### 👥 Tim Pengembang")
    st.markdown("**Mata Kuliah:** Data Mining")
    st.markdown("**Program Studi:** *Sistem Informasi*")
    st.markdown("**Universitas Negeri Surabaya**")
    st.markdown("")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        | No | Nama | NIM |
        |---|---|---|
        | 1 | *Teuku Rifqi Ar Rafi'* | *24051214129* |
        | 2 | *Belva Dzakwan Maula Ramadhan* | *24051214138* |
        """)
    with col2:
        st.info("""
        **Repository Proyek**
        🔗 *https://github.com/azusa-on/rekomendasi-tanaman.git*
        
        **Notebook**
        📓 *https://colab.research.google.com/drive/1lnK_66YZROrRksTe4Yig1urTocwQ-D6Z?usp=sharing*
        """)

# == Page Dataset Overview ==
def show_dataset():
    st.title("📊 Dataset Overview")
    st.markdown("---")

    # Info umum dataset
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Data", "2.200")
    col2.metric("Jumlah Fitur", "7")
    col3.metric("Jenis Tanaman", "22")
    col4.metric("Missing Values", "0")

    st.markdown("---")

    # Tampilkan dataframe
    st.markdown("### 📋 Sample Data")
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("---")

    # Statistik deskriptif
    st.markdown("### 📈 Statistik Deskriptif")
    st.dataframe(df.describe().round(2), use_container_width=True)

    st.markdown("---")

    # Distribusi label
    st.markdown("### 🌱 Distribusi Jenis Tanaman")
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("**Jumlah data per tanaman:**")
        label_counts = df['label'].value_counts().reset_index()
        label_counts.columns = ['Tanaman', 'Jumlah']
        st.dataframe(label_counts, use_container_width=True, hide_index=True)

    with col2:
        st.image('fig_distribusi_label.png', use_container_width=True)

    st.markdown("---")

    # Penjelasan fitur
    st.markdown("### 📌 Penjelasan Fitur")
    fitur_info = pd.DataFrame({
        'Fitur': ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'],
        'Nama Lengkap': ['Nitrogen', 'Fosfor', 'Kalium', 'Suhu', 'Kelembaban', 'pH Tanah', 'Curah Hujan'],
        'Satuan': ['mg/kg', 'mg/kg', 'mg/kg', '°C', '%', '-', 'mm'],
        'Min': [df['N'].min(), df['P'].min(), df['K'].min(),
                round(df['temperature'].min(), 2), round(df['humidity'].min(), 2),
                round(df['ph'].min(), 2), round(df['rainfall'].min(), 2)],
        'Max': [df['N'].max(), df['P'].max(), df['K'].max(),
                round(df['temperature'].max(), 2), round(df['humidity'].max(), 2),
                round(df['ph'].max(), 2), round(df['rainfall'].max(), 2)],
        'Rata-rata': [round(df['N'].mean(), 2), round(df['P'].mean(), 2),
                      round(df['K'].mean(), 2), round(df['temperature'].mean(), 2),
                      round(df['humidity'].mean(), 2), round(df['ph'].mean(), 2),
                      round(df['rainfall'].mean(), 2)]
    })
    st.dataframe(fitur_info, use_container_width=True, hide_index=True)

# == Page analysis ==
def show_analysis():
    st.title("🔍 Analysis")
    st.markdown("---")

    st.markdown("### 📋 Masukkan Parameter Kondisi Lahan")
    st.markdown("Isi semua parameter di bawah ini sesuai kondisi lahan kamu.")

    # Form input
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**🌿 Kandungan Unsur Hara Tanah**")
        N = st.slider("Nitrogen (N)", min_value=0, max_value=140, value=50,
                      help="Kandungan nitrogen dalam tanah (mg/kg)")
        P = st.slider("Fosfor (P)", min_value=5, max_value=145, value=50,
                      help="Kandungan fosfor dalam tanah (mg/kg)")
        K = st.slider("Kalium (K)", min_value=5, max_value=205, value=50,
                      help="Kandungan kalium dalam tanah (mg/kg)")
        ph = st.slider("pH Tanah", min_value=3.5, max_value=9.9, value=6.5, step=0.1,
                       help="Tingkat keasaman tanah (3.5=asam, 7=netral, 9.9=basa)")

    with col2:
        st.markdown("**🌦️ Kondisi Iklim**")
        temperature = st.slider("Suhu (°C)", min_value=8.0, max_value=44.0, value=25.0, step=0.1,
                                help="Suhu rata-rata lingkungan dalam °C")
        humidity = st.slider("Kelembaban (%)", min_value=14.0, max_value=100.0, value=70.0, step=0.1,
                             help="Kelembaban udara dalam persen")
        rainfall = st.slider("Curah Hujan (mm)", min_value=20.0, max_value=299.0, value=100.0, step=0.1,
                             help="Curah hujan rata-rata dalam mm")

    st.markdown("---")

    # Tombol proses
    if st.button("🚀 Analisis Sekarang", use_container_width=True, type="primary"):

        # Preprocessing input
        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        input_scaled = scaler.transform(input_data)

        # Prediksi Classification
        pred_rf  = le.inverse_transform(rf_model.predict(input_scaled))[0]
        pred_knn = le.inverse_transform(knn_model.predict(input_scaled))[0]

        # Prediksi Clustering
        cluster_id   = kmeans_model.predict(input_scaled)[0]
        cluster_name = cluster_names[cluster_id]

        # Tanaman dominan di cluster ini
        top_crops = df[df['cluster'] == cluster_id]['label'].value_counts().head(5).index.tolist() \
            if 'cluster' in df.columns else []

        st.markdown("---")
        st.markdown("## 🎯 Hasil Analisis")

        # Hasil clustering
        st.markdown("### 🗺️ Tipe Lahan Kamu")
        st.success(f"**{cluster_name}**")

        # Profil lahan
        cluster_profile = df.groupby('cluster')[['N','P','K','temperature','humidity','ph','rainfall']].mean()
        if cluster_id in cluster_profile.index:
            profile = cluster_profile.loc[cluster_id]
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Rata-rata N", f"{profile['N']:.1f}")
            col2.metric("Rata-rata P", f"{profile['P']:.1f}")
            col3.metric("Rata-rata K", f"{profile['K']:.1f}")
            col4.metric("Rata-rata pH", f"{profile['ph']:.2f}")

        st.markdown("---")

        # Hasil classification
        st.markdown("### 🌱 Rekomendasi Tanaman")
        col1, col2 = st.columns(2)

        with col1:
            st.info(f"""
            **🌲 Random Forest**
            
            Tanaman yang direkomendasikan:
            # {pred_rf.upper()}
            """)

        with col2:
            st.info(f"""
            **👥 K-Nearest Neighbors**
            
            Tanaman yang direkomendasikan:
            # {pred_knn.upper()}
            """)

        # Apakah kedua model sepakat
        st.markdown("---")
        if pred_rf == pred_knn:
            st.success(f"✅ Kedua model sepakat — Tanaman terbaik untuk lahan kamu adalah **{pred_rf.upper()}**")
        else:
            st.warning(f"⚠️ Kedua model berbeda pendapat — Random Forest merekomendasikan **{pred_rf.upper()}** "
                      f"sedangkan KNN merekomendasikan **{pred_knn.upper()}**. "
                      f"Disarankan mengikuti hasil **Random Forest** karena akurasinya lebih tinggi.")

        # Tanaman lain di cluster yang sama
        if top_crops:
            st.markdown("---")
            st.markdown("### 🌾 Tanaman Lain yang Cocok di Tipe Lahan Ini")
            cols = st.columns(len(top_crops))
            for i, crop in enumerate(top_crops):
                cols[i].success(f"🌱 {crop.upper()}")

# Page Visualisasi
def show_visualization():
    st.title("📈 Visualization")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📊 EDA", "🗺️ Clustering", "🎯 Classification"])

    # ── TAB 1 — EDA ──────────────────────────────────────────
    with tab1:
        st.markdown("### Distribusi Fitur")
        st.image('fig_distribusi_fitur.png', use_container_width=True)
        st.markdown("---")

        st.markdown("### Heatmap Korelasi Antar Fitur")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image('fig_korelasi.png', use_container_width=True)
        with col2:
            st.markdown("""
            **Insight:**
            - Hampir semua fitur tidak berkorelasi tinggi satu sama lain
            - Setiap fitur membawa informasi unik dan tidak redundan
            - Tidak ada fitur yang perlu di-drop
            """)
        st.markdown("---")

        st.markdown("### Rata-rata N, P, K per Tanaman")
        st.image('fig_npk.png', use_container_width=True)
        st.markdown("---")

        st.markdown("### Distribusi Kondisi Iklim & Tanah per Tanaman")
        st.image('fig_boxplot_iklim.png', use_container_width=True)

    # ── TAB 2 — CLUSTERING ───────────────────────────────────
    with tab2:
        st.markdown("### Penentuan Jumlah Cluster Optimal")
        st.image('fig_elbow.png', use_container_width=True)
        col1, col2 = st.columns(2)
        with col1:
            st.info("**Elbow Method** — Penurunan inertia mulai melambat di sekitar k=5–6")
        with col2:
            st.info("**Silhouette Score** — Skor stabil dan naik mulai k=7 (0.33)")
        st.markdown("---")

        st.markdown("### Visualisasi Cluster (PCA 2D)")
        st.image('fig_cluster_pca.png', use_container_width=True)
        st.markdown("---")

        st.markdown("### Profil Rata-rata Fitur per Cluster (Radar Chart)")
        st.image('fig_radar_chart.png', use_container_width=True)
        st.markdown("---")

        st.markdown("### Profil Cluster")
        cluster_profile = df.groupby('cluster_name')[
            ['N','P','K','temperature','humidity','ph','rainfall']
        ].mean().round(2)
        st.dataframe(cluster_profile, use_container_width=True)

    # ── TAB 3 — CLASSIFICATION ───────────────────────────────
    with tab3:
        st.markdown("### Perbandingan Akurasi Model")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🌲 Random Forest", "99.55%", help="Akurasi pada data test")
        with col2:
            st.metric("👥 KNN", "97.95%", help="Akurasi pada data test")

        st.markdown("---")

        st.markdown("### Confusion Matrix")
        st.image('fig_confusion_matrix.png', use_container_width=True)
        st.markdown("---")

        st.markdown("### Feature Importance — Random Forest")
        st.image('fig_feature_importance.png', use_container_width=True)
        col1, col2 = st.columns(2)
        with col1:
            st.success("Fitur dengan importance tinggi → paling berpengaruh dalam rekomendasi tanaman")
        with col2:
            st.warning("Fitur dengan importance rendah → kontribusinya kecil tapi tetap digunakan model")

        st.markdown("---")

        st.markdown("### KNN — Cross Validation")
        st.image('fig_knn_cv.png', use_container_width=True)

# == Page About ==
def show_about():
    st.title("ℹ️ About")
    st.markdown("---")

    # Tentang proyek
    st.markdown("### 📌 Tentang Proyek")
    st.markdown("""
    Proyek ini merupakan implementasi **Data Mining** untuk menyelesaikan permasalahan 
    nyata di bidang pertanian — yaitu membantu petani menentukan jenis tanaman yang paling 
    optimal berdasarkan kondisi lahan dan iklim secara otomatis berbasis data.
    """)

    st.markdown("---")

    # Tentang dataset
    st.markdown("### 📂 Dataset")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Nama Dataset**
        Crop Recommendation Dataset
        
        **Sumber**
        Kaggle — Atharva Ingle
        
        **Jumlah Data**
        2.200 records
        
        **Jumlah Fitur**
        7 fitur input + 1 label
        """)
    with col2:
        st.markdown("""
        **Jenis Tanaman**
        22 jenis tanaman
        
        **Missing Values**
        Tidak ada
        
        **Distribusi Data**
        Perfectly balanced (100 data/tanaman)
        
        **Link Dataset**
        [Kaggle — Crop Recommendation](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)
        """)

    st.markdown("---")

    # Tentang metode
    st.markdown("### 🤖 Metode yang Digunakan")
    col1, col2 = st.columns(2)

    with col1:
        st.info("""
        **K-Means Clustering**
        
        Mengelompokkan data lahan ke dalam 7 tipe 
        berdasarkan kemiripan karakteristik N, P, K, 
        suhu, kelembaban, pH, dan curah hujan.
        
        - Jumlah cluster : 7
        - Silhouette Score : 0.33
        - Penentuan k : Elbow Method + Silhouette Score
        """)

        st.success("""
        **Random Forest**
        
        Ensemble model yang membangun 100 pohon 
        keputusan dan menggabungkan hasilnya melalui 
        voting untuk menghasilkan rekomendasi tanaman.
        
        - Jumlah pohon : 100
        - Akurasi : 99.55%
        """)

    with col2:
        st.warning("""
        **K-Nearest Neighbors (KNN)**
        
        Mengklasifikasikan data baru berdasarkan 
        kemiripan dengan k tetangga terdekat di 
        ruang fitur yang telah dinormalisasi.
        
        - Nilai k : 5
        - Akurasi : 97.95%
        - Normalisasi : StandardScaler
        """)

        st.markdown("""
        **Preprocessing yang Dilakukan:**
        - ✅ Label Encoding
        - ✅ Normalisasi (StandardScaler)
        - ✅ Train-Test Split (80:20)
        - ✅ Stratified Sampling
        """)

    st.markdown("---")

# ── Routing halaman ──────────────────────────────────────────
if page == "🏠 Home":
    show_home()
elif page == "📊 Dataset Overview":
    show_dataset()
elif page == "🔍 Analysis":
    show_analysis()
elif page == "📈 Visualization":
    show_visualization()
elif page == "ℹ️ About":
    show_about()
