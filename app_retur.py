import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
import scipy.sparse as sp
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ─── CONFIG ───────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sistem Klasifikasi Retur",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── STYLE ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px 30px;
        border-radius: 12px;
        color: white;
        margin-bottom: 25px;
    }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 15px 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 4px solid #667eea;
        margin-bottom: 10px;
    }
    .valid-badge {
        background: #d4edda;
        color: #155724;
        padding: 3px 10px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 13px;
    }
    .invalid-badge {
        background: #f8d7da;
        color: #721c24;
        padding: 3px 10px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 13px;
    }
    .check-badge {
        background: #fff3cd;
        color: #856404;
        padding: 3px 10px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 13px;
    }
    .suspicious-badge {
        background: #f8d7da;
        color: #721c24;
        padding: 3px 10px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 13px;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 15px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
KEYWORD_LAMPIRAN = [
    'foto', 'video', 'bukti', 'lampir', 'gambar', 'rekam', 'ss', 'screenshot',
    'tangkap layar', 'dokumentasi', 'unboxing', 'capture', 'file', 'attachment'
]

ALASAN_BUTUH_BUKTI = [
    'Barang cacat produksi (luntur/lecet/patah/dsb.)',
    'Barang tidak berfungsi/tidak bisa dipakai',
    'Produk pecah/hancur',
    'Produk yang diterima berbeda dengan deskripsi.',
    'Cairan/isinya tumpah',
    'Outer packaging damaged',
    'Produk tidak original',
]

ALASAN_MAP = {
    'Pembeli tidak menerima pesanan.': 1,
    'Ingin kembalikan barang sesuai kondisi awal': 2,
    'Barang cacat produksi (luntur/lecet/patah/dsb.)': 3,
    'Barang tidak berfungsi/tidak bisa dipakai': 4,
    'Produk tidak lengkap': 5,
    'Pembeli menerima produk yang salah (contoh: salah ukuran, salah warna, beda produk).': 6,
    'Produk pecah/hancur': 7,
    'Produk yang diterima berbeda dengan deskripsi.': 8,
    'Outer packaging damaged': 9,
    'Cairan/isinya tumpah': 10,
    'Produk tidak original': 11,
}

# ─── HELPER FUNCTIONS ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model(path='model_retur.pkl'):
    try:
        bundle = joblib.load(path)
        return bundle
    except Exception as e:
        return None

def classify_single(row, bundle):
    """Klasifikasi satu baris pengajuan retur"""
    alasan = str(row.get('Alasan Pengembalian', ''))
    catatan = str(row.get('Catatan Pengembalian Barang', '')) if pd.notna(row.get('Catatan Pengembalian Barang')) else ''
    catatan_lower = catatan.lower().strip()

    # Hitung fitur
    ada_lampiran = int(any(kw in catatan_lower for kw in KEYWORD_LAMPIRAN))
    catatan_kosong = int(catatan_lower == '')
    alasan_butuh_bukti_flag = int(alasan in ALASAN_BUTUH_BUKTI)
    perlu_dicek = ada_lampiran or (alasan_butuh_bukti_flag and catatan_kosong)

    if perlu_dicek:
        if ada_lampiran:
            alasan_note = f"Catatan menyebut lampiran visual"
        else:
            alasan_note = f"Alasan butuh bukti visual, catatan kosong"
        return 'PERLU DICEK', None, alasan_note

    # Tanggal
    try:
        tgl_pesan = pd.to_datetime(row.get('Tanggal Pesanan Dibuat'))
        tgl_retur = pd.to_datetime(row.get('Waktu Pengembalian Diajukan'))
        selisih = max(0, (tgl_retur - tgl_pesan).days)
    except:
        selisih = 0

    tipe = str(row.get('Tipe Pengembalian', 'Seluruh Pesanan'))
    teks = (alasan + ' ' + catatan_lower).lower()

    tfidf = bundle['tfidf']
    model = bundle['model']
    le = bundle['label_encoder']
    num_features = bundle['num_features']

    X_tfidf = tfidf.transform([teks])
    X_num = np.array([[
        selisih,
        ALASAN_MAP.get(alasan, 0),
        len(catatan_lower),
        int(tipe == 'Seluruh Pesanan'),
        alasan_butuh_bukti_flag
    ]])
    X = sp.hstack([X_tfidf, sp.csr_matrix(X_num)])
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0].max()
    label = le.inverse_transform([pred])[0]
    return label, round(proba * 100, 1), f"Confidence: {proba:.1%}"

def classify_dataframe(df, bundle):
    """Klasifikasi seluruh dataframe"""
    labels, confidences, notes = [], [], []
    for _, row in df.iterrows():
        label, conf, note = classify_single(row, bundle)
        labels.append(label)
        confidences.append(conf if conf is not None else '-')
        notes.append(note)
    df = df.copy()
    df['🤖 Rekomendasi'] = labels
    df['📊 Confidence (%)'] = confidences
    df['📝 Catatan Sistem'] = notes
    return df

def get_label_badge(label):
    if label == 'VALID':
        return '<span class="valid-badge">✅ VALID</span>'
    elif label == 'TIDAK VALID':
        return '<span class="invalid-badge">❌ TIDAK VALID</span>'
    else:
        return '<span class="check-badge">🔍 PERLU DICEK</span>'

def analyze_suspicious_customers(df, bundle):
    """Analisis customer mencurigakan dari file yang diupload"""
    from sklearn.ensemble import IsolationForest

    df['Catatan Pengembalian Barang'] = df['Catatan Pengembalian Barang'].fillna('')
    df['catatan_lower'] = df['Catatan Pengembalian Barang'].str.lower()
    df['catatan_kosong'] = (df['catatan_lower'] == '').astype(int)

    try:
        df['Tanggal Pesanan Dibuat'] = pd.to_datetime(df['Tanggal Pesanan Dibuat'], errors='coerce')
        df['Waktu Pengembalian Diajukan'] = pd.to_datetime(df['Waktu Pengembalian Diajukan'], errors='coerce')
        df['selisih_hari'] = (df['Waktu Pengembalian Diajukan'] - df['Tanggal Pesanan Dibuat']).dt.days.fillna(0)
    except:
        df['selisih_hari'] = 0

    # Gunakan label dari bundle customer_stats jika ada
    label_map = {}
    if bundle and 'customer_stats' in bundle:
        cs = bundle['customer_stats']
        for _, row in cs.iterrows():
            label_map[row['Username (Pembeli)']] = row.get('pct_tidak_valid', 0)

    customer_stats = df.groupby('Username (Pembeli)').agg(
        total_retur=('No. Pengembalian', 'count'),
        avg_selisih_hari=('selisih_hari', 'mean'),
        alasan_unik=('Alasan Pengembalian', 'nunique'),
        pct_catatan_kosong=('catatan_kosong', 'mean'),
    ).reset_index()
    customer_stats['pct_tidak_valid'] = customer_stats['Username (Pembeli)'].map(label_map).fillna(0)

    multi = customer_stats[customer_stats['total_retur'] >= 2].copy()
    if len(multi) < 5:
        return multi, pd.DataFrame()

    iso_features = ['total_retur', 'avg_selisih_hari', 'pct_tidak_valid', 'alasan_unik', 'pct_catatan_kosong']
    X_iso = multi[iso_features].fillna(0).values
    iso = IsolationForest(contamination=0.1, random_state=42)
    multi['is_suspicious'] = (iso.fit_predict(X_iso) == -1).astype(int)
    multi['risk_score'] = -iso.score_samples(X_iso)

    suspicious = multi[multi['is_suspicious'] == 1].sort_values('risk_score', ascending=False)
    return multi, suspicious

def df_to_excel_bytes(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Hasil Klasifikasi')
    return output.getvalue()

# ─── MAIN APP ─────────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1 style="margin:0; font-size:28px;">🔄 Sistem Klasifikasi Retur Otomatis</h1>
    <p style="margin:5px 0 0 0; opacity:0.85; font-size:15px;">
        Upload file Excel retur → Sistem menambah kolom rekomendasi otomatis
    </p>
</div>
""", unsafe_allow_html=True)

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Pengaturan Model")
    model_path = st.text_input("Path model (.pkl)", value="model_retur.pkl")
    bundle = load_model(model_path)

    if bundle:
        st.success(f"✅ Model loaded")
        st.info(f"**Model:** {bundle.get('model_name', 'N/A')}")
    else:
        st.error("❌ Model tidak ditemukan.\nJalankan notebook terlebih dahulu untuk melatih model.")

    st.divider()
    st.markdown("""
    ### 📖 Panduan Label
    | Label | Arti |
    |-------|------|
    | ✅ **VALID** | Pengajuan layak diterima |
    | ❌ **TIDAK VALID** | Pengajuan ditolak |
    | 🔍 **PERLU DICEK** | Ada lampiran yang perlu diverifikasi manual |

    ### 🔍 Kapan PERLU DICEK?
    - Catatan menyebut: *foto, video, bukti, lampiran, gambar, dll*
    - Alasan butuh bukti visual **tapi catatan kosong**
    """)

# ─── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📂 Klasifikasi File", "🕵️ Deteksi Customer Nakal", "🧪 Test Manual"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: Klasifikasi File
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### 📂 Upload File Retur")
    uploaded_file = st.file_uploader(
        "Upload file Excel retur (.xlsx)",
        type=['xlsx', 'xls'],
        help="Format kolom harus sama dengan file training"
    )

    if uploaded_file and bundle:
        df_input = pd.read_excel(uploaded_file)
        st.success(f"✅ File dimuat: **{df_input.shape[0]} baris** retur")

        # Preview
        with st.expander("👁️ Preview Data Input (5 baris pertama)"):
            st.dataframe(df_input.head(), use_container_width=True)

        # Proses
        if st.button("🚀 Jalankan Klasifikasi", type="primary", use_container_width=True):
            with st.spinner("Memproses klasifikasi..."):
                df_result = classify_dataframe(df_input, bundle)

            st.markdown("---")
            st.markdown("### 📊 Hasil Klasifikasi")

            # Metrics
            counts = df_result['🤖 Rekomendasi'].value_counts()
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("📋 Total Pengajuan", len(df_result))
            col2.metric("✅ Valid", counts.get('VALID', 0),
                        delta=f"{counts.get('VALID',0)/len(df_result)*100:.1f}%")
            col3.metric("❌ Tidak Valid", counts.get('TIDAK VALID', 0),
                        delta=f"-{counts.get('TIDAK VALID',0)/len(df_result)*100:.1f}%", delta_color="inverse")
            col4.metric("🔍 Perlu Dicek", counts.get('PERLU DICEK', 0),
                        delta=f"{counts.get('PERLU DICEK',0)/len(df_result)*100:.1f}%", delta_color="off")

            # Pie chart ringkas
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(5, 3))
            pie_data = counts
            pie_colors = {
                'VALID': '#2ecc71',
                'TIDAK VALID': '#e74c3c',
                'PERLU DICEK': '#f39c12'
            }
            colors = [pie_colors.get(k, '#95a5a6') for k in pie_data.index]
            ax.pie(pie_data.values, labels=pie_data.index, autopct='%1.1f%%',
                   colors=colors, startangle=90)
            ax.set_title('Distribusi Rekomendasi', fontweight='bold')
            st.pyplot(fig, use_container_width=False)
            plt.close()

            # Filter & tampilkan
            st.markdown("#### 🔎 Filter Hasil")
            filter_label = st.multiselect(
                "Tampilkan label:",
                options=['VALID', 'TIDAK VALID', 'PERLU DICEK'],
                default=['VALID', 'TIDAK VALID', 'PERLU DICEK']
            )
            df_filtered = df_result[df_result['🤖 Rekomendasi'].isin(filter_label)]

            # Kolom yang ditampilkan
            display_cols = [
                'No. Pengembalian', 'Username (Pembeli)', 'Alasan Pengembalian',
                'Catatan Pengembalian Barang', 'Tanggal Pesanan Dibuat',
                'Waktu Pengembalian Diajukan', '🤖 Rekomendasi', '📊 Confidence (%)', '📝 Catatan Sistem'
            ]
            display_cols = [c for c in display_cols if c in df_filtered.columns]

            st.dataframe(
                df_filtered[display_cols].style.apply(
                    lambda x: [
                        'background-color: #d4edda' if v == 'VALID'
                        else 'background-color: #f8d7da' if v == 'TIDAK VALID'
                        else 'background-color: #fff3cd' if v == 'PERLU DICEK'
                        else '' for v in x
                    ] if x.name == '🤖 Rekomendasi' else ['' for _ in x],
                    axis=0
                ),
                use_container_width=True,
                height=400
            )

            # Download
            st.markdown("---")
            excel_bytes = df_to_excel_bytes(df_result)
            st.download_button(
                label="⬇️ Download Hasil Klasifikasi (.xlsx)",
                data=excel_bytes,
                file_name=f"hasil_klasifikasi_retur_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                type="primary"
            )

    elif uploaded_file and not bundle:
        st.warning("⚠️ Model belum dimuat. Latih model di notebook terlebih dahulu.")
    elif not uploaded_file:
        st.info("👆 Upload file Excel retur untuk memulai klasifikasi.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: Deteksi Customer Nakal
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 🕵️ Deteksi Customer Mencurigakan")
    st.markdown("""
    Sistem menganalisis pola retur per customer menggunakan **Isolation Forest** untuk mendeteksi
    customer yang memiliki pola retur tidak wajar (terlalu sering, alasan berulang, dll).
    """)

    uploaded_file_c = st.file_uploader(
        "Upload file Excel retur untuk analisis customer",
        type=['xlsx', 'xls'],
        key="customer_upload"
    )

    if uploaded_file_c:
        df_c = pd.read_excel(uploaded_file_c)
        st.success(f"✅ File dimuat: {df_c.shape[0]} baris, {df_c['Username (Pembeli)'].nunique()} customer unik")

        min_retur = st.slider("Minimum jumlah retur untuk dianalisis", 2, 10, 3)

        if st.button("🔍 Analisis Customer", type="primary", use_container_width=True):
            with st.spinner("Menganalisis pola customer..."):
                all_stats, suspicious = analyze_suspicious_customers(df_c, bundle)

            suspicious_filtered = suspicious[suspicious['total_retur'] >= min_retur]

            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            col1.metric("👥 Total Customer", df_c['Username (Pembeli)'].nunique())
            col2.metric("📊 Customer Multi-Retur", len(all_stats[all_stats['total_retur'] >= min_retur]))
            col3.metric("🚨 Terdeteksi Mencurigakan", len(suspicious_filtered))

            if len(suspicious_filtered) > 0:
                st.markdown("#### 🚨 Daftar Customer Mencurigakan")
                st.markdown("""
                > Customer di bawah ini memiliki pola retur yang **anomali** berdasarkan kombinasi:
                > frekuensi tinggi, % ditolak tinggi, alasan berulang, atau catatan kosong konsisten.
                """)

                display_suspicious = suspicious_filtered[[
                    'Username (Pembeli)', 'total_retur', 'avg_selisih_hari',
                    'pct_tidak_valid', 'alasan_unik', 'pct_catatan_kosong', 'risk_score'
                ]].copy()
                display_suspicious.columns = [
                    'Username', 'Total Retur', 'Avg Hari Pengajuan',
                    '% Retur Ditolak', 'Variasi Alasan', '% Catatan Kosong', 'Risk Score'
                ]
                display_suspicious['Avg Hari Pengajuan'] = display_suspicious['Avg Hari Pengajuan'].round(1)
                display_suspicious['% Retur Ditolak'] = (display_suspicious['% Retur Ditolak'] * 100).round(1).astype(str) + '%'
                display_suspicious['% Catatan Kosong'] = (display_suspicious['% Catatan Kosong'] * 100).round(1).astype(str) + '%'
                display_suspicious['Risk Score'] = display_suspicious['Risk Score'].round(3)

                st.dataframe(
                    display_suspicious.style.background_gradient(subset=['Risk Score'], cmap='Reds'),
                    use_container_width=True
                )

                # Download suspicious list
                excel_s = df_to_excel_bytes(suspicious_filtered)
                st.download_button(
                    "⬇️ Download Daftar Customer Mencurigakan",
                    data=excel_s,
                    file_name=f"customer_mencurigakan_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

                # Detail per customer
                st.markdown("#### 🔎 Detail Retur per Customer")
                selected_customer = st.selectbox(
                    "Pilih customer untuk lihat detail retur:",
                    options=suspicious_filtered['Username (Pembeli)'].tolist()
                )
                if selected_customer:
                    detail = df_c[df_c['Username (Pembeli)'] == selected_customer][[
                        'No. Pengembalian', 'Nama Produk', 'Alasan Pengembalian',
                        'Catatan Pengembalian Barang', 'Status Pembatalan/ Pengembalian',
                        'Tanggal Pesanan Dibuat', 'Waktu Pengembalian Diajukan'
                    ]]
                    st.dataframe(detail, use_container_width=True)
            else:
                st.info("✅ Tidak ada customer mencurigakan yang terdeteksi dengan kriteria saat ini.")

    else:
        st.info("👆 Upload file Excel untuk memulai analisis customer.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: Test Manual
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 🧪 Test Klasifikasi Manual")
    st.markdown("Input satu pengajuan retur secara manual untuk melihat hasil prediksi.")

    if not bundle:
        st.warning("⚠️ Model belum dimuat. Latih model di notebook terlebih dahulu.")
    else:
        col_a, col_b = st.columns(2)
        with col_a:
            alasan_options = list(ALASAN_MAP.keys())
            alasan_input = st.selectbox("Alasan Pengembalian", options=alasan_options)
            tgl_pesanan = st.date_input("Tanggal Pesanan Dibuat", value=datetime(2024, 1, 15))
            tgl_retur = st.date_input("Waktu Pengembalian Diajukan", value=datetime(2024, 1, 20))

        with col_b:
            catatan_input = st.text_area(
                "Catatan Pengembalian Barang",
                placeholder="Contoh: barang rusak, ada foto bukti kerusakan...",
                height=120
            )
            tipe_input = st.selectbox("Tipe Pengembalian", ['Seluruh Pesanan', 'Sebagian Pesanan'])

        if st.button("🔍 Prediksi Sekarang", type="primary", use_container_width=True):
            row = {
                'Alasan Pengembalian': alasan_input,
                'Catatan Pengembalian Barang': catatan_input,
                'Tanggal Pesanan Dibuat': str(tgl_pesanan),
                'Waktu Pengembalian Diajukan': str(tgl_retur),
                'Tipe Pengembalian': tipe_input
            }
            label, conf, note = classify_single(row, bundle)

            st.markdown("---")
            st.markdown("#### 🎯 Hasil Prediksi")

            if label == 'VALID':
                st.success(f"✅ **VALID** — Pengajuan retur layak diterima\n\n{note}")
            elif label == 'TIDAK VALID':
                st.error(f"❌ **TIDAK VALID** — Pengajuan retur ditolak\n\n{note}")
            else:
                st.warning(f"🔍 **PERLU DICEK MANUAL** — {note}\n\nTim perlu memverifikasi lampiran yang disebutkan dalam catatan.")

            # Detail reasoning
            catatan_lower = catatan_input.lower().strip()
            keywords_found = [kw for kw in KEYWORD_LAMPIRAN if kw in catatan_lower]
            alasan_butuh = alasan_input in ALASAN_BUTUH_BUKTI

            with st.expander("🔬 Detail Analisis"):
                selisih = (tgl_retur - tgl_pesanan).days
                st.markdown(f"""
                | Faktor | Nilai |
                |--------|-------|
                | Selisih Hari Pesanan → Retur | **{selisih} hari** |
                | Alasan Butuh Bukti Visual | **{'Ya ⚠️' if alasan_butuh else 'Tidak'}** |
                | Keyword Lampiran di Catatan | **{', '.join(keywords_found) if keywords_found else 'Tidak ada'}** |
                | Catatan Kosong | **{'Ya' if not catatan_lower else 'Tidak'}** |
                | Flag Perlu Dicek | **{'Ya 🔍' if (keywords_found or (alasan_butuh and not catatan_lower)) else 'Tidak'}** |
                """)

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#888; font-size:12px;'>"
    "Sistem Klasifikasi Retur Otomatis • Powered by Random Forest + IsolationForest"
    "</div>",
    unsafe_allow_html=True
)
