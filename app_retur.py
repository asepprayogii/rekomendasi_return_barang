import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
import json
import scipy.sparse as sp
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    from openai import OpenAI
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

# ─── CONFIG ───────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sistem Klasifikasi Retur",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px 30px;
        border-radius: 12px;
        color: white;
        margin-bottom: 25px;
    }
    .stTabs [data-baseweb="tab"] { font-size: 15px; font-weight: 600; }
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

# ─── MODEL ────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(path='model_retur.pkl'):
    try:
        return joblib.load(path)
    except:
        return None

# ─── TF-IDF CLASSIFY ──────────────────────────────────────────────────────────
def classify_single(row, bundle):
    alasan = str(row.get('Alasan Pengembalian', ''))
    catatan_raw = row.get('Catatan Pengembalian Barang')
    catatan = str(catatan_raw) if pd.notna(catatan_raw) else ''
    catatan_lower = catatan.lower().strip()

    ada_lampiran = int(any(kw in catatan_lower for kw in KEYWORD_LAMPIRAN))
    catatan_kosong = int(catatan_lower == '')
    alasan_butuh_bukti_flag = int(alasan in ALASAN_BUTUH_BUKTI)
    perlu_dicek = ada_lampiran or (alasan_butuh_bukti_flag and catatan_kosong)

    if perlu_dicek:
        note = "Catatan menyebut lampiran visual" if ada_lampiran else "Alasan butuh bukti visual, catatan kosong"
        return 'PERLU DICEK', None, note

    try:
        tgl_pesan = pd.to_datetime(row.get('Tanggal Pesanan Dibuat'))
        tgl_retur = pd.to_datetime(row.get('Waktu Pengembalian Diajukan'))
        selisih = max(0, (tgl_retur - tgl_pesan).days)
    except:
        selisih = 0

    tipe = str(row.get('Tipe Pengembalian', 'Seluruh Pesanan'))
    teks = (alasan + ' ' + catatan_lower).lower()

    X_tfidf = bundle['tfidf'].transform([teks])
    X_num = np.array([[
        selisih,
        ALASAN_MAP.get(alasan, 0),
        len(catatan_lower),
        int(tipe == 'Seluruh Pesanan'),
        alasan_butuh_bukti_flag
    ]])
    X = sp.hstack([X_tfidf, sp.csr_matrix(X_num)])
    pred = bundle['model'].predict(X)[0]
    proba = bundle['model'].predict_proba(X)[0].max()
    label = bundle['label_encoder'].inverse_transform([pred])[0]
    return label, round(proba * 100, 1), f"Confidence: {proba:.1%}"

# ─── GROQ AI ──────────────────────────────────────────────────────────────────
def build_prompt(alasan, catatan, selisih_hari):
    return f"""Kamu adalah sistem AI untuk memvalidasi pengajuan retur di marketplace Indonesia.
Bahasa pembeli bisa campur: formal, slang, atau Inggris — tetap pahami maknanya.

Tugasmu: Tentukan label pengajuan retur: VALID, TIDAK VALID, atau PERLU DICEK.

DATA PENGAJUAN:
- Alasan Pengembalian: {alasan}
- Catatan dari Pembeli: "{catatan if catatan else '(kosong)'}"
- Hari sejak pesanan dibuat: {selisih_hari} hari

KRITERIA VALID (jika yakin):
- Alasan jelas dan konsisten dengan catatan
- Barang bermasalah nyata: cacat, salah kirim, tidak sampai, tidak sesuai deskripsi
- Waktu pengajuan wajar, catatan mendukung alasan secara logis

KRITERIA TIDAK VALID (jika yakin):
- Alasan tidak konsisten atau bertentangan dengan catatan
- Terindikasi menyalahgunakan sistem retur
- Catatan kosong padahal alasan serius yang butuh penjelasan
- Alasan terlalu umum/tidak spesifik untuk barang bernilai tinggi
- Ada indikasi tekanan, ancaman, atau manipulasi

KRITERIA PERLU DICEK (jika tidak cukup yakin):
- Catatan menyebut ada foto, video, bukti, atau lampiran yang perlu diverifikasi
- Alasan dan catatan tidak konsisten tapi tidak jelas siapa yang salah
- Informasi tidak cukup untuk memutuskan valid atau tidak
- Ada kejanggalan yang perlu dikonfirmasi ke pembeli atau penjual

Jawab HANYA dalam format JSON berikut, tanpa teks lain:
{{
  "label": "VALID" atau "TIDAK VALID" atau "PERLU DICEK",
  "pct_valid": <angka 0-100>,
  "pct_tidak_valid": <angka 0-100>,
  "pct_perlu_dicek": <angka 0-100>,
  "alasan": "penjelasan singkat max 20 kata dalam bahasa Indonesia"
}}
Pastikan pct_valid + pct_tidak_valid + pct_perlu_dicek = 100"""

def parse_ai_response(raw):
    try:
        raw = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(raw)
        label = result.get("label", "PERLU DICEK")
        alasan = result.get("alasan", "-")
        if label not in ["VALID", "TIDAK VALID", "PERLU DICEK"]:
            label = "PERLU DICEK"
        pct_valid = int(result.get("pct_valid", 0))
        pct_tidak = int(result.get("pct_tidak_valid", 0))
        pct_perlu = int(result.get("pct_perlu_dicek", 0))
        total = pct_valid + pct_tidak + pct_perlu
        if total > 0 and total != 100:
            pct_valid = round(pct_valid / total * 100)
            pct_tidak = round(pct_tidak / total * 100)
            pct_perlu = 100 - pct_valid - pct_tidak
        conf_str = f"VALID {pct_valid}% | TIDAK VALID {pct_tidak}% | PERLU DICEK {pct_perlu}%"
        return label, alasan, conf_str
    except json.JSONDecodeError:
        return "PERLU DICEK", "Response AI tidak valid, perlu cek manual", "-"

def analyze_with_groq(alasan, catatan, selisih_hari, api_key):
    if not GROQ_AVAILABLE:
        return "PERLU DICEK", "Library openai tidak terinstall", "-"
    try:
        client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": build_prompt(alasan, catatan, selisih_hari)}],
            temperature=0,
            max_tokens=200,
        )
        return parse_ai_response(response.choices[0].message.content.strip())
    except Exception as e:
        return "PERLU DICEK", f"Error Groq: {str(e)[:60]}", "-"

# ─── CLASSIFY DATAFRAME ───────────────────────────────────────────────────────
def classify_dataframe(df, bundle, use_ai=False, api_key=None, progress_bar=None):
    labels, confidences, notes, tfidf_refs, ai_used = [], [], [], [], []
    total = len(df)

    for i, (_, row) in enumerate(df.iterrows()):
        tfidf_label, tfidf_conf, tfidf_note = classify_single(row, bundle)

        if use_ai and api_key:
            alasan = str(row.get('Alasan Pengembalian', ''))
            catatan_raw = row.get('Catatan Pengembalian Barang')
            catatan = str(catatan_raw) if pd.notna(catatan_raw) else ''
            try:
                selisih = max(0, (pd.to_datetime(row.get('Waktu Pengembalian Diajukan')) -
                                  pd.to_datetime(row.get('Tanggal Pesanan Dibuat'))).days)
            except:
                selisih = 0
            ai_label, ai_note, ai_conf = analyze_with_groq(alasan, catatan, selisih, api_key)
            labels.append(ai_label)
            confidences.append(ai_conf)
            notes.append(f'[Groq] {ai_note}')
            tfidf_refs.append(f'{tfidf_label} ({tfidf_conf}%)')
            ai_used.append('Ya')
        else:
            labels.append(tfidf_label)
            confidences.append(f'{tfidf_conf}%' if tfidf_conf else '-')
            notes.append(tfidf_note)
            tfidf_refs.append('-')
            ai_used.append('-')

        if progress_bar:
            progress_bar.progress((i + 1) / total, text=f"Memproses {i+1}/{total} baris...")

    df = df.copy()
    df['Rekomendasi']      = labels
    df['AI Confidence']    = confidences
    df['Alasan AI']        = notes
    df['TF-IDF Referensi'] = tfidf_refs
    df['AI Digunakan']     = ai_used
    return df

# ─── SUSPICIOUS CUSTOMERS ─────────────────────────────────────────────────────
def analyze_suspicious_customers(df, bundle):
    from sklearn.ensemble import IsolationForest

    EMPTY_COLS = ['Username (Pembeli)', 'total_retur', 'avg_selisih_hari',
                  'pct_tidak_valid', 'alasan_unik', 'pct_catatan_kosong',
                  'is_suspicious', 'risk_score']

    df = df.copy()
    df['Catatan Pengembalian Barang'] = df['Catatan Pengembalian Barang'].fillna('')
    df['catatan_lower'] = df['Catatan Pengembalian Barang'].str.lower()
    df['catatan_kosong'] = (df['catatan_lower'] == '').astype(int)

    try:
        df['Tanggal Pesanan Dibuat'] = pd.to_datetime(df['Tanggal Pesanan Dibuat'], errors='coerce')
        df['Waktu Pengembalian Diajukan'] = pd.to_datetime(df['Waktu Pengembalian Diajukan'], errors='coerce')
        df['selisih_hari'] = (df['Waktu Pengembalian Diajukan'] - df['Tanggal Pesanan Dibuat']).dt.days.fillna(0)
    except:
        df['selisih_hari'] = 0

    label_map = {}
    if bundle and 'customer_stats' in bundle:
        for _, row in bundle['customer_stats'].iterrows():
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
        return multi, pd.DataFrame(columns=EMPTY_COLS)

    iso_features = ['total_retur', 'avg_selisih_hari', 'pct_tidak_valid', 'alasan_unik', 'pct_catatan_kosong']
    X_iso = multi[iso_features].fillna(0).values
    iso = IsolationForest(contamination=0.1, random_state=42)
    multi['is_suspicious'] = (iso.fit_predict(X_iso) == -1).astype(int)
    multi['risk_score'] = -iso.score_samples(X_iso)

    suspicious = multi[multi['is_suspicious'] == 1].sort_values('risk_score', ascending=False).reset_index(drop=True)
    return multi, suspicious

def df_to_excel_bytes(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Hasil Klasifikasi')
    return output.getvalue()

# ─── MAIN APP ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1 style="margin:0; font-size:26px;">Sistem Klasifikasi Retur Otomatis</h1>
    <p style="margin:6px 0 0 0; opacity:0.85; font-size:14px;">
        Upload file Excel retur — sistem menambah kolom rekomendasi secara otomatis
    </p>
</div>
""", unsafe_allow_html=True)

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Pengaturan Model")
    model_path = st.text_input("Path model (.pkl)", value="model_retur.pkl")
    bundle = load_model(model_path)

    if bundle:
        st.success("Model berhasil dimuat")
        st.info(f"Model: {bundle.get('model_name', 'N/A')}")
    else:
        st.error("Model tidak ditemukan. Jalankan notebook terlebih dahulu.")

    st.divider()
    st.markdown("### Groq AI Analysis")

    # Ambil API key dari Streamlit Secrets
    groq_api_key = st.secrets.get("GROQ_API_KEY", "") if hasattr(st, 'secrets') else ""

    if groq_api_key:
        st.success("API Key Groq aktif dari Secrets")
        use_ai = st.toggle("Aktifkan Groq AI", value=True)
    else:
        st.warning("GROQ_API_KEY belum diset di Streamlit Secrets.")
        st.caption("Cara set: Settings > Secrets > tambah GROQ_API_KEY = 'gsk_...'")
        use_ai = False

    st.divider()
    st.markdown("""
    ### Panduan Label
    | Label | Arti |
    |-------|------|
    | VALID | Pengajuan layak diterima |
    | TIDAK VALID | Pengajuan ditolak |
    | PERLU DICEK | Butuh verifikasi manual |

    ### Alur Klasifikasi
    **AI aktif:** Groq analisis semua baris. TF-IDF sebagai referensi.

    **AI tidak aktif:** TF-IDF menentukan label.
    """)

# ─── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Klasifikasi File", "Deteksi Customer Nakal", "Test Manual"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Upload File Retur")
    uploaded_file = st.file_uploader(
        "Upload file Excel retur (.xlsx)",
        type=['xlsx', 'xls'],
        help="Format kolom harus sama dengan file training"
    )

    if uploaded_file and bundle:
        df_input = pd.read_excel(uploaded_file)
        st.success(f"File dimuat: {df_input.shape[0]} baris")

        with st.expander("Preview Data (5 baris pertama)"):
            st.dataframe(df_input.head(), use_container_width=True)

        if use_ai and groq_api_key:
            st.info(f"Groq AI aktif — semua {len(df_input)} baris akan dianalisis secara semantik")
        else:
            st.caption("Aktifkan Groq AI di Streamlit Secrets untuk analisis semantik")

        # Reset session state jika file baru
        file_key = f"{uploaded_file.name}_{len(df_input)}"
        if st.session_state.get('last_file') != file_key:
            st.session_state['df_result'] = None
            st.session_state['last_file'] = file_key

        if st.button("Jalankan Klasifikasi", type="primary", use_container_width=True):
            progress_bar = st.progress(0, text="Memulai klasifikasi...")
            st.session_state['df_result'] = classify_dataframe(
                df_input, bundle,
                use_ai=use_ai,
                api_key=groq_api_key if use_ai else None,
                progress_bar=progress_bar
            )
            progress_bar.empty()

        if st.session_state.get('df_result') is not None:
            df_result = st.session_state['df_result']

            st.markdown("---")
            st.markdown("### Hasil Klasifikasi")

            counts = df_result['Rekomendasi'].value_counts()
            n_valid   = counts.get('VALID', 0)
            n_invalid = counts.get('TIDAK VALID', 0)
            n_perlu   = counts.get('PERLU DICEK', 0)
            n_ai      = (df_result['AI Digunakan'] == 'Ya').sum()
            total     = len(df_result)

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Total", total)
            c2.metric("Valid", n_valid, delta=f"{n_valid/total*100:.1f}%")
            c3.metric("Tidak Valid", n_invalid, delta=f"-{n_invalid/total*100:.1f}%", delta_color="inverse")
            c4.metric("Perlu Dicek", n_perlu, delta_color="off")
            c5.metric("Dianalisis AI", n_ai, delta_color="off")

            if n_ai > 0:
                st.success(f"{n_ai} baris dianalisis Groq AI secara semantik")

            # Pie chart
            import plotly.graph_objects as go
            fig_pie = go.Figure(go.Pie(
                labels=counts.index.tolist(),
                values=counts.values.tolist(),
                marker_colors=[
                    {'VALID': '#2ecc71', 'TIDAK VALID': '#e74c3c', 'PERLU DICEK': '#f39c12'}.get(k, '#95a5a6')
                    for k in counts.index
                ],
                hole=0.35,
                textinfo='label+percent'
            ))
            fig_pie.update_layout(
                title='Distribusi Rekomendasi',
                height=320,
                margin=dict(t=50, b=20, l=20, r=20)
            )
            st.plotly_chart(fig_pie, use_container_width=False)

            # Filter
            st.markdown("#### Filter Hasil")
            col_f1, col_f2 = st.columns([1, 2])
            with col_f1:
                filter_mode = st.radio(
                    "Tampilkan:",
                    options=["Semua", "VALID", "TIDAK VALID", "PERLU DICEK"],
                )
            df_filtered = df_result if filter_mode == "Semua" else df_result[df_result['Rekomendasi'] == filter_mode]
            with col_f2:
                st.metric("Baris ditampilkan", len(df_filtered))

            display_cols = [
                'No. Pengembalian', 'Username (Pembeli)', 'Alasan Pengembalian',
                'Catatan Pengembalian Barang', 'Rekomendasi',
                'AI Confidence', 'Alasan AI', 'TF-IDF Referensi'
            ]
            display_cols = [c for c in display_cols if c in df_filtered.columns]

            def color_label(val):
                if val == 'VALID':       return 'background-color: #d4edda; color: #155724; font-weight:600'
                elif val == 'TIDAK VALID': return 'background-color: #f8d7da; color: #721c24; font-weight:600'
                elif val == 'PERLU DICEK': return 'background-color: #fff3cd; color: #856404; font-weight:600'
                return ''

            st.dataframe(
                df_filtered[display_cols].style.applymap(color_label, subset=['Rekomendasi']),
                use_container_width=True,
                height=400
            )

            st.markdown("---")
            st.download_button(
                label="Download Hasil Klasifikasi (.xlsx)",
                data=df_to_excel_bytes(df_result),
                file_name=f"hasil_klasifikasi_retur_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                type="primary"
            )

    elif uploaded_file and not bundle:
        st.warning("Model belum dimuat. Latih model di notebook terlebih dahulu.")
    else:
        st.info("Upload file Excel retur untuk memulai klasifikasi.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Deteksi Customer Mencurigakan")
    st.markdown(
        "Sistem menganalisis pola retur per customer menggunakan **Isolation Forest** "
        "untuk mendeteksi customer yang memiliki pola retur tidak wajar."
    )

    uploaded_file_c = st.file_uploader(
        "Upload file Excel retur untuk analisis customer",
        type=['xlsx', 'xls'],
        key="customer_upload"
    )

    if uploaded_file_c:
        df_c = pd.read_excel(uploaded_file_c)
        st.success(f"File dimuat: {df_c.shape[0]} baris, {df_c['Username (Pembeli)'].nunique()} customer unik")

        min_retur = st.slider("Minimum jumlah retur per customer", min_value=2, max_value=10, value=2)

        if st.button("Analisis Customer", type="primary", use_container_width=True):
            with st.spinner("Menganalisis pola retur customer..."):
                multi, suspicious = analyze_suspicious_customers(df_c, bundle)

            n_total   = df_c['Username (Pembeli)'].nunique()
            n_multi   = len(multi)
            n_suspect = len(suspicious)

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Customer", n_total)
            c2.metric("Customer Multi-Retur", n_multi)
            c3.metric("Terdeteksi Mencurigakan", n_suspect)

            if len(suspicious) > 0:
                st.markdown("### Daftar Customer Mencurigakan")
                st.caption(
                    "Customer di bawah ini memiliki pola retur anomali berdasarkan kombinasi: "
                    "frekuensi tinggi, % ditolak tinggi, alasan berulang, atau catatan kosong konsisten."
                )

                suspicious_filtered = suspicious[suspicious['total_retur'] >= min_retur].copy()

                if len(suspicious_filtered) > 0:
                    display_suspicious = suspicious_filtered[[
                        'Username (Pembeli)', 'total_retur', 'avg_selisih_hari',
                        'pct_tidak_valid', 'alasan_unik', 'pct_catatan_kosong', 'risk_score'
                    ]].copy()
                    display_suspicious.columns = [
                        'Username', 'Total Retur', 'Avg Hari Pengajuan',
                        '% Retur Ditolak', 'Variasi Alasan', '% Catatan Kosong', 'Risk Score'
                    ]
                    display_suspicious['Avg Hari Pengajuan'] = display_suspicious['Avg Hari Pengajuan'].round(1)
                    display_suspicious['% Retur Ditolak']   = display_suspicious['% Retur Ditolak'].round(2)
                    display_suspicious['% Catatan Kosong']  = display_suspicious['% Catatan Kosong'].round(2)
                    display_suspicious['Risk Score']        = display_suspicious['Risk Score'].round(3)

                    import plotly.graph_objects as go
                    max_risk = display_suspicious['Risk Score'].max()
                    min_risk = display_suspicious['Risk Score'].min()

                    def risk_to_color(val):
                        ratio = (val - min_risk) / (max_risk - min_risk) if max_risk != min_risk else 1.0
                        g = int(255 * (1 - ratio * 0.8))
                        return f'rgb(255,{g},{g})'

                    n_rows = len(display_suspicious)
                    cell_colors = [
                        ['#1e1e2e'] * n_rows,
                        ['#1e1e2e'] * n_rows,
                        ['#1e1e2e'] * n_rows,
                        ['#1e1e2e'] * n_rows,
                        ['#1e1e2e'] * n_rows,
                        ['#1e1e2e'] * n_rows,
                        [risk_to_color(v) for v in display_suspicious['Risk Score']],
                    ]
                    font_colors = [['white']*n_rows]*6 + [['#1a1a1a']*n_rows]

                    fig_tbl = go.Figure(go.Table(
                        columnwidth=[2, 1, 1.2, 1.2, 1.2, 1.5, 1],
                        header=dict(
                            values=[f'<b>{c}</b>' for c in display_suspicious.columns],
                            fill_color='#4a4aff',
                            font=dict(color='white', size=13),
                            align='center', height=38
                        ),
                        cells=dict(
                            values=[display_suspicious[c].tolist() for c in display_suspicious.columns],
                            fill_color=cell_colors,
                            font=dict(color=font_colors, size=12),
                            align='center', height=32
                        )
                    ))
                    fig_tbl.update_layout(
                        margin=dict(t=10, b=10, l=0, r=0),
                        height=min(500, 90 + n_rows * 35),
                        paper_bgcolor='rgba(0,0,0,0)',
                    )
                    st.plotly_chart(fig_tbl, use_container_width=True)

                    st.download_button(
                        "Download Daftar Customer Mencurigakan",
                        data=df_to_excel_bytes(suspicious_filtered),
                        file_name=f"customer_mencurigakan_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

                    st.markdown("#### Detail Retur per Customer")
                    selected_customer = st.selectbox(
                        "Pilih customer untuk lihat detail:",
                        options=suspicious_filtered['Username (Pembeli)'].tolist()
                    )
                    if selected_customer:
                        detail_cols = [c for c in [
                            'No. Pengembalian', 'Nama Produk', 'Alasan Pengembalian',
                            'Catatan Pengembalian Barang', 'Status Pembatalan/ Pengembalian',
                            'Tanggal Pesanan Dibuat', 'Waktu Pengembalian Diajukan'
                        ] if c in df_c.columns]
                        st.dataframe(
                            df_c[df_c['Username (Pembeli)'] == selected_customer][detail_cols],
                            use_container_width=True
                        )
                else:
                    st.info("Tidak ada customer mencurigakan dengan kriteria minimum retur tersebut.")
            else:
                st.info("Tidak ada customer mencurigakan yang terdeteksi.")
    else:
        st.info("Upload file Excel untuk memulai analisis customer.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Test Klasifikasi Manual")
    st.markdown("Input satu pengajuan retur secara manual untuk melihat hasil prediksi.")

    if not bundle:
        st.warning("Model belum dimuat. Latih model di notebook terlebih dahulu.")
    else:
        col_a, col_b = st.columns(2)
        with col_a:
            alasan_input = st.selectbox("Alasan Pengembalian", options=list(ALASAN_MAP.keys()))
            tgl_pesanan  = st.date_input("Tanggal Pesanan Dibuat", value=datetime(2024, 1, 15))
            tgl_retur    = st.date_input("Waktu Pengembalian Diajukan", value=datetime(2024, 1, 20))
        with col_b:
            catatan_input = st.text_area(
                "Catatan Pengembalian Barang",
                placeholder="Contoh: barang rusak, ada foto bukti kerusakan...",
                height=120
            )
            tipe_input = st.selectbox("Tipe Pengembalian", ['Seluruh Pesanan', 'Sebagian Pesanan'])

        if st.button("Prediksi", type="primary", use_container_width=True):
            row = {
                'Alasan Pengembalian': alasan_input,
                'Catatan Pengembalian Barang': catatan_input,
                'Tanggal Pesanan Dibuat': str(tgl_pesanan),
                'Waktu Pengembalian Diajukan': str(tgl_retur),
                'Tipe Pengembalian': tipe_input
            }
            label, conf, note = classify_single(row, bundle)

            st.markdown("---")
            st.markdown("#### Hasil TF-IDF (Referensi)")
            if label == 'VALID':
                st.success(f"VALID — {note}")
            elif label == 'TIDAK VALID':
                st.error(f"TIDAK VALID — {note}")
            else:
                st.warning(f"PERLU DICEK — {note}")

            if use_ai and groq_api_key:
                st.markdown("#### Hasil Groq AI")
                if st.button("Analisis dengan Groq", type="secondary", use_container_width=True):
                    with st.spinner("Groq sedang menganalisis..."):
                        selisih_tes = max(0, (tgl_retur - tgl_pesanan).days)
                        ai_label, ai_note, ai_conf = analyze_with_groq(
                            alasan_input, catatan_input, selisih_tes, groq_api_key
                        )
                    if ai_label == 'VALID':
                        st.success(f"VALID\n\nKonfidensitas: {ai_conf}\n\n{ai_note}")
                    elif ai_label == 'TIDAK VALID':
                        st.error(f"TIDAK VALID\n\nKonfidensitas: {ai_conf}\n\n{ai_note}")
                    else:
                        st.warning(f"PERLU DICEK\n\nKonfidensitas: {ai_conf}\n\n{ai_note}")
                    if ai_label != label:
                        st.info("Hasil AI berbeda dengan TF-IDF — AI lebih akurat karena membaca makna teks")
            else:
                st.caption("Set GROQ_API_KEY di Streamlit Secrets untuk analisis semantik")

            catatan_lower = catatan_input.lower().strip()
            keywords_found = [kw for kw in KEYWORD_LAMPIRAN if kw in catatan_lower]
            alasan_butuh = alasan_input in ALASAN_BUTUH_BUKTI

            with st.expander("Detail Analisis"):
                selisih = (tgl_retur - tgl_pesanan).days
                st.markdown(f"""
                | Faktor | Nilai |
                |--------|-------|
                | Selisih Hari Pesanan ke Retur | **{selisih} hari** |
                | Alasan Butuh Bukti Visual | **{'Ya' if alasan_butuh else 'Tidak'}** |
                | Keyword Lampiran di Catatan | **{', '.join(keywords_found) if keywords_found else 'Tidak ada'}** |
                | Catatan Kosong | **{'Ya' if not catatan_lower else 'Tidak'}** |
                """)

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#888; font-size:12px;'>"
    "Sistem Klasifikasi Retur Otomatis — Powered by Random Forest + Groq AI"
    "</div>",
    unsafe_allow_html=True
)