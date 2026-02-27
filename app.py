"""
============================================================
AI RETURN CLASSIFICATION - STREAMLIT APP
Versi 3: 4 Input, 11 Kategori Resmi, Keyword Fix
============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import io
from scipy.sparse import hstack, csr_matrix
from datetime import datetime, date, timedelta

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================

st.set_page_config(
    page_title="Sistm rekomendasi return barang",
    page_icon="📦",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; padding: 22px 28px; border-radius: 12px;
        margin-bottom: 28px; text-align: center;
    }
    .main-header h1 { margin: 0; font-size: 1.8rem; }
    .main-header p  { margin: 6px 0 0; opacity: 0.85; font-size: 0.95rem; }
    .result-valid   { background:#d4edda; border-left:5px solid #28a745; padding:16px; border-radius:8px; margin:10px 0; }
    .result-invalid { background:#f8d7da; border-left:5px solid #dc3545; padding:16px; border-radius:8px; margin:10px 0; }
    .result-manual  { background:#fff3cd; border-left:5px solid #ffc107; padding:16px; border-radius:8px; margin:10px 0; }
    div.stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; border: none; border-radius: 8px;
        padding: 10px 0; font-weight: 600; font-size: 1rem; width: 100%;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# 11 KATEGORI RESMI SESUAI DATA
# ============================================================

KATEGORI_ALASAN = [
    "Pembeli tidak menerima pesanan.",
    "Barang cacat produksi (luntur/lecet/patah/dsb.)",
    "Cairan/isinya tumpah",
    "Produk pecah/hancur",
    "Ingin kembalikan barang sesuai kondisi awal",
    "Produk tidak lengkap",
    "Pembeli menerima produk yang salah (contoh: salah ukuran, salah warna, beda produk).",
    "Produk yang diterima berbeda dengan deskripsi.",
    "Outer packaging damaged",
    "Produk tidak original",
    "Barang tidak berfungsi/tidak bisa dipakai",
]

ALASAN_ENCODING = {k: i for i, k in enumerate(KATEGORI_ALASAN)}

ALASAN_GRUP = {
    "Pembeli tidak menerima pesanan."                                                                      : 0,
    "Barang cacat produksi (luntur/lecet/patah/dsb.)"                                                      : 1,
    "Cairan/isinya tumpah"                                                                                 : 1,
    "Produk pecah/hancur"                                                                                  : 1,
    "Ingin kembalikan barang sesuai kondisi awal"                                                          : 2,
    "Produk tidak lengkap"                                                                                 : 3,
    "Pembeli menerima produk yang salah (contoh: salah ukuran, salah warna, beda produk)."                 : 4,
    "Produk yang diterima berbeda dengan deskripsi."                                                       : 5,
    "Outer packaging damaged"                                                                              : 1,
    "Produk tidak original"                                                                                : 5,
    "Barang tidak berfungsi/tidak bisa dipakai"                                                            : 6,
}


# ============================================================
# KEYWORD SESUAI DATA ASLI
# ============================================================

RUSAK_KW    = ['rusak', 'cacat', 'luntur', 'lecet', 'patah', 'pecah', 'hancur', 'tumpah']
FUNGSI_KW   = ['tidak berfungsi', 'tidak bisa dipakai', 'tidak bisa digunakan', 'tidak nyala', 'mati']
UKURAN_KW   = ['salah ukuran', 'salah warna', 'beda produk', 'salah produk', 'ukuran salah']
SESUAI_KW   = ['berbeda dengan deskripsi', 'tidak sesuai deskripsi', 'tidak sesuai']
SAMPAI_KW   = ['tidak menerima pesanan', 'tidak sampai', 'tidak diterima', 'tidak terima']
KEMASAN_KW  = ['outer packaging', 'kemasan rusak', 'packaging damaged']
ORIGINAL_KW = ['tidak original', 'palsu', 'bukan original']
VISUAL_KW   = ['foto', 'video', 'gambar', 'bukti', 'dokumentasi', 'screenshot']


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    try:
        with open('return_classifier_model.pkl', 'rb') as f:
            pkg = pickle.load(f)
        return pkg, None
    except FileNotFoundError:
        return None, "File `return_classifier_model.pkl` tidak ditemukan."
    except Exception as e:
        return None, str(e)


# ============================================================
# HELPER
# ============================================================

def preprocess_text(text):
    if pd.isna(text) or str(text).strip() == '':
        return ''
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    STOPWORDS = {'yang','dan','di','ke','dari','ini','itu','dengan','untuk','tidak',
                 'pada','ada','sudah','saya','kami','kita','mereka','akan','bisa',
                 'juga','atau','tapi','karena','jika','kalau','adalah','nya','lebih',
                 'sangat','sesuai','kondisi','awal','barang','produk','pembeli'}
    words = [w for w in text.split() if w not in STOPWORDS and len(w) > 1]
    return ' '.join(words)

def flag_kw(teks, keywords):
    teks = str(teks).lower()
    return int(any(k in teks for k in keywords))

def layer1_check(alasan, catatan, tanggal_pesanan, tanggal_pengajuan, max_days=7):
    reasons = []
    catatan_lower = str(catatan).lower()
    alasan_lower  = str(alasan).lower()

    # A. Selisih hari
    try:
        selisih = (pd.to_datetime(tanggal_pengajuan) - pd.to_datetime(tanggal_pesanan)).days
        if selisih > max_days:
            reasons.append(f"⏰ Waktu pengajuan terlambat ({selisih} hari, maks {max_days} hari)")
    except:
        pass

    # B. Bukti visual di catatan
    for kw in VISUAL_KW:
        if kw in catatan_lower:
            reasons.append(f"📸 Catatan memerlukan verifikasi visual (kata: '{kw}')")
            break

    # C. Mismatch alasan vs catatan
    a_rusak  = flag_kw(alasan_lower, RUSAK_KW + FUNGSI_KW)
    a_ukuran = flag_kw(alasan_lower, UKURAN_KW)
    c_rusak  = flag_kw(catatan_lower, RUSAK_KW + FUNGSI_KW)
    c_ukuran = flag_kw(catatan_lower, UKURAN_KW)
    if (a_rusak and c_ukuran) or (a_ukuran and c_rusak):
        reasons.append("⚠️ Alasan dan catatan tidak konsisten")

    return len(reasons) > 0, reasons


def build_features(alasan, catatan, tanggal_pesanan, tanggal_pengajuan, model_pkg):
    """Bangun vektor fitur dari 4 input."""

    def safe_float(v):
        """Paksa nilai ke float64 — cegah ValueError scipy.sparse dtype str."""
        try:
            f = float(v)
            return f if np.isfinite(f) else 0.0
        except (TypeError, ValueError):
            return 0.0

    # Numerik dari tanggal
    try:
        tgl1    = pd.to_datetime(tanggal_pesanan)
        tgl2    = pd.to_datetime(tanggal_pengajuan)
        selisih = int((tgl2 - tgl1).days)
        dow     = int(tgl2.dayofweek)
        dom     = int(tgl2.day)
    except Exception:
        selisih = -1; dow = -1; dom = -1

    alasan_enc  = safe_float(model_pkg.get('alasan_encoding', ALASAN_ENCODING).get(alasan, -1))
    alasan_grup = safe_float(model_pkg.get('alasan_grup', ALASAN_GRUP).get(alasan, -1))
    is_menyesal = 1.0 if alasan == 'Ingin kembalikan barang sesuai kondisi awal' else 0.0

    alasan_c  = preprocess_text(alasan)
    catatan_c = preprocess_text(catatan)
    teks_gab  = alasan_c + ' ' + catatan_c

    num_dict = {
        'selisih_hari'        : safe_float(selisih),
        'hari_dalam_minggu'   : safe_float(dow),
        'hari_dalam_bulan'    : safe_float(dom),
        'lewat_1_hari'        : 1.0 if selisih > 1 else 0.0,
        'lewat_3_hari'        : 1.0 if selisih > 3 else 0.0,
        'lewat_7_hari'        : 1.0 if selisih > 7 else 0.0,
        'alasan_enc'          : alasan_enc,
        'alasan_grup'         : alasan_grup,
        'is_menyesal'         : is_menyesal,
        'panjang_catatan'     : float(len(catatan_c)),
        'panjang_total'       : float(len(alasan_c) + len(catatan_c)),
        'ada_rusak'           : float(flag_kw(catatan, RUSAK_KW)),
        'ada_fungsi'          : float(flag_kw(catatan, FUNGSI_KW)),
        'ada_ukuran'          : float(flag_kw(catatan, UKURAN_KW)),
        'ada_sesuai'          : float(flag_kw(catatan, SESUAI_KW)),
        'ada_sampai'          : float(flag_kw(catatan, SAMPAI_KW)),
        'ada_kemasan'         : float(flag_kw(catatan, KEMASAN_KW)),
        'ada_original'        : float(flag_kw(catatan, ORIGINAL_KW)),
        'ada_visual'          : float(flag_kw(catatan, VISUAL_KW)),
        'mismatch'            : 1.0 if (
            (flag_kw(alasan, RUSAK_KW) and flag_kw(catatan, UKURAN_KW)) or
            (flag_kw(alasan, UKURAN_KW) and flag_kw(catatan, RUSAK_KW))
        ) else 0.0,
    }

    numerical_cols = model_pkg['numerical_cols']

    # Eksplisit dtype=np.float64 — ini yang fix ValueError scipy.sparse
    num_vec = np.array(
        [[safe_float(num_dict.get(c, 0.0)) for c in numerical_cols]],
        dtype=np.float64
    )
    X_text  = model_pkg['tfidf'].transform([teks_gab])
    return hstack([csr_matrix(num_vec, dtype=np.float64), X_text])


def predict(alasan, catatan, tanggal_pesanan, tanggal_pengajuan, model_pkg):
    max_days = model_pkg.get('thresholds', {}).get('max_days', 7)

    # Layer 1
    perlu_manual, reasons = layer1_check(alasan, catatan, tanggal_pesanan, tanggal_pengajuan, max_days)
    if perlu_manual:
        return {'layer':1, 'label':'MANUAL', 'hasil':'PERLU CEK MANUAL',
                'confidence':None, 'proba_valid':None, 'proba_invalid':None, 'reasons':reasons}

    # Layer 2
    X     = build_features(alasan, catatan, tanggal_pesanan, tanggal_pengajuan, model_pkg)
    rf    = model_pkg['rf_model']
    proba = rf.predict_proba(X)[0]
    pred  = rf.predict(X)[0]
    lmap  = model_pkg.get('label_encoder', {0:'VALID', 1:'TIDAK VALID'})
    hasil = lmap.get(int(pred), '?')

    return {
        'layer'         : 2,
        'label'         : hasil,
        'hasil'         : hasil,
        'confidence'    : proba[int(pred)],
        'proba_valid'   : proba[0],
        'proba_invalid' : proba[1],
        'reasons'       : []
    }


# ============================================================
# SHOW RESULT
# ============================================================

def show_result(result):
    if result['label'] == 'VALID':
        st.markdown(f"""
        <div class="result-valid">
            <h3 style="margin:0">✅ VALID — Return Disetujui</h3>
            <p style="margin:6px 0 0">Pengajuan memenuhi kriteria pengembalian.</p>
            <p style="margin:4px 0 0"><strong>Confidence: {result['confidence']:.1%}</strong>
            &nbsp;|&nbsp; Layer 2 (AI)</p>
        </div>""", unsafe_allow_html=True)

    elif result['label'] == 'TIDAK VALID':
        st.markdown(f"""
        <div class="result-invalid">
            <h3 style="margin:0">❌ TIDAK VALID — Return Ditolak</h3>
            <p style="margin:6px 0 0">Pengajuan tidak memenuhi kriteria pengembalian.</p>
            <p style="margin:4px 0 0"><strong>Confidence: {result['confidence']:.1%}</strong>
            &nbsp;|&nbsp; Layer 2 (AI)</p>
        </div>""", unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="result-manual">
            <h3 style="margin:0">⚠️ PERLU CEK MANUAL</h3>
            <p style="margin:6px 0 0">Perlu diverifikasi oleh tim — Layer 1 (Rule-Based)</p>
        </div>""", unsafe_allow_html=True)
        for r in result['reasons']:
            st.write(f"• {r}")

    if result['layer'] == 2:
        st.markdown("<br>", unsafe_allow_html=True)
        p_v = result['proba_valid']
        p_i = result['proba_invalid']
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""<div style="text-align:center;padding:12px;background:#d4edda;border-radius:8px;">
                <div style="font-size:1.4rem;font-weight:700;color:#155724;">{p_v:.1%}</div>
                <div style="color:#155724;font-size:0.85rem;">Probabilitas VALID</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""<div style="text-align:center;padding:12px;background:#f8d7da;border-radius:8px;">
                <div style="font-size:1.4rem;font-weight:700;color:#721c24;">{p_i:.1%}</div>
                <div style="color:#721c24;font-size:0.85rem;">Probabilitas TIDAK VALID</div>
            </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.progress(float(p_v))
        st.caption(f"◄ VALID {p_v:.1%} ── TIDAK VALID {p_i:.1%} ►")


# ============================================================
# MAIN
# ============================================================

def main():
    st.markdown("""
    <div class="main-header">
        <h1>Sistem Rekomendasi Return Barang</h1>
        <p>Sistem Klasifikasi Pengembalian Barang — 2-Layer Hybrid</p>
    </div>""", unsafe_allow_html=True)

    model_pkg, error = load_model()
    if error:
        st.error(f"❌ {error}")
        st.info("Pastikan file `return_classifier_model.pkl` ada di folder yang sama.")
        st.stop()

    # Gunakan kategori dari model (jika tersimpan) atau default
    dropdown_options = model_pkg.get('kategori_alasan', KATEGORI_ALASAN)
    cv = model_pkg.get('cv_score', 0)
    st.caption(f"Model: Random Forest  |  CV F1-Score: {cv:.3f}  |  Kategori: {len(dropdown_options)}")

    tab1, tab2 = st.tabs(["📝 Input Manual", "📤 Upload Batch"])

    # ── TAB 1: INPUT MANUAL ───────────────────────────────────
    with tab1:
        st.subheader("Form Pengajuan Return")

        alasan = st.selectbox(
            "1️⃣  Alasan Pengembalian",
            options=dropdown_options,
            help="Pilih alasan sesuai kategori resmi"
        )

        catatan = st.text_area(
            "2️⃣  Catatan Pengembalian",
            placeholder="Contoh: Barang tiba dalam kondisi penyok, sudah dicoba tapi tidak bisa menyala.",
            height=110,
        )

        col1, col2 = st.columns(2)
        with col1:
            tanggal_pesanan = st.date_input(
                "3️⃣  Tanggal Pesanan",
                value=date.today() - timedelta(days=4)
            )
        with col2:
            tanggal_pengajuan = st.date_input(
                "4️⃣  Tanggal Pengajuan Return",
                value=date.today()
            )

        # Info selisih hari
        selisih = (tanggal_pengajuan - tanggal_pesanan).days
        max_days = model_pkg.get('thresholds', {}).get('max_days', 7)
        if selisih < 0:
            st.error("⚠️ Tanggal pengajuan tidak boleh sebelum tanggal pesanan!")
        else:
            color = "#d4edda" if selisih <= max_days else "#fff3cd"
            icon  = "✅" if selisih <= max_days else "⚠️"
            st.markdown(
                f'<div style="background:{color};border-radius:6px;padding:8px 14px;font-size:0.9rem;">'
                f'{icon} Selisih: <strong>{selisih} hari</strong> (batas: {max_days} hari)</div>',
                unsafe_allow_html=True
            )

        if st.button("🔍 Klasifikasi Sekarang"):
            if not catatan.strip():
                st.warning("Harap isi catatan pengembalian.")
            elif selisih < 0:
                st.error("Periksa kembali tanggal.")
            else:
                with st.spinner("Memproses..."):
                    result = predict(alasan, catatan,
                                     str(tanggal_pesanan), str(tanggal_pengajuan),
                                     model_pkg)
                st.markdown("---")
                st.subheader("Hasil Klasifikasi")
                show_result(result)

                with st.expander("📋 Detail Input"):
                    st.write(f"**Alasan**      : {alasan}")
                    st.write(f"**Catatan**     : {catatan}")
                    st.write(f"**Tgl Pesanan** : {tanggal_pesanan}")
                    st.write(f"**Tgl Pengajuan**: {tanggal_pengajuan}")
                    st.write(f"**Selisih Hari** : {selisih}")

    # ── TAB 2: UPLOAD BATCH ───────────────────────────────────
    with tab2:
        st.subheader("Upload File Batch")
        st.markdown("""
        **Kolom wajib** (nama harus sama persis):

        | Kolom | Keterangan |
        |---|---|
        | `Alasan Pengembalian` | Harus sesuai 11 kategori resmi |
        | `Catatan Pengembalian Barang` | Teks bebas |
        | `Tanggal Pesanan` | Format: YYYY-MM-DD |
        | `Waktu Pengembalian` | Format: YYYY-MM-DD |
        """)

        with st.expander("📋 Lihat 11 Kategori Alasan yang Valid"):
            for i, k in enumerate(dropdown_options, 1):
                st.write(f"{i:2d}. {k}")

        uploaded = st.file_uploader("Upload Excel (.xlsx) atau CSV (.csv)", type=['xlsx','csv'])

        if uploaded:
            try:
                df_up = pd.read_csv(uploaded) if uploaded.name.endswith('.csv') \
                        else pd.read_excel(uploaded)

                st.success(f"✅ {len(df_up)} baris dimuat")

                REQUIRED = ['Alasan Pengembalian', 'Catatan Pengembalian Barang',
                            'Tanggal Pesanan', 'Waktu Pengembalian']
                missing = [c for c in REQUIRED if c not in df_up.columns]
                if missing:
                    st.error(f"❌ Kolom tidak ditemukan: {missing}")
                    st.stop()

                # Cek apakah ada alasan yang tidak dikenal
                alasan_unknown = df_up[
                    ~df_up['Alasan Pengembalian'].isin(dropdown_options)
                ]['Alasan Pengembalian'].unique()
                if len(alasan_unknown) > 0:
                    st.warning(f"⚠️ {len(alasan_unknown)} nilai alasan tidak dikenal (akan di-encode -1):")
                    for a in alasan_unknown:
                        st.write(f"  • `{a}`")

                with st.expander("Preview (5 baris)"):
                    st.dataframe(df_up.head(), use_container_width=True)

                if st.button("🚀 Proses Semua Data"):
                    results = []
                    bar  = st.progress(0)
                    info = st.empty()
                    total = len(df_up)

                    for i, row in df_up.iterrows():
                        info.text(f"Memproses {i+1}/{total}...")
                        bar.progress((i+1)/total)
                        r = predict(
                            row.get('Alasan Pengembalian', ''),
                            row.get('Catatan Pengembalian Barang', ''),
                            row.get('Tanggal Pesanan', None),
                            row.get('Waktu Pengembalian', None),
                            model_pkg
                        )
                        results.append({
                            'Prediksi'  : r['label'],
                            'Layer'     : f"Layer {r['layer']}",
                            'Confidence': f"{r['confidence']:.1%}" if r['confidence'] else 'N/A',
                            'Keterangan': '; '.join(r['reasons']) if r['reasons'] else 'Otomatis (AI)'
                        })

                    bar.empty(); info.empty()
                    df_result = pd.concat([df_up.reset_index(drop=True), pd.DataFrame(results)], axis=1)

                    st.markdown("---")
                    st.subheader("📊 Ringkasan Hasil")
                    counts = df_result['Prediksi'].value_counts()
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        n = counts.get('VALID', 0)
                        st.metric("✅ Valid", n, f"{n/total:.0%}")
                    with c2:
                        n = counts.get('TIDAK VALID', 0)
                        st.metric("❌ Tidak Valid", n, f"{n/total:.0%}")
                    with c3:
                        n = counts.get('MANUAL', 0)
                        st.metric("⚠️ Cek Manual", n, f"{n/total:.0%}")
                    with c4:
                        auto = total - counts.get('MANUAL', 0)
                        st.metric("🤖 Otomatis", auto, f"{auto/total:.0%}")

                    st.dataframe(df_result, use_container_width=True)

                    buf = io.BytesIO()
                    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
                        df_result.to_excel(writer, index=False, sheet_name='Hasil')
                    buf.seek(0)
                    st.download_button(
                        "📥 Download Hasil (.xlsx)", data=buf,
                        file_name=f"hasil_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
            except Exception as e:
                st.error(f"Error: {e}")
                st.exception(e)

    st.markdown("---")
    st.caption("Sistem Rekomendasi Return Barang· 2-Layer Hybrid · Random Forest · 11 Kategori Resmi")


if __name__ == "__main__":
    main()