"""
app.py — Mantar Sınıflandırıcı (Türkçe · Full-Width Yeniden Tasarım)
─────────────────────────────────────────────────────────────────────
Çalıştırmak için:
    streamlit run app.py
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# ─────────────────────────────────────────────────────────────────────
# SAYFA YAPILANDIRMASI  (sidebar gizle, tam genişlik)
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "🍄 Mantar Sınıflandırıcı",
    page_icon  = "🍄",
    layout     = "wide",
    initial_sidebar_state = "collapsed",
)

# ─────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Sidebar tamamen gizle ────────────────────────────────────────── */
[data-testid="stSidebar"]        { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }

/* ── Sayfa arka planı ─────────────────────────────────────────────── */
.stApp { background: #0d1117; color: #e2e8f0; }
.block-container { padding: 0 3rem 2rem 3rem !important; max-width: 1280px; }

/* ── Tipografi ────────────────────────────────────────────────────── */
html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }

/* ── Selectbox ────────────────────────────────────────────────────── */
.stSelectbox label {
    color: #64748b !important;
    font-size: 11.5px !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: .5px;
}
.stSelectbox > div > div {
    background: #161b2e !important;
    border: 1px solid #1e2d4a !important;
    border-radius: 10px !important;
    color: #cbd5e1 !important;
}
.stSelectbox > div > div:focus-within {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,.2) !important;
}

/* ── Ana tahmin butonu ───────────────────────────────────────────── */
div[data-testid="stVerticalBlock"] div.predict-btn-wrap .stButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 18px 0 !important;
    font-size: 18px !important;
    font-weight: 800 !important;
    width: 100% !important;
    letter-spacing: .3px;
    box-shadow: 0 6px 28px rgba(99,102,241,.45) !important;
    transition: all .25s ease !important;
}
div[data-testid="stVerticalBlock"] div.predict-btn-wrap .stButton > button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 12px 36px rgba(99,102,241,.6) !important;
}

/* ── Grup kartı ──────────────────────────────────────────────────── */
.feat-group-card {
    background: #111827;
    border: 1px solid #1e2d4a;
    border-radius: 16px;
    padding: 20px 24px 16px;
    margin-bottom: 20px;
}
.feat-group-title {
    font-size: 13px;
    font-weight: 700;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 14px;
    padding-bottom: 8px;
    border-bottom: 1px solid #1e2d4a;
}

/* ── Sonuç kartları ──────────────────────────────────────────────── */
.result-card-edible {
    background: linear-gradient(135deg, #052e16, #064e3b);
    border: 2px solid #34d399;
    border-radius: 20px;
    padding: 36px 32px;
    text-align: center;
    box-shadow: 0 8px 40px rgba(52,211,153,.2);
    animation: fadeUp .4s ease;
}
.result-card-poison {
    background: linear-gradient(135deg, #3b0000, #7f1d1d);
    border: 2px solid #f87171;
    border-radius: 20px;
    padding: 36px 32px;
    text-align: center;
    box-shadow: 0 8px 40px rgba(248,113,113,.2);
    animation: fadeUp .4s ease;
}
@keyframes fadeUp {
    from { opacity:0; transform:translateY(16px); }
    to   { opacity:1; transform:translateY(0); }
}

/* ── Olasılık çubuk etiketi ──────────────────────────────────────── */
.prob-label { font-size: 13px; font-weight: 700; }
.prob-pct   { font-size: 17px; font-weight: 800; }

/* ── Alt bilgi paneli ────────────────────────────────────────────── */
.info-card {
    background: #0f1623;
    border: 1px solid #1a2540;
    border-radius: 14px;
    padding: 18px 22px;
}
.info-card-title {
    font-size: 11px;
    font-weight: 700;
    color: #334155;
    text-transform: uppercase;
    letter-spacing: .8px;
    margin-bottom: 12px;
}

/* ── Feature-imp satır ───────────────────────────────────────────── */
.fi-row { display:flex; align-items:center; gap:8px; margin-bottom:7px; }
.fi-name { font-size:11.5px; color:#475569; width:190px; flex-shrink:0; }
.fi-bar-wrap { flex:1; background:#161b2e; border-radius:4px; height:6px; }
.fi-bar { height:6px; border-radius:4px;
          background: linear-gradient(90deg,#6366f1,#8b5cf6); }
.fi-pct { font-size:11px; color:#4f46e5; width:38px; text-align:right; }

/* ── Divider ─────────────────────────────────────────────────────── */
hr { border-color: #1e2d4a !important; margin: 32px 0 !important; }

/* ── Metric widget ───────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: #0f1623;
    border: 1px solid #1a2540;
    border-radius: 10px;
    padding: 12px 16px;
}
[data-testid="stMetricLabel"] { color: #334155 !important; font-size:11px !important; }
[data-testid="stMetricValue"] { color: #6366f1 !important; font-size:20px !important; font-weight:800 !important; }

/* ── Progress bar ────────────────────────────────────────────────── */
.stProgress > div > div { border-radius: 6px !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# FEATURE KATALOĞU
# ─────────────────────────────────────────────────────────────────────
FEATURE_OPTIONS: dict[str, dict[str, str]] = {
    "cap-shape": {
        "b": "Bell — Çan",      "c": "Conical — Konik",
        "f": "Flat — Düz",      "k": "Knobbed — Düğmeli",
        "s": "Sunken — Çukur",  "x": "Convex — Konveks",
    },
    "cap-surface": {
        "f": "Fibrous — Lifli", "g": "Grooves — Oluklu",
        "s": "Smooth — Pürüzsüz", "y": "Scaly — Pullu",
    },
    "cap-color": {
        "b": "Buff — Krem",       "c": "Cinnamon — Tarçın",
        "e": "Red — Kırmızı",    "g": "Gray — Gri",
        "n": "Brown — Kahve",    "p": "Pink — Pembe",
        "r": "Green — Yeşil",    "u": "Purple — Mor",
        "w": "White — Beyaz",    "y": "Yellow — Sarı",
    },
    "bruises": {
        "f": "Hayır — Leke Yok",
        "t": "Evet — Leke Var",
    },
    "odor": {
        "a": "Badem",     "c": "Kreozot",   "f": "Kötü Koku",
        "l": "Anason",    "m": "Küflü",     "n": "Kokusuz",
        "p": "Keskin",    "s": "Baharatlı", "y": "Balık Kokusu",
    },
    "gill-attachment": {
        "a": "Attached — Tutunmuş",
        "f": "Free — Serbest",
    },
    "gill-spacing": {
        "c": "Close — Sık",
        "w": "Crowded — Çok Sık",
    },
    "gill-size": {
        "b": "Broad — Geniş",
        "n": "Narrow — Dar",
    },
    "gill-color": {
        "b": "Buff — Krem",     "e": "Red — Kırmızı",
        "g": "Gray — Gri",      "h": "Chocolate — Çikolata",
        "k": "Black — Siyah",   "n": "Brown — Kahve",
        "o": "Orange — Turuncu","p": "Pink — Pembe",
        "r": "Green — Yeşil",   "u": "Purple — Mor",
        "w": "White — Beyaz",   "y": "Yellow — Sarı",
    },
    "stalk-shape": {
        "e": "Enlarging — Genişleyen",
        "t": "Tapering — Daralan",
    },
    "stalk-root": {
        "?": "Bilinmiyor",        "b": "Bulbous — Yumrulu",
        "c": "Club — Topuz",      "e": "Equal — Düz",
        "r": "Rooted — Köklü",
    },
    "stalk-surface-above-ring": {
        "f": "Fibrous — Lifli",  "k": "Silky — İpeksi",
        "s": "Smooth — Pürüzsüz","y": "Scaly — Pullu",
    },
    "stalk-surface-below-ring": {
        "f": "Fibrous — Lifli",  "k": "Silky — İpeksi",
        "s": "Smooth — Pürüzsüz","y": "Scaly — Pullu",
    },
    "stalk-color-above-ring": {
        "b": "Buff — Krem",    "c": "Cinnamon — Tarçın",
        "e": "Red — Kırmızı", "g": "Gray — Gri",
        "n": "Brown — Kahve", "o": "Orange — Turuncu",
        "p": "Pink — Pembe",  "w": "White — Beyaz",
        "y": "Yellow — Sarı",
    },
    "stalk-color-below-ring": {
        "b": "Buff — Krem",    "c": "Cinnamon — Tarçın",
        "e": "Red — Kırmızı", "g": "Gray — Gri",
        "n": "Brown — Kahve", "o": "Orange — Turuncu",
        "p": "Pink — Pembe",  "w": "White — Beyaz",
        "y": "Yellow — Sarı",
    },
    "veil-type": {
        "p": "Partial — Kısmi",
    },
    "veil-color": {
        "n": "Brown — Kahve", "o": "Orange — Turuncu",
        "w": "White — Beyaz", "y": "Yellow — Sarı",
    },
    "ring-number": {
        "n": "None — Yok", "o": "One — Bir", "t": "Two — İki",
    },
    "ring-type": {
        "e": "Evanescent — Geçici", "f": "Flaring — Açık",
        "l": "Large — Büyük",       "n": "None — Yok",
        "p": "Pendant — Sarkan",
    },
    "spore-print-color": {
        "b": "Buff — Krem",     "h": "Chocolate — Çikolata",
        "k": "Black — Siyah",  "n": "Brown — Kahve",
        "o": "Orange — Turuncu","r": "Green — Yeşil",
        "u": "Purple — Mor",   "w": "White — Beyaz",
        "y": "Yellow — Sarı",
    },
    "population": {
        "a": "Abundant — Bol",    "c": "Clustered — Küme",
        "n": "Numerous — Çok",    "s": "Scattered — Dağınık",
        "v": "Several — Birkaç",  "y": "Solitary — Tek",
    },
    "habitat": {
        "d": "Woods — Orman",  "g": "Grasses — Çayır",
        "l": "Leaves — Yaprak","m": "Meadows — Mera",
        "p": "Paths — Yol",    "u": "Urban — Kentsel",
        "w": "Waste — Çöplük",
    },
}

# Gruplar ve görünüm sırası (her grup 3 sütunlu ızgara)
FEATURE_GROUPS = [
    ("🍄 Şapka Özellikleri",       ["cap-shape", "cap-surface", "cap-color"]),
    ("🔬 Temel Özellikler",        ["bruises", "odor"]),
    ("🌿 Solungaç Özellikleri",    ["gill-attachment", "gill-spacing", "gill-size", "gill-color"]),
    ("🌱 Sap Özellikleri",         [
        "stalk-shape", "stalk-root",
        "stalk-surface-above-ring", "stalk-surface-below-ring",
        "stalk-color-above-ring",   "stalk-color-below-ring",
    ]),
    ("💍 Peçe & Halka",            ["veil-type", "veil-color", "ring-number", "ring-type"]),
    ("🌍 Çevre & Popülasyon",      ["spore-print-color", "population", "habitat"]),
]


# ─────────────────────────────────────────────────────────────────────
# MODEL YÜKLEME
# ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(path: str = "mushroom_model.pkl"):
    if not os.path.exists(path):
        return None
    return joblib.load(path)


bundle = load_model()


# ─────────────────────────────────────────────────────────────────────
# YARDIMCI: girdileri encode et
# ─────────────────────────────────────────────────────────────────────
def encode_inputs(inputs: dict, feat_names: list, encoders: dict) -> np.ndarray:
    row = [encoders[f].transform([inputs[f]])[0] for f in feat_names]
    return np.array(row).reshape(1, -1)


# ─────────────────────────────────────────────────────────────────────
# SESSION STATE başlangıcı
# ─────────────────────────────────────────────────────────────────────
if "prediction" not in st.session_state:
    st.session_state.prediction = None


# ═════════════════════════════════════════════════════════════════════
# HEADER
# ═════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="padding: 48px 0 28px; text-align: center;">
    <div style="font-size: 72px; line-height:1; margin-bottom: 16px;">🍄</div>
    <h1 style="
        font-size: 42px; font-weight: 900; margin: 0;
        background: linear-gradient(135deg, #34d399 20%, #818cf8 80%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        letter-spacing: -1.5px;
    ">Mantar Sınıflandırıcı</h1>
    <p style="color: #334155; font-size: 15px; margin-top: 10px; letter-spacing: .2px;">
        22 özellik seç &nbsp;→&nbsp; Tahmin Et &nbsp;→&nbsp; Yenilebilir mi Zehirli mi?
    </p>
</div>
""", unsafe_allow_html=True)

# Model yoksa dur
if bundle is None:
    st.error("⚠️ **mushroom_model.pkl bulunamadı!** Önce `python train_model.py` komutunu çalıştırın.")
    st.stop()

model      = bundle["model"]
encoders   = bundle["encoders"]
feat_names = bundle["feature_names"]
feat_imp   = bundle["feature_importances"]
test_acc   = bundle["test_accuracy"]


# ═════════════════════════════════════════════════════════════════════
# ÖZELLİK SEÇİM IZGARASI
# ═════════════════════════════════════════════════════════════════════
user_inputs: dict[str, str] = {}

# İçerik genişliği için tek orta sütun
_, center, _ = st.columns([1, 14, 1])

with center:

    # Her özellik grubu ayrı bir kart içinde
    for group_name, features in FEATURE_GROUPS:
        st.markdown(
            f'<div class="feat-group-card">'
            f'<div class="feat-group-title">{group_name}</div>',
            unsafe_allow_html=True,
        )

        # 3'lü ızgara
        cols = st.columns(3)
        for idx, feat in enumerate(features):
            opts   = FEATURE_OPTIONS[feat]
            codes  = list(opts.keys())
            labels = list(opts.values())

            with cols[idx % 3]:
                chosen_label = st.selectbox(
                    label   = feat,
                    options = labels,
                    index   = 0,
                    key     = f"inp_{feat}",
                )
                user_inputs[feat] = codes[labels.index(chosen_label)]

        st.markdown("</div>", unsafe_allow_html=True)

    # ── TAHMİN BUTONU ────────────────────────────────────────────────
    st.markdown("<div class='predict-btn-wrap'>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    predict_clicked = st.button("🔍  Tahmin Et", use_container_width=True, key="predict_btn")
    st.markdown("</div>", unsafe_allow_html=True)

    if predict_clicked:
        X_in = encode_inputs(user_inputs, feat_names, encoders)
        pred = model.predict(X_in)[0]
        prob = model.predict_proba(X_in)[0]
        st.session_state.prediction = {
            "pred": pred,
            "p_edible": float(prob[0]),
            "p_poison": float(prob[1]),
        }

    # ── SONUÇ ALANI ──────────────────────────────────────────────────
    if st.session_state.prediction:
        res       = st.session_state.prediction
        is_edible = res["pred"] == 0
        p_e       = res["p_edible"]
        p_p       = res["p_poison"]

        st.markdown("<br>", unsafe_allow_html=True)

        if is_edible:
            st.markdown(f"""
<div class="result-card-edible">
    <div style="font-size:64px; margin-bottom:10px;">✅</div>
    <h2 style="color:#34d399; font-size:36px; font-weight:900; margin:0 0 6px 0;
               letter-spacing:-1px;">YENİLEBİLİR</h2>
    <p style="color:#6ee7b7; font-size:14px; margin:0 0 24px 0;">
        EDIBLE — Bu mantar büyük olasılıkla güvenlidir.
    </p>
</div>
""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
<div class="result-card-poison">
    <div style="font-size:64px; margin-bottom:10px;">☠️</div>
    <h2 style="color:#f87171; font-size:36px; font-weight:900; margin:0 0 6px 0;
               letter-spacing:-1px;">ZEHİRLİ</h2>
    <p style="color:#fca5a5; font-size:14px; margin:0 0 24px 0;">
        POISONOUS — Bu mantar tehlikeli! Yemeyiniz.
    </p>
</div>
""", unsafe_allow_html=True)

        # ── Olasılık çubukları ────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)

        prob_l, prob_r = st.columns(2)

        with prob_l:
            st.markdown(
                f'<p class="prob-label" style="color:#34d399;">✅ Yenilebilir</p>'
                f'<p class="prob-pct" style="color:#34d399;">%{p_e*100:.1f}</p>',
                unsafe_allow_html=True,
            )
            st.progress(p_e)

        with prob_r:
            st.markdown(
                f'<p class="prob-label" style="color:#f87171;">☠️ Zehirli</p>'
                f'<p class="prob-pct" style="color:#f87171;">%{p_p*100:.1f}</p>',
                unsafe_allow_html=True,
            )
            st.progress(p_p)


# ═════════════════════════════════════════════════════════════════════
# ALT BİLGİ PANELI  (minimalist, arka planda)
# ═════════════════════════════════════════════════════════════════════
    st.markdown("<br><hr>", unsafe_allow_html=True)

    info_l, info_m, info_r = st.columns([1.2, 1.8, 1])

    # ── Sol: Model metrikleri ─────────────────────────────────────────
    with info_l:
        st.markdown('<div class="info-card"><div class="info-card-title">Model Bilgisi</div>', unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        m1.metric("Test Accuracy", f"%{test_acc*100:.1f}")
        m2.metric("Ağaç Sayısı", "200")
        _, m3, _ = st.columns([0.5, 2, 0.5])
        m3.metric("Feature Sayısı", "22")
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Orta: Feature importance ──────────────────────────────────────
    with info_m:
        st.markdown('<div class="info-card"><div class="info-card-title">En Önemli Özellikler</div>', unsafe_allow_html=True)

        sorted_fi = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:10]
        max_fi    = sorted_fi[0][1]

        rows_html = ""
        for feat, score in sorted_fi:
            bar_w = int(score / max_fi * 100)
            rows_html += (
                f'<div class="fi-row">'
                f'  <span class="fi-name">{feat}</span>'
                f'  <div class="fi-bar-wrap"><div class="fi-bar" style="width:{bar_w}%"></div></div>'
                f'  <span class="fi-pct">{score*100:.1f}%</span>'
                f'</div>'
            )
        st.markdown(rows_html, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Sağ: Hızlı referans ──────────────────────────────────────────
    with info_r:
        st.markdown('<div class="info-card"><div class="info-card-title">Hızlı Rehber</div>', unsafe_allow_html=True)
        st.markdown("""
<div style="font-size:12px; color:#334155; line-height:1.9;">
    <span style="color:#f87171;">☠</span> odor=Kötü Koku → %100 zehirli<br>
    <span style="color:#34d399;">✓</span> odor=Badem → %100 yenilebilir<br>
    <span style="color:#f87171;">☠</span> gill-size=Dar → %88 zehirli<br>
    <span style="color:#34d399;">✓</span> habitat=Yaprak → %100 yenilebilir<br>
    <span style="color:#f87171;">☠</span> spore=Yeşil → %100 zehirli<br>
    <span style="color:#34d399;">✓</span> Leke Var → %81 yenilebilir
</div>
""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════
# FOOTER
# ═════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="text-align:center; color:#1e2d4a; font-size:12px; padding: 32px 0 16px 0;">
    🍄 Mantar Sınıflandırıcı &nbsp;·&nbsp; UCI Mushroom Dataset
    &nbsp;·&nbsp; Random Forest &nbsp;·&nbsp; Streamlit
    <br><br>
    <span style="color:#161b2e; font-size:11px;">
        ⚠️ Bu uygulama yalnızca eğitim amaçlıdır. Gerçek mantarlar için uzman görüşü alın.
    </span>
</div>
""", unsafe_allow_html=True)
