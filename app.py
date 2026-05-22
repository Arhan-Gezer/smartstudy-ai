import io
import json
import os
import re
from collections import Counter
from datetime import date, timedelta

import google.generativeai as genai
import matplotlib.pyplot as plt
import nltk
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from wordcloud import WordCloud

SENTIMENT_MODEL = "savasy/bert-base-turkish-sentiment-cased"
MAX_SENTENCES = 100

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-flash-latest"

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


def extract_pdf_text(uploaded_file) -> tuple[str, int]:
    """Return (full_text, page_count). Raises on unreadable PDF."""
    reader = PdfReader(uploaded_file)
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n\n".join(pages).strip(), len(reader.pages)


def summarize_with_gemini(text: str) -> str:
    prompt = (
        "Aşağıdaki ders notunu Türkçe olarak özetle.\n"
        "Kurallar:\n"
        "- Önce 2-3 cümlelik kısa bir genel özet ver.\n"
        "- Sonra '## Ana Kavramlar' başlığı altında ana kavramları "
        "madde madde (Markdown listesi) çıkar.\n"
        "- Sonra '## Önemli Noktalar' başlığı altında öğrencinin "
        "sınava çalışırken dikkat etmesi gereken 3-5 noktayı yaz.\n"
        "- Akademik ama anlaşılır bir dil kullan.\n\n"
        "--- DERS NOTU ---\n"
        f"{text}\n"
        "--- BİTTİ ---"
    )
    model = genai.GenerativeModel(GEMINI_MODEL)
    response = model.generate_content(prompt)
    return response.text


QUIZ_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "konu": {"type": "string"},
        "soru": {"type": "string"},
        "secenekler": {"type": "array", "items": {"type": "string"}},
        "dogru_cevap_index": {"type": "integer"},
        "aciklama": {"type": "string"},
    },
    "required": ["konu", "soru", "secenekler", "dogru_cevap_index", "aciklama"],
}

QUIZ_SCHEMA = {"type": "array", "items": QUIZ_ITEM_SCHEMA}


def generate_quiz(text: str, n_questions: int, difficulty: str) -> list[dict]:
    prompt = (
        f"Aşağıdaki ders metninden {n_questions} adet çoktan seçmeli Türkçe "
        f"quiz sorusu üret. Zorluk seviyesi: {difficulty}.\n\n"
        "Kurallar:\n"
        "- Her soru tam 4 şıklı olmalı (secenekler dizisi 4 string içermeli).\n"
        "- `dogru_cevap_index` 0-3 arasında bir tam sayı "
        "(doğru şıkkın sıfır-tabanlı indeksi).\n"
        "- `konu` alanı 2-3 kelimelik, dersin alt başlığını temsil etsin "
        "(örn: 'Süreç Durumları', 'Bellek Yönetimi', 'IPC', 'Zamanlayıcılar').\n"
        "- `aciklama` alanı doğru cevabın neden doğru olduğunu kısaca açıklasın.\n"
        f"- Zorluk '{difficulty}': "
        "Kolay = temel tanım/hatırlama, Orta = uygulama/karşılaştırma, "
        "Zor = analiz/sentez/edge-case.\n"
        "- Sorular birbirinden farklı konuları kapsasın (zayıf konu analizi için).\n\n"
        "--- METİN ---\n"
        f"{text}\n"
        "--- BİTTİ ---"
    )
    model = genai.GenerativeModel(
        GEMINI_MODEL,
        generation_config={
            "response_mime_type": "application/json",
            "response_schema": QUIZ_SCHEMA,
            "max_output_tokens": 8192,
        },
    )
    response = model.generate_content(prompt)
    return json.loads(response.text)


EXTRA_STOPWORDS = {
    # Türkçe dolgu
    "bir", "bu", "ile", "için", "olarak", "olan", "ki", "de", "da",
    "mi", "mı", "mu", "mü", "ya", "ve", "veya", "ama", "fakat", "ancak",
    "gibi", "kadar", "daha", "çok", "az", "her", "tüm", "şu", "o",
    "ben", "sen", "biz", "siz", "onlar", "şekil", "tablo", "sayfa",
    "bkz", "yani", "ise",
    # Yazar / kaynak gürültüsü
    "silberschatz", "galvin", "gagne", "byahmet", "koltuksuz",
    "march", "edition",
    # Akademik dolgu
    "concept", "concepts", "example", "examples", "chapter",
    "figure", "see", "section", "page", "ch", "fig",
    # Genel zayıf İngilizce
    "must", "may", "can", "will", "shall", "however", "thus", "also", "etc",
}


@st.cache_resource
def get_combined_stopwords() -> set[str]:
    try:
        from nltk.corpus import stopwords as _sw
        tr = set(_sw.words("turkish"))
        en = set(_sw.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords as _sw
        tr = set(_sw.words("turkish"))
        en = set(_sw.words("english"))
    return tr | en | EXTRA_STOPWORDS


def turkish_lower(text: str) -> str:
    """Turkish-aware lowercase: İ→i, I→ı before generic .lower()."""
    return text.replace("İ", "i").replace("I", "ı").lower()


def preprocess_for_wordcloud(
    text: str, min_len: int, stopwords: set[str]
) -> list[str]:
    text = turkish_lower(text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = text.replace("_", " ")
    return [
        tok for tok in text.split()
        if len(tok) >= min_len and tok not in stopwords
    ]


@st.cache_resource(show_spinner="Türkçe BERT modeli yükleniyor (ilk seferde ~30 sn)...")
def load_sentiment_pipeline():
    return pipeline("sentiment-analysis", model=SENTIMENT_MODEL)


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) >= 5]


def generate_study_plan(
    days: int, hours: int, weak_df: pd.DataFrame, strong_df: pd.DataFrame
) -> str:
    weak_lines = "\n".join(
        f"- {r['konu']}: %{r['accuracy']:.0f} doğruluk "
        f"({int(r['correct'])}/{int(r['total'])})"
        for _, r in weak_df.iterrows()
    ) or "- (yok)"
    strong_lines = "\n".join(
        f"- {r['konu']}: %{r['accuracy']:.0f} doğruluk"
        for _, r in strong_df.iterrows()
    ) or "- (yok)"
    prompt = (
        f"Öğrenci {days} gün sonra sınava girecek, günde {hours} saat "
        f"çalışabilir (toplam {days * hours} saat).\n\n"
        f"**Zayıf olduğu konular (öncelik):**\n{weak_lines}\n\n"
        f"**Güçlü olduğu konular (tekrar amaçlı):**\n{strong_lines}\n\n"
        "Türkçe, Markdown formatında günlük çalışma planı oluştur:\n"
        "- Her gün için '### Gün N' başlığı kullan.\n"
        "- O gün hangi konu(lar) çalışılacak, kaç saat ayrılacak, "
        "ne tür çalışma yapılacak (okuma / quiz çözme / özet çıkarma / "
        "kavram haritası / örnek problem).\n"
        "- Zayıf konulara orantılı olarak daha fazla zaman ayır.\n"
        "- Son 2 günde tekrar + deneme sınavı + zihinsel hazırlık olsun.\n"
        "- Akademik ama motive edici, kısa cümlelerle yaz.\n"
        "- En sonda '## Sınav Günü İpuçları' başlığı altında "
        "3-5 maddelik liste ekle."
    )
    model = genai.GenerativeModel(GEMINI_MODEL)
    return model.generate_content(prompt).text


def accuracy_color(acc: float) -> str:
    if acc < 60:
        return "#e74c3c"
    if acc < 80:
        return "#f39c12"
    return "#2ecc71"


st.set_page_config(
    page_title="SmartStudy AI",
    page_icon=":books:",
    layout="wide",
)

st.title("SmartStudy AI")
st.caption("ECS-VB-YZ Dönem Sonu Projesi — Öğrenci Çalışma Asistanı")

with st.sidebar:
    st.header("Durum")
    if GEMINI_API_KEY:
        st.success("Gemini API key yüklendi (.env)")
    else:
        st.error("GEMINI_API_KEY bulunamadı. .env dosyasını doldurun.")
    st.divider()
    st.markdown(
        "**Modüller**\n"
        "1. PDF Özet (Gemini)\n"
        "2. Quiz Üretici (Gemini)\n"
        "3. Kelime Bulutu\n"
        "4. Sentiment (Türkçe BERT)\n"
        "5. Zayıf Konu Tahmini"
    )

tab_summary, tab_quiz, tab_wordcloud, tab_sentiment, tab_weak = st.tabs(
    [
        "1. PDF Özet",
        "2. Quiz Üretici",
        "3. Kelime Bulutu",
        "4. Sentiment",
        "5. Zayıf Konu",
    ]
)


with tab_summary:
    st.subheader("PDF Yükle ve Özetle")
    st.caption(f"Model: `{GEMINI_MODEL}` · Türkçe ders notları için optimize edildi")

    uploaded_pdf = st.file_uploader(
        "PDF seçin (ders notu, slayt, makale)",
        type=["pdf"],
        key="summary_pdf",
    )

    pdf_text = ""
    page_count = 0
    if uploaded_pdf is not None:
        file_id = (uploaded_pdf.name, uploaded_pdf.size)
        if st.session_state.get("summary_file_id") != file_id:
            try:
                pdf_text, page_count = extract_pdf_text(uploaded_pdf)
                st.session_state["summary_file_id"] = file_id
                st.session_state["summary_pdf_text"] = pdf_text
                st.session_state["summary_pdf_pages"] = page_count
                if pdf_text:
                    st.session_state["last_pdf_text"] = pdf_text
            except PdfReadError as e:
                st.error(f"PDF okunamadı: {e}")
            except Exception as e:
                st.error(f"Beklenmeyen hata: {e}")
        pdf_text = st.session_state.get("summary_pdf_text", "")
        page_count = st.session_state.get("summary_pdf_pages", 0)

        if pdf_text:
            st.success(
                f"PDF okundu: **{page_count}** sayfa, "
                f"**{len(pdf_text):,}** karakter çıkarıldı."
            )
            with st.expander("Çıkarılan metni görüntüle (debug)"):
                st.text_area(
                    "Ham metin",
                    value=pdf_text,
                    height=300,
                    key="summary_raw_text",
                    label_visibility="collapsed",
                )
        elif page_count > 0:
            st.warning(
                "PDF okundu ama metin çıkarılamadı. "
                "Taranmış (görüntü tabanlı) bir PDF olabilir — OCR gerekir."
            )

    can_summarize = bool(pdf_text) and bool(GEMINI_API_KEY)
    if not GEMINI_API_KEY:
        st.warning("Özet üretmek için `.env` içine `GEMINI_API_KEY` ekleyin.")

    if st.button("Özet Çıkar", key="summary_btn", disabled=not can_summarize, type="primary"):
        with st.spinner("Gemini özet üretiyor..."):
            try:
                summary = summarize_with_gemini(pdf_text)
                st.session_state["last_summary"] = summary
            except Exception as e:
                st.error(f"Gemini API hatası: {e}")

    last_summary = st.session_state.get("last_summary")
    if last_summary:
        st.markdown("### Özet")
        st.markdown(last_summary)


with tab_quiz:
    st.subheader("Quiz Üretici")
    st.caption(f"Model: `{GEMINI_MODEL}` · JSON şemalı çıktı · 4 şıklı çoktan seçmeli")

    source_options = []
    if st.session_state.get("last_summary"):
        source_options.append("Modül 1 — PDF özeti")
    if st.session_state.get("last_pdf_text"):
        source_options.append("Modül 1 — PDF tam metni")
    source_options.append("Metin yapıştır")

    source_choice = st.radio(
        "Kaynak metin",
        source_options,
        key="quiz_source",
        horizontal=True,
    )

    quiz_source_text = ""
    if source_choice == "Modül 1 — PDF özeti":
        quiz_source_text = st.session_state.get("last_summary", "")
        with st.expander("Kaynak: özet (önizleme)"):
            st.markdown(quiz_source_text)
    elif source_choice == "Modül 1 — PDF tam metni":
        quiz_source_text = st.session_state.get("last_pdf_text", "")
        with st.expander(f"Kaynak: PDF tam metni ({len(quiz_source_text):,} karakter)"):
            preview = quiz_source_text[:2000]
            if len(quiz_source_text) > 2000:
                preview += "\n\n... (kalan kısım Gemini'ye gönderilecek)"
            st.text(preview)
    else:
        quiz_source_text = st.text_area(
            "Metin yapıştırın",
            height=200,
            key="quiz_pasted_text",
            placeholder="Ders notu, slayt içeriği veya konu açıklaması...",
        )

    col1, col2 = st.columns([1, 2])
    with col1:
        n_q = st.slider("Soru sayısı", 3, 10, 5, key="quiz_n")
    with col2:
        difficulty = st.radio(
            "Zorluk",
            ["Kolay", "Orta", "Zor"],
            index=1,
            key="quiz_difficulty",
            horizontal=True,
        )

    can_generate = bool(quiz_source_text.strip()) and bool(GEMINI_API_KEY)
    if not GEMINI_API_KEY:
        st.warning("Quiz üretmek için `.env` içine `GEMINI_API_KEY` ekleyin.")
    elif not quiz_source_text.strip():
        st.info("Kaynak metin gerekli (özet, tam PDF veya yapıştırılan metin).")

    if st.button(
        "Quiz Oluştur",
        key="quiz_gen_btn",
        disabled=not can_generate,
        type="primary",
    ):
        with st.spinner("Gemini quiz üretiyor..."):
            try:
                quiz_data = generate_quiz(quiz_source_text, n_q, difficulty)
                assert isinstance(quiz_data, list) and len(quiz_data) > 0, "Boş liste"
                for q in quiz_data:
                    assert all(
                        k in q
                        for k in ("soru", "secenekler", "dogru_cevap_index", "aciklama", "konu")
                    ), "Eksik alan"
                    assert len(q["secenekler"]) == 4, "4 şık olmalı"
                    assert 0 <= q["dogru_cevap_index"] <= 3, "Index 0-3 olmalı"
                st.session_state["quiz"] = quiz_data
                st.session_state["quiz_revealed"] = False
                st.session_state["quiz_difficulty_used"] = difficulty
                for k in list(st.session_state.keys()):
                    if k.startswith("quiz_ans_"):
                        del st.session_state[k]
                if "quiz_score" in st.session_state:
                    del st.session_state["quiz_score"]
                st.rerun()
            except json.JSONDecodeError as e:
                st.error(f"JSON parse hatası: {e}")
            except AssertionError as e:
                st.error(f"Quiz formatı geçersiz ({e}). Tekrar deneyin.")
            except Exception as e:
                st.error(f"Gemini API hatası: {e}")

    quiz = st.session_state.get("quiz")
    if quiz:
        st.divider()
        used_diff = st.session_state.get("quiz_difficulty_used", "?")
        revealed = st.session_state.get("quiz_revealed", False)
        st.markdown(f"### Quiz — {len(quiz)} soru · Zorluk: **{used_diff}**")

        for i, q in enumerate(quiz):
            st.markdown(f"**Soru {i + 1}.** {q['soru']}")
            st.radio(
                "Şıkkı seçin",
                options=list(range(4)),
                format_func=lambda idx, opts=q["secenekler"]: f"{chr(65 + idx)}) {opts[idx]}",
                index=None,
                key=f"quiz_ans_{i}",
                disabled=revealed,
                label_visibility="collapsed",
            )
            if revealed:
                user_idx = st.session_state.get(f"quiz_ans_{i}")
                correct_idx = q["dogru_cevap_index"]
                correct_label = f"{chr(65 + correct_idx)}) {q['secenekler'][correct_idx]}"
                if user_idx == correct_idx:
                    st.success(f"✓ Doğru — {correct_label}")
                elif user_idx is None:
                    st.warning(f"⚠ Cevap verilmedi. Doğru: {correct_label}")
                else:
                    user_label = f"{chr(65 + user_idx)}) {q['secenekler'][user_idx]}"
                    st.error(f"✗ Sizin: {user_label}  ·  Doğru: {correct_label}")
                with st.expander(f"Açıklama · Konu: _{q['konu']}_"):
                    st.write(q["aciklama"])
            st.divider()

        if not revealed:
            if st.button("Cevapları Göster", key="quiz_reveal_btn", type="primary"):
                results = []
                correct = 0
                for i, q in enumerate(quiz):
                    user_idx = st.session_state.get(f"quiz_ans_{i}")
                    is_correct = user_idx == q["dogru_cevap_index"]
                    if is_correct:
                        correct += 1
                    results.append({"konu": q["konu"], "dogru_mu": is_correct})
                st.session_state["quiz_results"] = results
                st.session_state["quiz_score"] = (correct, len(quiz))
                st.session_state["quiz_revealed"] = True
                st.rerun()
        else:
            correct, total = st.session_state.get("quiz_score", (0, len(quiz)))
            pct = round(100 * correct / total) if total else 0
            cscore, cnew = st.columns([1, 1])
            with cscore:
                st.metric("Skor", f"{correct}/{total}", f"%{pct}")
            with cnew:
                st.write("")
                if st.button("Yeni Quiz", key="quiz_new_btn"):
                    for k in ("quiz", "quiz_revealed", "quiz_score", "quiz_difficulty_used"):
                        st.session_state.pop(k, None)
                    for k in list(st.session_state.keys()):
                        if k.startswith("quiz_ans_"):
                            del st.session_state[k]
                    st.rerun()


with tab_wordcloud:
    st.subheader("Türkçe Kelime Bulutu")
    st.caption(
        "Türkçe-aware küçük harf · NLTK TR+EN stopwords + akademik gürültü filtresi · "
        "noktalama/sayı temizleme · matplotlib + wordcloud"
    )

    source_options = []
    if st.session_state.get("last_pdf_text"):
        source_options.append("Modül 1 — PDF tam metni")
    if st.session_state.get("last_summary"):
        source_options.append("Modül 1 — Özet")
    source_options.append("Metin yapıştır")

    source_choice = st.radio(
        "Kaynak metin",
        source_options,
        key="wc_source",
        horizontal=True,
    )

    wc_source_text = ""
    if source_choice == "Modül 1 — PDF tam metni":
        wc_source_text = st.session_state.get("last_pdf_text", "")
        st.caption(f"PDF metni: {len(wc_source_text):,} karakter")
    elif source_choice == "Modül 1 — Özet":
        wc_source_text = st.session_state.get("last_summary", "")
        st.caption(f"Özet metni: {len(wc_source_text):,} karakter")
    else:
        wc_source_text = st.text_area(
            "Metin yapıştırın",
            height=180,
            key="wc_pasted_text",
        )

    c1, c2, c3 = st.columns(3)
    with c1:
        max_words = st.slider(
            "Maks. kelime sayısı", 50, 200, 100, step=10, key="wc_max_words"
        )
    with c2:
        min_len = st.slider(
            "Min. kelime uzunluğu", 3, 6, 3, key="wc_min_len"
        )
    with c3:
        palette = st.selectbox(
            "Renk paleti",
            ["viridis", "plasma", "magma", "Blues", "Greens"],
            key="wc_palette",
        )

    can_build = bool(wc_source_text.strip())
    if not can_build:
        st.info("Kaynak metin gerekli.")

    if st.button(
        "Kelime Bulutu Oluştur",
        key="wc_btn",
        disabled=not can_build,
        type="primary",
    ):
        with st.spinner("Önişleme + bulut üretimi..."):
            try:
                stopwords = get_combined_stopwords()
                tokens = preprocess_for_wordcloud(
                    wc_source_text, min_len, stopwords
                )
                if not tokens:
                    st.error(
                        "Önişleme sonrası kelime kalmadı. "
                        "Daha uzun bir metin veya daha küçük "
                        "min. uzunluk değeri deneyin."
                    )
                else:
                    freq = Counter(tokens)
                    wc = WordCloud(
                        width=1200,
                        height=600,
                        background_color="white",
                        colormap=palette,
                        max_words=max_words,
                        collocations=False,
                    ).generate_from_frequencies(freq)
                    buf = io.BytesIO()
                    wc.to_image().save(buf, format="PNG")
                    st.session_state["wc_png_bytes"] = buf.getvalue()
                    st.session_state["wc_top20"] = freq.most_common(20)
                    st.session_state["wc_stats"] = {
                        "unique": len(freq),
                        "total": sum(freq.values()),
                        "stopwords_n": len(stopwords),
                    }
            except Exception as e:
                st.error(f"Hata: {e}")

    if st.session_state.get("wc_png_bytes"):
        col_img, col_table = st.columns([2, 1])
        with col_img:
            st.image(
                st.session_state["wc_png_bytes"],
                use_container_width=True,
            )
            st.download_button(
                "PNG indir",
                data=st.session_state["wc_png_bytes"],
                file_name="kelime_bulutu.png",
                mime="image/png",
            )
        with col_table:
            st.markdown("**En sık 20 kelime**")
            st.dataframe(
                pd.DataFrame(
                    st.session_state["wc_top20"],
                    columns=["Kelime", "Sıklık"],
                ),
                hide_index=True,
                use_container_width=True,
            )
            s = st.session_state["wc_stats"]
            st.caption(
                f"Farklı kelime: **{s['unique']:,}** · "
                f"Toplam token: **{s['total']:,}** · "
                f"Stopwords (TR+EN+ekstra): **{s['stopwords_n']}**"
            )


with tab_sentiment:
    st.subheader("Not Tonu — Sentiment Analizi")
    st.caption(
        f"Model: `{SENTIMENT_MODEL}` · Cümle bazlı, truncation=512 token · "
        "Ders notundaki risk/uyarı ifadelerinin yoğunluğu — negatif tondaki "
        "cümleler genellikle sınavda dikkat edilmesi gereken kavramları işaret eder."
    )

    sent_source_options = []
    if st.session_state.get("last_pdf_text"):
        sent_source_options.append("Modül 1 — PDF tam metni")
    if st.session_state.get("last_summary"):
        sent_source_options.append("Modül 1 — Özet")
    sent_source_options.append("Metin yapıştır")

    sent_source_choice = st.radio(
        "Kaynak metin",
        sent_source_options,
        key="sent_source",
        horizontal=True,
    )

    sent_source_text = ""
    if sent_source_choice == "Modül 1 — PDF tam metni":
        sent_source_text = st.session_state.get("last_pdf_text", "")
        st.caption(f"PDF metni: {len(sent_source_text):,} karakter")
    elif sent_source_choice == "Modül 1 — Özet":
        sent_source_text = st.session_state.get("last_summary", "")
        st.caption(f"Özet metni: {len(sent_source_text):,} karakter")
    else:
        sent_source_text = st.text_area(
            "Cümle veya paragraf",
            height=180,
            key="sent_pasted_text",
        )

    can_analyze = bool(sent_source_text.strip())
    if not can_analyze:
        st.info("Kaynak metin gerekli.")

    if st.button(
        "Analiz Et",
        key="sent_btn",
        disabled=not can_analyze,
        type="primary",
    ):
        try:
            sentences = split_sentences(sent_source_text)
            if not sentences:
                st.error("Metinden cümle çıkarılamadı.")
            else:
                truncated = len(sentences) > MAX_SENTENCES
                if truncated:
                    st.warning(
                        f"Metin {len(sentences)} cümle içeriyor — "
                        f"performans için ilk **{MAX_SENTENCES}** cümle analiz edilecek."
                    )
                    sentences = sentences[:MAX_SENTENCES]

                with st.spinner(f"{len(sentences)} cümle BERT ile analiz ediliyor..."):
                    pipe = load_sentiment_pipeline()
                    results = pipe(
                        sentences,
                        truncation=True,
                        max_length=512,
                        batch_size=8,
                    )

                pos = sum(1 for r in results if r["label"] == "positive")
                neg = len(results) - pos
                avg_score = sum(r["score"] for r in results) / len(results)

                fig, ax = plt.subplots(figsize=(4, 4))
                ax.pie(
                    [pos, neg],
                    labels=["Pozitif", "Negatif"],
                    colors=["#2ecc71", "#e74c3c"],
                    autopct="%1.1f%%",
                    startangle=90,
                    textprops={"fontsize": 11},
                )
                ax.axis("equal")
                pie_buf = io.BytesIO()
                fig.savefig(pie_buf, format="PNG", bbox_inches="tight", dpi=120)
                plt.close(fig)

                st.session_state["sent_data"] = {
                    "sentences": sentences,
                    "labels": [r["label"] for r in results],
                    "scores": [r["score"] for r in results],
                    "pos": pos,
                    "neg": neg,
                    "avg_score": avg_score,
                    "pie_png": pie_buf.getvalue(),
                }
                st.session_state["last_sentiment_summary"] = {
                    "pos": pos, "neg": neg, "total": len(results),
                    "avg_score": avg_score,
                }
        except Exception as e:
            st.error(f"Hata: {e}")

    sent_data = st.session_state.get("sent_data")
    if sent_data:
        total = len(sent_data["sentences"])
        pos = sent_data["pos"]
        neg = sent_data["neg"]
        pos_ratio = 100 * pos / total
        avg_score = sent_data["avg_score"]

        m1, m2, m3 = st.columns(3)
        m1.metric(
            "Genel Ton (Pozitif %)", f"%{pos_ratio:.0f}",
            delta=f"{pos} poz / {neg} neg", delta_color="off",
        )
        m2.metric("Toplam Cümle", f"{total}")
        m3.metric("Ortalama Güven", f"%{avg_score * 100:.1f}")

        col_pie, col_table = st.columns([1, 2])
        with col_pie:
            st.image(sent_data["pie_png"], use_container_width=True)
        with col_table:
            df = pd.DataFrame({
                "Cümle": sent_data["sentences"],
                "Tahmin": [
                    "✅ Pozitif" if lbl == "positive" else "⚠️ Negatif"
                    for lbl in sent_data["labels"]
                ],
                "Skor": sent_data["scores"],
            })
            st.dataframe(
                df,
                hide_index=True,
                use_container_width=True,
                height=400,
                column_config={
                    "Cümle": st.column_config.TextColumn(width="large"),
                    "Tahmin": st.column_config.TextColumn(width="small"),
                    "Skor": st.column_config.ProgressColumn(
                        min_value=0.0, max_value=1.0, format="%.3f",
                    ),
                },
            )


with tab_weak:
    st.subheader("Performans Analizi + Kişisel Çalışma Planı")
    st.caption(
        "Quiz performansını analiz eder, zayıf konuları PDF ile eşleştirir, "
        "kişisel çalışma planı üretir. Kullanılan teknikler: scikit-learn "
        "TF-IDF + cosine similarity, matplotlib, Gemini LLM."
    )

    quiz_results = st.session_state.get("quiz_results")
    if not quiz_results:
        st.warning("Önce **Modül 2**'de bir quiz çöz, sonra buraya gel.")
        st.info(
            "Yukarıdaki **2. Quiz Üretici** sekmesine geç → quiz oluştur → "
            "cevapları göster. Sonra bu sekme analiz için hazır olacak."
        )
    else:
        quiz_key = json.dumps(quiz_results, sort_keys=True)
        if st.session_state.get("topic_stats_key") != quiz_key:
            df_quiz = pd.DataFrame(quiz_results)
            topic_stats = (
                df_quiz.groupby("konu")
                .agg(total=("dogru_mu", "count"), correct=("dogru_mu", "sum"))
                .reset_index()
            )
            topic_stats["accuracy"] = (
                100.0 * topic_stats["correct"] / topic_stats["total"]
            )
            topic_stats = topic_stats.sort_values("accuracy", ascending=True)

            fig_h = max(4, len(topic_stats) * 0.6)
            fig, ax = plt.subplots(figsize=(10, fig_h))
            colors = [accuracy_color(a) for a in topic_stats["accuracy"]]
            ax.barh(topic_stats["konu"], topic_stats["accuracy"], color=colors)
            ax.set_xlim(0, 110)
            ax.set_xlabel("Doğruluk (%)")
            ax.axvline(60, color="#888", linestyle="--", linewidth=0.8, alpha=0.6)
            ax.axvline(80, color="#888", linestyle="--", linewidth=0.8, alpha=0.6)
            for i, (acc, total, correct) in enumerate(
                zip(
                    topic_stats["accuracy"],
                    topic_stats["total"],
                    topic_stats["correct"],
                )
            ):
                ax.text(
                    acc + 2, i, f"{int(correct)}/{int(total)}",
                    va="center", fontsize=9,
                )
            plt.tight_layout()
            chart_buf = io.BytesIO()
            fig.savefig(chart_buf, format="PNG", bbox_inches="tight", dpi=120)
            plt.close(fig)

            st.session_state["topic_stats_df"] = topic_stats
            st.session_state["topic_stats_png"] = chart_buf.getvalue()
            st.session_state["topic_stats_key"] = quiz_key
            st.session_state.pop("tfidf_match_key", None)

        topic_stats = st.session_state["topic_stats_df"]

        st.markdown("### Konu Bazlı Doğruluk")
        col_chart, col_metric = st.columns([3, 1])
        with col_chart:
            st.image(
                st.session_state["topic_stats_png"],
                use_container_width=True,
            )
        with col_metric:
            total_q = int(topic_stats["total"].sum())
            total_c = int(topic_stats["correct"].sum())
            overall_pct = 100 * total_c / total_q
            st.metric("Genel Başarı", f"%{overall_pct:.0f}", f"{total_c}/{total_q}")
            n_weak = int((topic_stats["accuracy"] < 60).sum())
            n_strong = int((topic_stats["accuracy"] >= 80).sum())
            st.metric("Zayıf Konu (<%60)", n_weak)
            st.metric("Güçlü Konu (≥%80)", n_strong)

        st.divider()
        st.markdown("### Çalışman Önerilen Bölümler")
        st.caption("TF-IDF + cosine similarity ile zayıf konu adı en alakalı PDF paragraflarıyla eşleştirilir.")

        weak_df = topic_stats[topic_stats["accuracy"] < 60].copy()
        if weak_df.empty:
            st.success("Tüm konularda %60 üstü başarı — zayıf konu yok.")
        else:
            pdf_text = st.session_state.get("last_pdf_text", "")
            if not pdf_text:
                st.info(
                    "PDF bağlantılı paragraf önerisi için **Modül 1**'de "
                    "bir PDF yükleyin."
                )
            else:
                weak_konular = tuple(weak_df["konu"].tolist())
                match_key = (quiz_key, len(pdf_text), weak_konular)
                if st.session_state.get("tfidf_match_key") != match_key:
                    paragraphs = [
                        p.strip() for p in pdf_text.split("\n\n")
                        if len(p.strip()) >= 50
                    ]
                    matches: dict[str, list[tuple[float, str]]] = {}
                    if paragraphs:
                        pre_paras = [turkish_lower(p) for p in paragraphs]
                        vec = TfidfVectorizer(
                            lowercase=False, ngram_range=(1, 2), min_df=1,
                        )
                        matrix = vec.fit_transform(pre_paras)
                        for konu in weak_konular:
                            q_vec = vec.transform([turkish_lower(konu)])
                            sims = cosine_similarity(q_vec, matrix).flatten()
                            top_idx = sims.argsort()[-2:][::-1]
                            matches[konu] = [
                                (float(sims[i]), paragraphs[i])
                                for i in top_idx
                            ]
                    st.session_state["tfidf_matches"] = matches
                    st.session_state["tfidf_match_key"] = match_key

                matches = st.session_state.get("tfidf_matches", {})
                if not matches:
                    st.warning("PDF metninden anlamlı paragraf çıkarılamadı.")
                else:
                    for _, row in weak_df.iterrows():
                        konu = row["konu"]
                        acc = row["accuracy"]
                        with st.expander(
                            f"**{konu}** — %{acc:.0f} "
                            f"({int(row['correct'])}/{int(row['total'])})"
                        ):
                            shown = False
                            for sim, para in matches.get(konu, []):
                                if sim > 0:
                                    st.markdown(f"**Benzerlik:** `{sim:.3f}`")
                                    st.markdown(f"> {para}")
                                    shown = True
                            if not shown:
                                st.caption(
                                    "Bu konu için PDF'te eşleşen "
                                    "paragraf bulunamadı."
                                )

        st.divider()
        st.markdown("### Kişisel Çalışma Planı")

        c1, c2 = st.columns(2)
        with c1:
            exam_date = st.date_input(
                "Sınav tarihi",
                value=date.today() + timedelta(days=7),
                key="plan_exam_date",
            )
        with c2:
            hours = st.slider(
                "Günlük çalışma saati", 1, 8, 2, key="plan_hours"
            )

        days_left = (exam_date - date.today()).days
        if days_left <= 0:
            st.warning(
                "Sınav tarihi bugün veya geçmişte. Lütfen ileri bir tarih seçin."
            )
        else:
            st.caption(
                f"Sınava **{days_left} gün** kaldı · "
                f"Toplam çalışma bütçesi: **{days_left * hours} saat**"
            )

            can_plan = bool(GEMINI_API_KEY)
            if not GEMINI_API_KEY:
                st.warning(
                    "Plan üretmek için `.env` içine `GEMINI_API_KEY` ekleyin."
                )

            if st.button(
                "Plan Oluştur",
                key="plan_btn",
                disabled=not can_plan,
                type="primary",
            ):
                strong_df = topic_stats[topic_stats["accuracy"] >= 80].copy()
                with st.spinner("Gemini çalışma planı hazırlıyor..."):
                    try:
                        plan_md = generate_study_plan(
                            days=days_left,
                            hours=hours,
                            weak_df=weak_df,
                            strong_df=strong_df,
                        )
                        st.session_state["last_study_plan"] = plan_md
                    except Exception as e:
                        st.error(f"Gemini API hatası: {e}")

            plan_md = st.session_state.get("last_study_plan")
            if plan_md:
                st.markdown(plan_md)
                st.download_button(
                    "Planı .md olarak indir",
                    data=plan_md.encode("utf-8"),
                    file_name=f"calisma_plani_{exam_date.isoformat()}.md",
                    mime="text/markdown",
                )


st.divider()
st.caption("v0.1 — iskelet | SmartStudy AI")
