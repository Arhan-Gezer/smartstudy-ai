# SmartStudy AI

**ECS-VB-YZ (Yapay Zeka ve Veri Bilimi) — Dönem Sonu Projesi**

PDF ders notlarını yapay zeka ile analiz eden, öğrencilere kişiselleştirilmiş çalışma yardımı sunan Streamlit uygulaması.

---

## 🎯 Proje Özeti

SmartStudy AI, öğrencilerin uzun ders notlarını verimli bir şekilde çalışmasına yardım eden 5 modüllü bir AI asistanıdır. Bir PDF yükleyerek özet alabilir, otomatik quiz çözebilir, kavram bulutu görebilir, metnin tonunu analiz edebilir ve kişisel çalışma planı oluşturabilirsiniz.

---

## 🧩 Modüller

| # | Modül | Teknoloji | Ne Yapar |
|---|---|---|---|
| 1 | **PDF Özet** | PyPDF2 + Google Gemini | PDF'ten metin çıkarır, Türkçe akademik özet üretir |
| 2 | **Quiz Üretici** | Gemini (JSON schema) | Çoktan seçmeli 3-10 soru, konu etiketli |
| 3 | **Kelime Bulutu** | NLTK + WordCloud + matplotlib | Türkçe + İngilizce stopwords filtreli kavram görselleştirmesi |
| 4 | **Sentiment Analizi** | HuggingFace BERT (Türkçe) | Cümle bazlı pozitif/negatif ton tespiti |
| 5 | **Performans + Çalışma Planı** | scikit-learn TF-IDF + cosine similarity + Gemini | Zayıf konuları tespit eder, PDF paragraflarıyla eşler, kişisel çalışma planı üretir |

---

## 🎓 Derste Öğrenilen Tekniklerin Kullanımı

| Konu | Modül | Açıklama |
|---|---|---|
| Veri görselleştirme (pandas, matplotlib) | 3, 5 | Bar chart, pasta grafik, wordcloud |
| NLP ön işleme | 3 | Tokenization, stopwords (TR+EN), Türkçe-aware normalizasyon |
| Klasik ML (TF-IDF + cosine similarity) | 5 | Bilgi getirimi: zayıf konu → ilgili PDF paragrafı eşleştirme |
| Transformer modelleri ve transfer learning | 4 | Pretrained Türkçe BERT ile inference |
| LLM API entegrasyonu | 1, 2, 5 | Google Gemini, structured output, prompt engineering |

---

## 🛠️ Teknoloji Yığını

- **Frontend & Runtime:** Streamlit
- **PDF İşleme:** PyPDF2
- **LLM:** Google Gemini (`gemini-flash-latest`)
- **NLP Modeli:** [`savasy/bert-base-turkish-sentiment-cased`](https://huggingface.co/savasy/bert-base-turkish-sentiment-cased)
- **ML & NLP Araçları:** scikit-learn, NLTK, HuggingFace Transformers, PyTorch, WordCloud
- **Görselleştirme:** matplotlib

---

## 🚀 Kurulum

### 1. Repo'yu klonla

```bash
git clone https://github.com/Arhan-Gezer/smartstudy-ai.git
cd smartstudy-ai
```

### 2. Sanal ortam oluştur

```bash
python -m venv .venv
.\.venv\Scripts\activate          # Windows
# source .venv/bin/activate       # macOS / Linux
```

### 3. Bağımlılıkları yükle

```bash
pip install -r requirements.txt
```

### 4. API key ayarla

- [Google AI Studio](https://aistudio.google.com/apikey) üzerinden Gemini API key oluştur
- `.env.example` dosyasını `.env` olarak kopyala
- Key'i yapıştır:

```env
GEMINI_API_KEY=AIzaSy...senin_key...
```

### 5. Çalıştır

```bash
streamlit run app.py
```

Tarayıcı otomatik açılır: `http://localhost:8501`

---

## 📖 Kullanım

1. **Modül 1**: Sol üstten PDF yükle → "Özet Çıkar"
2. **Modül 2**: Kaynak olarak "PDF tam metni" seç → soru sayısı/zorluk ayarla → "Quiz Oluştur" → soruları cevapla → "Cevapları Göster"
3. **Modül 3**: "Kelime Bulutu Oluştur" → PNG olarak indirilebilir
4. **Modül 4**: "Analiz Et" → cümle bazlı pasta + tablo
5. **Modül 5**: Quiz çözdükten sonra → konu doğruluk grafiği + zayıf konu paragrafları + sınav tarihi ile kişiselleştirilmiş plan

> Modüller birbirini besler: bir PDF yüklendikten sonra tüm modüller aynı kaynağı kullanır.

---

## 🧠 Önemli Teknik Kararlar

- **`generate_from_frequencies()` kullanımı (WordCloud)**: Default `generate()` İngilizce collocation detection yapar; Türkçe için gürültü oluşturur. Frekans tabanlı API daha sağlam sonuç verir.
- **TR + EN stopwords birleştirme**: Akademik PDF'lerde Türkçe açıklama + İngilizce kaynak terimleri karışık olduğu için iki dilin stopword listesi NLTK'dan birleştirildi, akademik dolgu kelimeleri ek olarak filtrelendi.
- **Türkçe-aware lowercase**: Python'un default `.lower()` Türkçe 'İ' harfini bozar. Manuel düzeltme yapıldı.
- **Gemini structured output**: `response_mime_type='application/json'` + manuel şema validasyonu ile JSON çıktısı garantilendi.
- **State-guarded cache**: Streamlit rerun'larında pahalı işlemler (PDF extract, wordcloud, sentiment inference) `session_state` anahtarlarıyla cache'lendi.
- **Transfer learning**: BERT modeli sıfırdan eğitilmedi; HuggingFace `pipeline()` ile pretrained model inference için kullanıldı.

---

## 📂 Proje Yapısı

```
smartstudy-ai/
├── app.py                  # Tüm modülleri içeren Streamlit uygulaması
├── requirements.txt        # Python bağımlılıkları
├── .env.example            # API key şablonu
├── .gitignore
└── README.md
```

---

## ⚙️ Sınırlılıklar

- Sentiment modeli film/ürün yorumu üzerine fine-tune edildiği için ironi ve negation'da hata yapabilir
- Gemini free tier kullanılıyor, günlük limit dolabilir
- PyPDF2 taranmış (görüntü tabanlı) PDF'leri okuyamaz, OCR entegrasyonu yok
- TF-IDF runtime'da fit ediliyor; çok büyük PDF'lerde performans düşebilir

---

## 🔮 Gelecek İyileştirmeler

- **RAG (Retrieval-Augmented Generation)**: Quiz hallüsinasyonunu önlemek için embedding tabanlı retrieval
- **OCR desteği**: Taranmış PDF'ler için Tesseract entegrasyonu
- **Lokal LLM seçeneği**: Veri gizliliği için Ollama / llama.cpp alternatifi
- **Çoklu kullanıcı**: Streamlit yerine FastAPI + frontend ayrımı
- **Flashcard ve sesli okuma**: Mobil destekli çalışma araçları

---

## 👤 Geliştirici

**Arhan Gezer** — Computer Engineering, ECS-VB-YZ Final Project

---

## 📄 Lisans

Akademik kullanım içindir.
