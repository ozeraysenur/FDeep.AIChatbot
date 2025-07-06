# FDeep.AIChatbot Dijital Pazarlama ve Reklam Asistanı (Teknofest 2025)

## Proje Tanımı

*FDeepAI, KOBİ'ler ve bireysel girişimciler için ajanslara olan bağımlılığı azaltmayı hedefleyen, **doğal dil işleme* ve *çok modelli yapay zeka* tabanlı bir asistandır.  

Sistem:  
- Dijital pazarlama ve reklamcılık alanında bilgi sunar.  
- Yüklenen *PDF belgeleri* üzerinden RAG (Retrieval-Augmented Generation) ile analiz yapar ve Türkçe yanıtlar üretir.  
- *Görsel yorumlama* (resimden açıklama üretme) ve *OCR (görselden yazı okuma)* özelliklerine sahiptir.  
- Görsellerden elde edilen bilgilerle *metin sınıflandırması* yapar ve pazarlama odaklı açıklamalar üretir.  
- Teknik bilgisi sınırlı kullanıcıların bile profesyonel kampanya içerikleri hazırlamasına yardımcı olur.  

Tüm sistem *yerel ortamda çalışır* ve bu sayede *veri gizliliği %100 güvence* altına alınır.  
Bu özellik, veri güvenliğine öncelik veren büyük ölçekli şirketler için de önemli bir tercih sebebidir.

---

## Kullanılan Teknolojiler ve Mimariler  

- *Retrieval-Augmented Generation (RAG)* mimarisi  
- *LangChain*: Belge bölme, sorgulama zinciri, RAG yapılandırması  
- *FAISS + SentenceTransformers*: Vektör tabanlı belge arama  
- *Ollama + LLaMA.cpp*: Yerel LLM çalıştırma (Mistral, LLaMA, Qwen, Gemma, DeepSeek destekli)  
- *LLaVA 1.5 / Onevision*: Görselden içerik önerisi
- *BLIP-2*: Görselden açıklama üretme  
- *OCR (Tesseract)*: Görsellerdeki yazılı metni tanıma ve çıkartma  
- *Metin Sınıflandırma (Text Classification)*: Görseldeki nesneyi tanıma
- *Streamlit*: Web tabanlı kullanıcı arayüzü  
- *Python 3.10+*, torch (CUDA destekli), langchain, sentence-transformers, transformers, deep-translator  

---

## Kurulum Talimatları

1. Ortam Kurulumu

'''bash
git clone https://github.com/ozeraysenur/FDeep.AIChatbot.git
cd FDeep.AIChatbot
python -m venv .venv
.venv\Scripts\activate  # MacOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
'''
------------

Takım Adı: FDeep.AI
Teknofest 2025 Türkçe Doğal Dil İşleme Yarışması Serbest Kategori için oluşturulmuştur.

------------

Takım Üyeleri
Feyza Kıranlıoğlu GitHub: https://github.com/feyzakir
Derin Çıvgın GitHub: https://github.com/Derincvgn
Ayşe Nur Özer GitHub: https://github.com/ozeraysenur
Ləman Osmanlı GitHub: https://github.com/Leman2006

------------

2. PDF Veri Setini Ekleme
Proje, `dijitalpazarlama_reklamkaynakları` adlı klasör altında PDF belgesiyle birlikte gelir.
Bu klasör, `main.py` çalıştırıldığında otomatik olarak yüklenir. Ekstra bir bağlantıya ihtiyaç yoktur.

 ------------

3. Uygulamayı Başlatma
bash
streamlit run main.py 
