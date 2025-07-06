import streamlit as st
from PIL import Image
from rag_pipeline import load_and_translate_documents, create_vectorstore, retrieve_relevant_chunks
from model_utils import load_model, generate_response
from llava_utils import load_llava_model, analyze_image, load_blip2_model, analyze_with_blip2
from deep_translator import GoogleTranslator


st.set_page_config(page_title="Dijital Pazarlama Chatbot'u", layout="wide")

st.image("FDEEP.AI.png", width=300)
st.title("📈 Dijital Pazarlama ve Dijital Reklamcılık Chatbot'u 🤖 ")
st.markdown("Kişisel dijital pazarlama asistanınız 🤖 ")

# Belge klasörü yolu
PDF_FOLDER = r"C:\Users\feyza\OneDrive\Masaüstü\teknofest_chatbot\dijitalpazarlama_reklamkaynakları"

# Belgeleri yükle ve vektör oluştur
with st.spinner("Belgeler işleniyor..."):
    documents = load_and_translate_documents(PDF_FOLDER)
    vectorstore = create_vectorstore(documents)

# Model seçimi
model_key = st.selectbox("Kullanmak istediğiniz modeli seçin:", ["mistral", "llama3.1.8", "qwen3", "gemma3", "deepseek-r1"])
model_name = load_model(model_key)

# 🧠 Chat Arayüzü
st.subheader("💬 Soru-Cevap (Chatbot)")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Dijital pazarlama ve dijital reklamcılık ile ilgili sorunuzu yazın:")

if st.button("Gönder"):
    if not user_input.strip():
        st.warning("Lütfen bir soru yazın.")
    else:
        relevant_docs = retrieve_relevant_chunks(vectorstore, user_input)
        context = "\n".join(relevant_docs)
        prompt = f"Belgelere dayalı olarak yanıt ver. Cevap Türkçe olsun ve sade bir dille yaz:\n\nKullanıcının sorusu: {user_input}\n\nİlgili belgeler:\n{context}"

        with st.spinner("Yanıt oluşturuluyor..."):
            answer = generate_response(model_name, prompt)

        st.session_state.chat_history.append(("🧑‍💻 Soru", user_input))
        st.session_state.chat_history.append(("🤖 Yanıt", answer))

# Sohbet geçmişini göster
for role, msg in st.session_state.chat_history:
    st.markdown(f"*{role}:* {msg}")

# 🖼️ Görsel Yorumlama Asistanı (Yalnızca Ücretsiz Lokal Modeller)

st.subheader("🖼️ Görsel Yorumlama Asistanı")

model_selector = st.selectbox(
    "Görsel yorumlama için model seçin:",
    [
        "llava-hf (lokal)",
        "blip2-hf (lokal)",
        "ocr (görseldeki yazıları oku)",
        "classification (ne resmi olduğunu tahmin et)"
    ]
)

uploaded_image = st.file_uploader("Bir görsel yükleyin", type=["jpg", "jpeg", "png"])
question = st.text_input("Görselle ilgili ne öğrenmek istiyorsunuz?")

if uploaded_image:
    image = Image.open(uploaded_image).convert("RGB")

    if model_selector == "llava-hf (lokal)" and question:
        with st.spinner("LLaVA v1.6 modeli yükleniyor..."):
            processor, llava_model = load_llava_model()
        with st.spinner("Görsel analiz ediliyor..."):
            output = analyze_image(processor, llava_model, image, question)
        st.success("LLaVA Cevabı:")
        st.write(output)

    elif model_selector == "blip2-hf (lokal)" and question:
        with st.spinner("BLIP-2 modeli yükleniyor..."):
            blip_processor, blip_model = load_blip2_model()
        with st.spinner("BLIP-2 analiz ediyor..."):
            output = analyze_with_blip2(blip_processor, blip_model, image, question)
        st.success("BLIP-2 (İngilizce) Tanım:")
        st.write(output)

        from llava_utils import translate_and_summarize
        st.markdown("**💬 Türkçeye Çevrilmiş Pazarlama Metni:**")
        marketing_text = translate_and_summarize(output, model_name)
        st.write(marketing_text)

    elif model_selector == "ocr (görseldeki yazıları oku)":
        with st.spinner("Metin algılanıyor (OCR)..."):
            from ocr_utils import extract_text_from_image
            ocr_text = extract_text_from_image(image)
        st.success("OCR Sonucu:")
        st.code(ocr_text if ocr_text else "Metin bulunamadı.")

    elif model_selector == "classification (ne resmi olduğunu tahmin et)":
        with st.spinner("Görsel sınıflandırılıyor..."):
            from image_classification_utils import classify_image
            label = classify_image(image)
        st.success(f"🖼️ Tahmini sınıf:")
        st.write(label)

        # Çıktıyı göster
    st.image(image, caption="Yüklenen Görsel", use_container_width=True)
    if model_selector in ["llava-hf (lokal)", "blip2-hf (lokal)"] and question:
        st.markdown("**Yanıt:**")
        st.write(output)

"""
OCR seçildiyse → Resimdeki metni çıkarır → sana sunar.

Sınıflandırma seçildiyse → Resmin ne olduğunu söyler. (“fincan” gibi)

LLaVA seçildiyse → “Fincan” + “reklam önerisi” gibi sorulara zengin cevap verir.

BLIP-2 seçildiyse → Görselin özeti + fikir üretir, bu da pazarlama içerikleri için kullanışlıdır. """