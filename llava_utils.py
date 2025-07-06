from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
from deep_translator import GoogleTranslator
from model_utils import generate_response, load_model
from transformers import LlavaProcessor
from transformers import LlavaNextProcessor, LlavaOnevisionForConditionalGeneration,AutoProcessor

def load_llava_model():
    model_id = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
    
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,

    ).to(0 if torch.cuda.is_available() else "cpu")

    processor = AutoProcessor.from_pretrained(model_id)
    return processor, model

def analyze_image(processor, model, image: Image.Image, question: str):
    # Konuşma şablonu
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image"},
            ],
        }
    ]
    
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device, torch.float16)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=150, do_sample=False)

    return processor.decode(outputs[0][2:], skip_special_tokens=True)
# 🔹 BLIP-2 Modelini Yükleme (GPU destekli)
def load_blip2_model():
    model_id = "Salesforce/blip-image-captioning-base"
    processor = BlipProcessor.from_pretrained(model_id)
    model = BlipForConditionalGeneration.from_pretrained(model_id)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return processor, model

# 🔹 BLIP-2 Görsel Analizi
def analyze_with_blip2(processor, model, image: Image.Image, question: str = "What is in the image?"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = processor(images=image, text=question, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=100)
    return processor.tokenizer.decode(out[0], skip_special_tokens=True)

def translate_and_summarize(english_caption: str, selected_model_name: str):
    # 1. İngilizce'yi Türkçeye çevir
    try:
        turkish_caption = GoogleTranslator(source='en', target='tr').translate(english_caption)
    except Exception as e:
        return f"Çeviri sırasında hata oluştu: {str(e)}"
    
    # 2. Pazarlama odaklı içerik üret
    prompt = f"""
Aşağıdaki açıklamayı bir pazarlama metnine dönüştür.
1. Ürünün duygusal veya işlevsel faydasını vurgula.
2. Hedef kitleyi düşünerek yaz.
3. Etkileyici bir çağrı (CTA - Call To Action) ekle.
4. Sonuç Türkçe ve sade bir dille olsun.

Ürün açıklaması: {turkish_caption}
"""
    try:
        summary = generate_response(selected_model_name, prompt)
        return summary
    except Exception as e:
        return f"Pazarlama metni üretilemedi: {str(e)}"
