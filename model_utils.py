import subprocess

def load_model(model_key):
    supported_models = {
        "mistral": "mistral:latest",
        "qwen3": "qwen3:8b",
        "gemma3": "gemma3:latest",
        "deepseek-r1": "deepseek-r1:8b",
        "llama3.1.8": "llama3.1:8b"
    }
    if model_key not in supported_models:
        raise ValueError(f"'{model_key}' desteklenmiyor. Şunlardan birini kullanın: {list(supported_models.keys())}")
    return supported_models[model_key]

def generate_response(model_name, prompt):
    try:
        result = subprocess.run(
            ["ollama", "run", model_name, prompt],
            capture_output=True,
            text=True,
            encoding='utf-8',  # <<< BURASI önemli
            errors='replace'   # bozuk karakterleri ? ile değiştirir
        )

        if result.returncode != 0:
            return f"Hata oluştu: {result.stderr}"

        if not result.stdout.strip():
            return "Model boş yanıt döndürdü."

        return result.stdout.strip()
    except Exception as e:
        return f"Hata oluştu: {str(e)}"