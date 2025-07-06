import os
import torch
from torchvision import models, transforms
from PIL import Image
import urllib.request

# 🌐 imagenet_classes.txt yoksa otomatik indir
if not os.path.exists("imagenet_classes.txt"):
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    urllib.request.urlretrieve(url, "imagenet_classes.txt")

# 🧠 Model ve sınıf etiketleri yalnızca bir kez yüklensin
model = models.resnet18(pretrained=True)
model.eval()

# 🔠 ImageNet sınıf etiketlerini oku
with open("imagenet_classes.txt") as f:
    labels = [line.strip() for line in f.readlines()]

# 🎨 Görsel ön işleme dönüşümleri
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 🧩 Ana fonksiyon
def classify_image(image: Image.Image) -> str:
    input_tensor = preprocess(image).unsqueeze(0)  # (1, 3, 224, 224)
    with torch.no_grad():
        output = model(input_tensor)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top_class = torch.argmax(probabilities).item()
    label = labels[top_class]
    confidence = probabilities[top_class].item()
    return f"{label} ({confidence:.2%} güven)"
