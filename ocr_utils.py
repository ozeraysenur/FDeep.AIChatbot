import easyocr
from PIL import Image
import numpy as np

def extract_text_from_image(image: Image.Image):
    reader = easyocr.Reader(['en', 'tr'], gpu=False)
    img_array = np.array(image)
    result = reader.readtext(img_array)
    texts = [item[1] for item in result]
    return "\n".join(texts)
