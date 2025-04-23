from rembg import remove
from PIL import Image
import io
import numpy as np
from tensorflow.keras.models import load_model

class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
model = load_model('asl_model1.keras')
IMAGE_SIZE = (64, 64)

def preprocess_image(image_path, remove_background=True):
    with open(image_path, 'rb') as f:
        input_image = f.read()
    
    if remove_background:
        output_image = remove(input_image)
        image = Image.open(io.BytesIO(output_image)).convert('RGBA')
        white_background = Image.new('RGBA', image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(white_background, image)
        image = image.convert('RGB')
    else:
        image = Image.open(io.BytesIO(input_image)).convert('RGB')
    
    image = image.resize(IMAGE_SIZE)
    image_array = np.array(image) / 255.0
    
    return np.expand_dims(image_array, axis=0)

def predict_image(image_path, remove_background=True):
    processed_image = preprocess_image(image_path, remove_background)
    
    prediction = model.predict(processed_image)
    pred_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    
    print(f'Predicted Class: {class_names[pred_class]}')
    print(f'Confidence: {confidence:.2f}%')
    
image_path = 'e1.JPEG'
predict_image(image_path)