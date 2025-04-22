from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

model = load_model('asl_model.keras')

def prepare_image(img_path):
    img = load_img(img_path, target_size=(64, 64))
    img_array = img_to_array(img)
    img_array = (img_array - 127.5) / 127.5
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

image = prepare_image('imagen1.jpg')
prediction = model.predict(image)
predicted_class = np.argmax(prediction, axis=1)[0]

class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
print(f"La letra predicha es: {class_names[predicted_class]}")