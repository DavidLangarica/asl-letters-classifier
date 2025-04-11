# Reconocimiento de Lengua de Señas Americana (ASL)

## Descripción

Este proyecto implementa un modelo de detección de gestos para la lengua de señas americana (ASL) utilizando redes neuronales convolucionales (CNN). El modelo es capaz de clasificar imágenes de gestos manuales representando las primeras siete letras del alfabeto ASL (A, B, C, D, E, F, G).

**Autor:** David René Langarica Hernández | A01708936

## Conjunto de Datos

El conjunto de datos utilizado para entrenar este modelo es una fusión de tres datasets diferentes:

- [American Sign Language](https://www.kaggle.com/datasets/kapillondhe/american-sign-language)
- [ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- [American Sign Language Dataset](https://www.kaggle.com/datasets/ayuraj/asl-dataset)

**Nota importante:** En este repositorio se incluye una muestra del conjunto de datos (`dataset_sample/`), pero no el conjunto completo debido a limitaciones de espacio. La muestra es suficiente para entender la estructura de los datos, pero para reproducir completamente los resultados se recomienda descargar los datasets originales.

### División de Datos

Los datos fueron divididos de la siguiente manera:

- 70% para entrenamiento
- 10% para validación
- 20% para pruebas

## Estructura del Proyecto

```
asl/
│
├── dataset/             # Conjunto de datos completo
│   ├── train/          # Imágenes de entrenamiento (70%)
│   ├── validation/     # Imágenes de validación (10%)
│   └── test/           # Imágenes de prueba (20%)
│
├── dataset_sample/      # Muestra del conjunto de datos
│   ├── train/          # Muestra de imágenes de entrenamiento
│   ├── validation/     # Muestra de imágenes de validación
│   └── test/           # Muestra de imágenes de prueba
│
├── README.md            # Este archivo
├── ASL_Model.ipynb      # Notebook con el código del modelo
└── asl_model.keras      # Modelo entrenado guardado
```

## Requisitos

Para ejecutar este proyecto, necesitas tener instalado:

- Python 3.7+
- TensorFlow 2.x
- NumPy
- Matplotlib
- scikit-learn
- seaborn

Puedes instalar todas las dependencias con:

```
pip install tensorflow numpy matplotlib scikit-learn seaborn
```

## Uso del Modelo

Para utilizar el modelo entrenado:

```python
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

model = load_model('asl_model.keras')

def prepare_image(img_path):
    img = load_img(img_path, target_size=(64, 64))
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

image = prepare_image('ruta/a/tu/imagen.jpg')
prediction = model.predict(image)
predicted_class = np.argmax(prediction, axis=1)[0]

class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
print(f"La letra predicha es: {class_names[predicted_class]}")
```

## Rendimiento del Modelo

El modelo alcanza una precisión de aproximadamente 99.9% en el conjunto de prueba, lo que indica un rendimiento excelente en la clasificación de las letras ASL incluidas en este proyecto.

## Limitaciones

Este modelo está entrenado únicamente para reconocer las primeras siete letras del alfabeto ASL (A-G). Para un sistema completo de reconocimiento, sería necesario extender el conjunto de datos para incluir todas las letras y posiblemente números y otros gestos comunes.
