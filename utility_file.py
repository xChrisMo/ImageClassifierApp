import numpy as np
import tensorflow as tf
from PIL import Image
import json
import tensorflow_hub as hub

image_size = 224

def load_mapping(mapping='label_map.json'):
    with open(mapping, 'r') as f:
        class_names = json.load(f)
    return class_names

def process_image(image):
    image = tf.convert_to_tensor(image)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255.0
    return image.numpy()

def predict_image(image_path, model_path, top_k, class_names):
    model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer}, compile=False)

    im = Image.open(image_path)
    test_image = np.asarray(im)
    processed_image = process_image(test_image)

    probs_predict = model.predict(np.expand_dims(processed_image, axis=0))
    probs = probs_predict[0].tolist()    
    values, indices = tf.math.top_k(probs, k=top_k)    
    top_k_probs = values.numpy().tolist()   
    top_k_class_indices = indices.numpy().tolist()    
    top_k_classes = [str(i) for i in top_k_class_indices]    
    top_k_class_labels = [class_names[label] for label in top_k_classes]
    return top_k_probs, top_k_classes, top_k_class_labels