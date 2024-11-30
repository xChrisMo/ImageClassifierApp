import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import argparse
import json
from utility_file import predict_image, load_mapping

def main():
    parser = argparse.ArgumentParser(description='Flower predicting App')
    parser.add_argument("image_path", help="Image Path")
    parser.add_argument("saved_model", help="Saved Model")
    parser.add_argument("--top_k", help="Fetch top k predictions", default=5, type=int)
    parser.add_argument('--category_names', dest="category_names", default='label_map.json')

    args = parser.parse_args()

    class_names = load_mapping(args.category_names)
    top_k_probs, top_k_classes, top_k_names = predict_image(args.image_path, args.saved_model, args.top_k, class_names)
    top_k_probs = [f"{prob * 100:.2f}%" for prob in top_k_probs]

    print("Top K Probabilities:", top_k_probs)
    print("Top K Classes:", top_k_classes)
    print("Top K Names:", top_k_names)

if __name__ == '__main__':
    main()