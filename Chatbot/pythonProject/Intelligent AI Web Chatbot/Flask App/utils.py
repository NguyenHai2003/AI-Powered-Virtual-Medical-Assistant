import os
import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model
from gemini import get_gemini_response

# Ẩn cảnh báo từ TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')

# Tải dữ liệu cần thiết
lemmatizer = WordNetLemmatizer()
words = pickle.load(open(os.path.join(MODEL_DIR, 'words.pkl'), 'rb'))
classes = pickle.load(open(os.path.join(MODEL_DIR, 'classes.pkl'), 'rb'))
model = load_model(os.path.join(MODEL_DIR, 'chatbot_model.keras'))

# ------------------ Hàm xử lý ngôn ngữ ------------------ #
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# ------------------ Dự đoán intent ------------------ #
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.95

    results = [{"intent": classes[i], "probability": str(prob)} for i, prob in enumerate(res) if prob > ERROR_THRESHOLD]
    results.sort(key=lambda x: x["probability"], reverse=True)

    print("Predicted intents:", results)
    return results

# ------------------ Xử lý sức khỏe ------------------ #
def get_health_check_response():
    bpm, spo2, prediction = 95, 95, "Bình thường"
    message = f"if my heart rate is {bpm}, and oxygen level is {spo2}, is my health ok?"
    return get_gemini_response(message)

# ------------------ Trả lời người dùng ------------------ #
def get_response(intents_list, message):
    if not intents_list:
        return get_gemini_response(message)

    with open(os.path.join(BASE_DIR, 'intents.json'), encoding='utf-8') as f:
        intents_json = json.load(f)

    tag = intents_list[0]['intent']
    if tag == "health_check":
        return get_health_check_response()

    for intent in intents_json['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

    # Fallback nếu không khớp intent nào
    return get_gemini_response(message)
