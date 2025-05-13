import re
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import database_helper

lemmatizer = WordNetLemmatizer()

# Đọc file intents.json và entities.json
with open('intents.json', encoding='utf-8') as file:
    intents = json.load(file)
with open('entities.json', 'r', encoding='utf-8') as f:
    entities = json.load(f)

words = pickle.load(open('model/words.pkl', 'rb'))
classes = pickle.load(open('model/classes.pkl', 'rb'))
model = load_model('model/chatbot_model.keras')


# Hàm làm sạch câu đầu vào
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


# Hàm tạo bag of words từ câu đầu vào
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


# Hàm dự đoán intent của câu
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]
    return return_list


# Hàm chuyển đổi chữ số thành số
def convert_text_to_number(text):
    numbers = {
        "một": 1, "hai": 2, "ba": 3, "bốn": 4, "năm": 5,
        "sáu": 6, "bảy": 7, "tám": 8, "chín": 9, "mười": 10
    }
    return numbers.get(text, text)  # Trả về số nếu có, hoặc giữ nguyên nếu không phải chữ số


# Hàm nhận diện thực thể và số lượng
def recognize_entities(sentence):
    entity_dict = {}
    words_in_sentence = nltk.word_tokenize(sentence.lower())

    # Nhận diện thực thể từ file entities.json
    for entity, keywords in entities.items():
        for keyword in keywords:
            # Sử dụng biểu thức chính quy để tìm kiếm cả cụm từ
            if re.search(r'\b' + re.escape(keyword) + r'\b', sentence.lower()):
                entity_dict[entity] = keyword

    # Nhận diện số lượng (dưới dạng chữ hoặc số)
    for word in words_in_sentence:
        if word.isdigit():  # Nếu là chữ số
            entity_dict["quantity"] = word
        elif word in entities.get("quantity", []):  # Nếu là số dưới dạng chữ
            entity_dict["quantity"] = str(convert_text_to_number(word))
    return entity_dict



# Hàm lấy phản hồi từ chatbot, tích hợp nhận diện thực thể và số
def get_response(intents_list, intents_json, entities_dict):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    result = "Xin lỗi, tôi chưa hiểu yêu cầu của bạn."  # Phản hồi mặc định

    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            # Thay thế các placeholder bằng thực thể từ `entities_dict`
            for entity, value in entities_dict.items():
                result = result.replace(f"<{entity}>", value)
            break
    return result


# Chạy chatbot
print("Chatbot is running!")
while True:
    message = input('')
    ints = predict_class(message)
    entities_found = recognize_entities(message)
    res = get_response(ints, intents, entities_found)
    print(res)
