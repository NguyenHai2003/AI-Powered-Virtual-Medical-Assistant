import os
# Ẩn cảnh báo của TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

lemmatizer = WordNetLemmatizer()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Đọc intents.json
intents_path = os.path.join(BASE_DIR, 'intents.json')
with open(intents_path, encoding='utf-8') as file:
    intents = json.load(file)

words = []  # Từ vựng duy nhất
classes = []  # Danh sách nhãn (tags)
documents = []  # Câu + nhãn
ignore_letters = ['?', '!', '.', ',']  # Loại bỏ ký tự không cần thiết

# Xử lý dữ liệu: tách từ, chuẩn hóa, lưu câu và nhãn
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize & loại bỏ trùng lặp
words = sorted(set(lemmatizer.lemmatize(word) for word in words if word not in ignore_letters))
classes = sorted(set(classes))

model_dir = os.path.join(BASE_DIR, 'model')
os.makedirs(model_dir, exist_ok=True)

# Lưu các file pickle
pickle.dump(words, open(os.path.join(model_dir, 'words.pkl'), 'wb'))
pickle.dump(classes, open(os.path.join(model_dir, 'classes.pkl'), 'wb'))

# Tạo dữ liệu huấn luyện (bag of words + one-hot encoding)
training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = [1 if word in [lemmatizer.lemmatize(w.lower()) for w in document[0]] else 0 for word in words]
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1  # One-hot encoding của nhãn
    training.append([bag, output_row])

# Trộn dữ liệu và chuyển thành NumPy array
random.shuffle(training)
training = np.array(training, dtype=object)

# Tách đặc trưng (train_x) và nhãn (train_y)
train_x = list(training[:, 0])
train_y = list(training[:, 1])

# Xây dựng mô hình mạng nơ-ron
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))  # Lớp ẩn 1
model.add(Dropout(0.5))  # Tránh overfitting
model.add(Dense(64, activation='relu'))  # Lớp ẩn 2
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))  # Lớp đầu ra (phân loại)

# Cấu hình thuật toán tối ưu
sgd = SGD(learning_rate=0.01, weight_decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Huấn luyện mô hình
model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

# Lưu mô hình đã huấn luyện
model.save(os.path.join(model_dir, 'chatbot_model.keras'))
