# 🤖 Trợ lý Y tế Ảo Thông Minh (AI-Powered Virtual Medical Assistant)
Dự án trợ lý y tế sử dụng AI, thu thập dữ liệu sức khỏe thời gian thực và đưa ra tư vấn thông minh.

## 🩺 Chức năng chính
Thu thập dữ liệu nhịp tim và SpO₂ bằng cảm biến MAX30102

Nhúng mô hình học máy vào ESP32 (C++) để dự đoán tình trạng sức khỏe

Giao diện web tương tác với người dùng, sử dụng Gemini API để đưa ra lời khuyên

## 🔧 Công nghệ
Python, TensorFlow (Huấn luyện mô hình)

C++, ESP32 (Nhúng model .tflite)

Flask (Web server)

Gemini API (Tư vấn sức khỏe thông minh)
