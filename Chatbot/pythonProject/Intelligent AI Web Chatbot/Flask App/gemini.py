import google.generativeai as genai
import os

# Lấy API key từ biến môi trường (khuyến nghị)
api_key = "AIzaSyD4_N_1KH04ziah9_TptoR_-N9-55GnLMA"

if not api_key:
    print("Vui lòng thiết lập biến môi trường GOOGLE_API_KEY.")
    exit()

# Cấu hình API key
genai.configure(api_key=api_key)

# Chọn mô hình
model = genai.GenerativeModel('gemini-2.0-flash')  # Hoặc 'gemini-pro' tùy bạn

def get_gemini_response(prompt):
    """Gửi prompt đến mô hình Gemini và trả về phản hồi."""
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Đã xảy ra lỗi: {e}"

def main():
    print("Chào mừng bạn đến với chatbot Gemini đơn giản!")
    while True:
        user_input = input("Bạn: ")
        if user_input.lower() in ["thoát", "bye", "tạm biệt"]:
            print("Chatbot: Tạm biệt!")
            break
        if user_input:
            gemini_response = get_gemini_response(user_input)
            print(f"Chatbot: {gemini_response}")

if __name__ == "__main__":
    main()