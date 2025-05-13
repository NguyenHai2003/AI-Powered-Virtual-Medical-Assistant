from flask import Flask, request, jsonify, render_template
from utils import get_response, predict_class
from chat_db import init_db, save_message, get_all_sessions, get_session_messages, clear_history, start_new_session, delete_session

app = Flask(__name__, template_folder='templates')

# Khởi tạo DB khi chạy lần đầu
init_db()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/handle_message', methods=['POST'])
def handle_message():
    data = request.json
    message = data['message']
    session_id = data.get('session_id')

    if not session_id:
        session_id = start_new_session()['session_id']

    intents_list = predict_class(message)
    raw_response = get_response(intents_list, message)

    # Định dạng câu trả lời thành HTML
    formatted_response = format_response(raw_response)

    save_message(session_id, 'user', message)
    save_message(session_id, 'bot', raw_response)

    return jsonify({'response': formatted_response, 'session_id': session_id})

def format_response(raw_response):
    # Ví dụ: Định dạng thủ công dựa trên nội dung (có thể cải tiến bằng regex hoặc parser)
    lines = raw_response.split('\n')
    html = '<div class="formatted-response">'
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith('**') and line.endswith('**'):
            # Tiêu đề
            html += f'<h3>{line[2:-2]}</h3>'
        elif line.startswith('*'):
            # Danh sách gạch đầu dòng
            html += f'<ul><li>{line[2:]}</li></ul>'
        elif line.startswith('```php'):
            # Code block
            code_content = []
            for next_line in lines[lines.index(line) + 1:]:
                if next_line.startswith('```'):
                    break
                code_content.append(next_line)
            html += '<pre><code>' + '\n'.join(code_content) + '</code></pre>'
        else:
            # Đoạn văn thường
            html += f'<p>{line}</p>'
    
    html += '</div>'
    return html

@app.route('/get_sessions', methods=['GET'])
def get_sessions():
    sessions = get_all_sessions()
    return jsonify(sessions)

@app.route('/get_session_messages', methods=['GET'])
def get_session_messages_route():
    session_id = request.args.get('session_id')
    messages = get_session_messages(session_id)
    # Định dạng lại tin nhắn bot khi tải từ lịch sử
    formatted_messages = []
    for msg in messages:
        if msg['sender'] == 'bot':
            formatted_messages.append({'sender': msg['sender'], 'message': format_response(msg['message'])})
        else:
            formatted_messages.append(msg)
    return jsonify(formatted_messages)

@app.route('/start_new_session', methods=['POST'])
def start_new_session_route():
    session_id = start_new_session()['session_id']
    return jsonify({'session_id': session_id})

@app.route('/clear_history', methods=['GET'])
def clear_history_route():
    clear_history()
    return jsonify({'status': 'success'})

@app.route('/delete_session', methods=['POST'])
def delete_session_route():
    data = request.json
    session_id = data.get('session_id')
    delete_session(session_id)
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)