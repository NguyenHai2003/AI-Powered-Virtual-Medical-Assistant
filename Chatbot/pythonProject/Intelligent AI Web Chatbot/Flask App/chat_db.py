import sqlite3
from datetime import datetime

DB_NAME = 'chat_history.db'

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS sessions (
                    session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT,
                    created_at TEXT
                )''')
    c.execute('''CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER,
                    sender TEXT,
                    message TEXT,
                    timestamp TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                )''')
    conn.commit()
    conn.close()
    migrate_db()

def migrate_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("PRAGMA table_info(messages)")
    columns = [col[1] for col in c.fetchall()]
    if 'session_id' not in columns:
        c.execute("ALTER TABLE messages ADD COLUMN session_id INTEGER")
        c.execute("UPDATE messages SET session_id = 1 WHERE session_id IS NULL")
    conn.commit()
    conn.close()

def start_new_session():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    created_at = datetime.utcnow().isoformat()
    c.execute("INSERT INTO sessions (title, created_at) VALUES (?, ?)", ("Cuộc trò chuyện mới", created_at))
    session_id = c.lastrowid
    conn.commit()
    conn.close()
    return {'session_id': session_id}

def save_message(session_id, sender, message):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    timestamp = datetime.utcnow().isoformat()
    c.execute("INSERT INTO messages (session_id, sender, message, timestamp) VALUES (?, ?, ?, ?)",
              (session_id, sender, message, timestamp))
    
    # Chỉ cập nhật tiêu đề nếu đây là tin nhắn đầu tiên của user trong session
    if sender == 'user':
        c.execute("SELECT COUNT(*) FROM messages WHERE session_id = ? AND sender = 'user'", (session_id,))
        user_message_count = c.fetchone()[0]
        if user_message_count == 1:  # Chỉ cập nhật nếu là tin nhắn user đầu tiên
            c.execute("UPDATE sessions SET title = ? WHERE session_id = ?",
                      (message[:50] + ('...' if len(message) > 50 else ''), session_id))
    
    conn.commit()
    conn.close()

def get_all_sessions():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT session_id, title FROM sessions ORDER BY created_at DESC")
    sessions = [{'session_id': row[0], 'title': row[1]} for row in c.fetchall()]
    conn.close()
    return sessions

def get_session_messages(session_id):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT sender, message FROM messages WHERE session_id = ? ORDER BY timestamp ASC", (session_id,))
    messages = [{'sender': row[0], 'message': row[1]} for row in c.fetchall()]
    conn.close()
    return messages

def clear_history():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("DELETE FROM messages")
    c.execute("DELETE FROM sessions")
    conn.commit()
    conn.close()

def delete_session(session_id):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
    c.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()