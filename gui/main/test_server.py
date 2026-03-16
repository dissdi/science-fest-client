from flask import Flask, request, jsonify
import time

app = Flask(__name__)


# 1. 동의 (Consent)
@app.route('/consent', methods=['POST'])
def consent():
    # 무조건 성공했다고 가짜 세션 ID를 줍니다.
    return jsonify({"session_id": "mock_session_12345"})


# 2. 파일 업로드 (Upload)
@app.route('/files', methods=['POST'])
def upload_file():
    # 파일이 실제로 오든 말든, 잘 받았다고 거짓말을 합니다.
    file = request.files['file']
    return jsonify({
        "file_id": "mock_file_999",
        "filename": file.filename,
        "mime": file.content_type,
        "size": 12345
    })


# 3. 메시지 전송 (Send Message)
@app.route('/chat/send', methods=['POST'])
def chat_send():
    data = request.json
    user_msg = data.get("message", "")

    # 1초 정도 생각하는 척 시간을 끕니다 (로딩바 테스트용)
    time.sleep(1)

    # 앵무새처럼 대답하거나 정해진 말을 합니다.
    response_text = f"가짜 서버입니다. 당신은 이렇게 말했군요: '{user_msg}'"

    return jsonify({
        "assistant": response_text,
        "events": [{"type": "mock_event"}]
    })


# 4. 대화 초기화 (Reset)
@app.route('/chat/reset', methods=['POST'])
def chat_reset():
    return jsonify({"ok": True})


if __name__ == '__main__':
    # 5177 포트에서 서버를 켭니다.
    print("🤖 가짜 서버가 5177 포트에서 실행 중입니다...")
    app.run(port=5177, debug=True)