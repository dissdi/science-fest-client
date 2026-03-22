import streamlit as st
import requests
import json

# ==========================================
# 설정 및 초기화
# ==========================================
BASE_URL = "http://127.0.0.1:5177"

st.set_page_config(page_title="Local Agent GUI", layout="wide")
st.title("🤖 Chat Client")

# 세션 상태 초기화 (변수 유지용)
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_file_ids" not in st.session_state:
    st.session_state.uploaded_file_ids = []
if "uploaded_file_names" not in st.session_state:
    st.session_state.uploaded_file_names = []

# ==========================================
# 사이드바: 설정 및 파일 관리
# ==========================================
with st.sidebar:
    st.header("⚙️ 제어 패널")

    # Consent (동의 및 세션 발급)
    if st.session_state.session_id is None:
        st.warning("먼저 동의하고 세션을 시작하세요.")
        if st.button("✅ 시작 (Consent)"):
            try:
                res = requests.post(f"{BASE_URL}/consent", json={"consent": True})
                if res.status_code == 200:
                    data = res.json()
                    st.session_state.session_id = data.get("session_id")
                    st.success(f"세션 발급 완료: {st.session_state.session_id}")
                    st.rerun()
                else:
                    st.error(f"동의 실패: {res.text}")
            except Exception as e:
                st.error(f"서버 연결 오류: {e}")
    else:
        st.success(f"연결됨 (Session: {st.session_state.session_id})")

        # Upload File (파일 업로드)
        st.divider()
        st.subheader("📁 파일 업로드")
        uploaded_file = st.file_uploader("파일 선택", key="file_uploader")

        if uploaded_file is not None:
            # 중복 업로드 방지를 위해 파일명으로 체크
            if uploaded_file.name not in st.session_state.uploaded_file_names:
                with st.spinner("파일 업로드 중..."):
                    try:
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                        data = {"session_id": st.session_state.session_id}

                        res = requests.post(f"{BASE_URL}/files", files=files, data=data)

                        if res.status_code == 200:
                            data = res.json()
                            file_id = data["file_id"]
                            st.session_state.uploaded_file_ids.append(file_id)
                            st.session_state.uploaded_file_names.append(uploaded_file.name)
                            st.success(f"업로드 완료: {uploaded_file.name}")
                        else:
                            st.error(f"업로드 실패: {res.text}")
                    except Exception as e:
                        st.error(f"업로드 오류: {e}")

        # 현재 첨부된 파일 목록 표시
        if st.session_state.uploaded_file_ids:
            st.caption(f"전송 대기 파일 ({len(st.session_state.uploaded_file_ids)}개):")
            st.code(st.session_state.uploaded_file_names)

        # Reset Chat (대화 초기화)
        st.divider()
        if st.button("🗑️ 대화 초기화 (Reset)"):
            try:
                payload = {"session_id": st.session_state.session_id}
                resp = requests.post(f"{BASE_URL}/chat/reset", json=payload)
                resp.raise_for_status()
                result = resp.json()

                st.session_state.session_id = result["session_id"]
                st.session_state.messages = []
                st.session_state.uploaded_file_ids = []
                st.session_state.uploaded_file_names = []
                st.rerun()
            except Exception as e:
                st.error(f"초기화 오류: {e}")

# ==========================================
# 메인: 채팅 인터페이스
# ==========================================

# 동의가 안 되어 있으면 채팅 입력 막음
if st.session_state.session_id is None:
    st.info("👈 서버 연결 확인.")
else:
    # 기존 대화 기록 출력
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 3. Send Message (메시지 전송)
    if prompt := st.chat_input("메시지를 입력하세요..."):
        # 사용자 메시지 UI에 추가
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # API 요청 준비
        payload = {
            "session_id": st.session_state.session_id,
            "message": prompt,
            "file_ids": st.session_state.uploaded_file_ids
        }

        # 서버로 전송
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("⏳ 분석 중...")

            try:
                res = requests.post(f"{BASE_URL}/chat/send", json=payload)

                if res.status_code == 200:
                    data = res.json()
                    assistant_response = data.get("assistant", "")

                    message_placeholder.markdown(assistant_response)
                    st.session_state.messages.append({"role": "assistant", "content": assistant_response})

                elif res.status_code == 403:
                    message_placeholder.error("세션이 만료되었습니다. 다시 연결해주세요.")
                    st.session_state.session_id = None
                    st.rerun()
                else:
                    message_placeholder.error(f"오류 발생: {res.text}")

            except Exception as e:
                message_placeholder.error(f"통신 오류: {e}")