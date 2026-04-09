client 폴더에 .env.secret 파일을 만든다.
[.env.secret]
OPENAI_API_KEY=
SERVER_HOST=
SERVER_PORT=

실행방법
1. conda activate your_env
2. pip install -r requirements.txt
3. run_client

시연 방법
1. get started를 누른다
2. file upload 버튼을 통해 'agent_test_injection.pdf' 를 업로드한다.
3. '내용을 요약해줘' 입력
4. 카메라가 찍히고 dashboard에 올라온 자신의 사진을 확인한다.

만약 dashboard에 빈 화면, dummy화면 같은게 올라갈 경우,
.env의 CAM_INDEX를 0 또는 1, 2 ...로 바꿔볼것  