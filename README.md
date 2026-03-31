client 폴더에 .env.secret 파일을 만든다.
[.env.secret]
OPENAI_API_KEY=
SERVER_HOST=
SERVER_PORT=

실행방법
1. conda activate your_env
2. pip install -r requirements.txt
3. run_client

만약 dashboard에 빈 화면, dummy화면 같은게 올라갈 경우,
.env의 CAM_INDEX를 0 또는 1, 2 ...로 바꿔볼것