## `API.md`

# Local Agent API (for HTML GUI)

## Base URL

* `http://127.0.0.1:5177`
* 로컬 전용(외부 접속 불가)으로 운영

## Common

* 모든 기능은 “동의 세션”이 필요함
* 동의 성공 시 발급되는 `session_id`를 이후 요청에 포함해야 함

### Error format (권장)

* `403` (동의 안 됨)

```json
{ "error": "consent_required" }
```

* `400` (요청 형식 문제)

```json
{ "error": "bad_request", "detail": "..." }
```

* `500` (서버 내부 오류)

```json
{ "error": "internal_error" }
```

---

## 1) Consent (버튼: 동의하고 시작)

### POST `/consent`

**Request (JSON)**

```json
{ "consent": true }
```

**Response (200)**

```json
{ "session_id": "abc123" }
```

Notes:

* GUI는 `session_id`를 저장하고 이후 요청에 사용

---

## 2) Upload File (버튼: 파일 업로드)

### POST `/files`

**Request**

* `multipart/form-data`
* field: `file` (binary)

**Response (200)**

```json
{
  "file_id": "f1",
  "filename": "report.pdf",
  "mime": "application/pdf",
  "size": 123456
}
```

Notes:

* GUI는 `file_id` 목록을 “현재 첨부 목록”으로 관리

---

## 3) Send Message with Files (버튼: 파일 전송 = 글 + 파일 같이)

### POST `/chat/send`

**Request (JSON)**

```json
{
  "session_id": "abc123",
  "message": "사용자 입력 텍스트",
  "file_ids": ["f1", "f2"]
}
```

**Response (200)**

```json
{
  "assistant": "에이전트 응답 텍스트",
  "events": [
    { "type": "analysis_started" },
    { "type": "file_parsed" },
    { "type": "llm_called" }
  ]
}
```

Notes:

* `events`는 선택(로그/연출용). GUI는 안 써도 됨.
* Enter 전송(글만)은 `file_ids: []`로 동일 엔드포인트 호출

---

## 4) Reset Chat (버튼: 대화 초기화)

### POST `/chat/reset`

**Request (JSON)**

```json
{ "session_id": "abc123" }
```

**Response (200)**

```json
{ "ok": true }
```

Notes:

* GUI는 채팅 UI + 첨부 목록도 함께 초기화

---
