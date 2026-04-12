---
name: code-review
description: 코드 품질 검사 및 개선 제안
---

## 코드 리뷰 규칙

### 파일 크기
- py 200줄 초과: DANGER → 분리 필수
- py 150줄 초과: WARNING → 분리 검토

### 실행
```bash
python .claude/hooks/scan_all.py
```
DANGER/WARNING 발생 시 해당 파일의 분리 방안을 제안하고 개선할 것.
