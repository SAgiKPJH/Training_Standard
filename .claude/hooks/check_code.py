"""
코드 품질 검사 스크립트 (Claude Code Hook용)
- py 파일 200줄 초과: DANGER
- py 파일 150줄 초과: WARNING
- py 문법 오류: ERROR
"""
import sys
import os
import json
import ast


def check_file(file_path):
    """단일 파일 검사. 문제가 있으면 메시지 리스트 반환."""
    issues = []

    if not file_path.endswith('.py'):
        return issues
    if not os.path.exists(file_path):
        return issues
    if any(skip in file_path for skip in ('20260331', '__pycache__', 'Sample')):
        return issues

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
    except Exception:
        return issues

    line_count = len(lines)
    rel_path = os.path.relpath(file_path)

    # 문법 오류 검사
    try:
        ast.parse(content)
    except SyntaxError as e:
        issues.append(f"ERROR: {rel_path}:{e.lineno} - SyntaxError: {e.msg}")
        return issues

    # 줄 수 검사
    if line_count > 200:
        issues.append(f"DANGER: {rel_path} - {line_count} lines (> 200, split recommended)")
    elif line_count > 150:
        issues.append(f"WARNING: {rel_path} - {line_count} lines (> 150, consider splitting)")

    return issues


def main():
    """stdin에서 hook 데이터를 읽고, 변경된 파일을 검사."""
    try:
        hook_data = json.loads(sys.stdin.read())
    except Exception:
        return

    # tool_input에서 file_path 추출
    tool_input = hook_data.get('tool_input', {})
    file_path = tool_input.get('file_path', '')

    if not file_path:
        return

    issues = check_file(file_path)

    if issues:
        for issue in issues:
            print(issue, file=sys.stderr)
        # ERROR가 있으면 exit 1 (블로킹)
        if any(i.startswith('ERROR') for i in issues):
            sys.exit(1)


if __name__ == '__main__':
    main()
