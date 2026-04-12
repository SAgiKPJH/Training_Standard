"""
프로젝트 전체 py 파일 스캔
실행: python .claude/hooks/scan_all.py
"""
import os
import sys

# check_code의 check_file 함수 재사용
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_code import check_file


def scan_project(root_dir):
    all_issues = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        # 제외 디렉토리
        dirnames[:] = [d for d in dirnames if d not in ('__pycache__', '.git', '.claude', 'node_modules', 'Sample')]
        if '20260331' in dirpath:
            continue

        for filename in filenames:
            if filename.endswith('.py'):
                file_path = os.path.join(dirpath, filename)
                issues = check_file(file_path)
                all_issues.extend(issues)

    return all_issues


if __name__ == '__main__':
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    issues = scan_project(root)

    if not issues:
        print("All clear.")
    else:
        errors = [i for i in issues if i.startswith('ERROR')]
        dangers = [i for i in issues if i.startswith('DANGER')]
        warnings = [i for i in issues if i.startswith('WARNING')]

        for i in errors:
            print(i)
        for i in dangers:
            print(i)
        for i in warnings:
            print(i)

        print(f"\nSummary: {len(errors)} errors, {len(dangers)} dangers, {len(warnings)} warnings")
