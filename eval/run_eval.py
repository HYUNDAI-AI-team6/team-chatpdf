"""매뉴얼 챗봇 평가 스크립트 (사람 채점용).

eval_set.json의 질문을 app.py의 answer()로 답하게 해서 grading_sheet.csv에 적는다.
채점은 사람이 시트의 '개선 전' / '개선 후' 열에 O / △ / X 로 입력한다.

순서 (어느 폴더에서 실행해도 됨):
    1. python eval/run_eval.py before    # 개선 전 답변 생성
    2. grading_sheet.csv의 '개선 전' 열 채점
    3. (코드 개선)
    4. python eval/run_eval.py after     # 개선 후 답변 생성
    5. grading_sheet.csv의 '개선 후' 열 채점
    6. python eval/run_eval.py score     # 점수 집계

일부 문항만:  python eval/run_eval.py before 3 9 15
이미 채점한 문항을 다시 생성:  --force (해당 문항의 채점이 지워짐)
"""
import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from langchain_core.tracers.context import collect_runs

EVAL_DIR = Path(__file__).resolve().parent
ROOT = EVAL_DIR.parent
EVAL_SET_PATH = EVAL_DIR / "eval_set.json"
SHEET_PATH = EVAL_DIR / "grading_sheet.csv"
RESULTS_DIR = EVAL_DIR / "results"

PHASES = {"before": "개선 전", "after": "개선 후"}
TYPES = ["단순 사실", "비교·종합", "표현 변형", "문서에 없는 내용"]
MARKS = {  # 채점 기준: 정확 1점 / 부분 정답 0.5점 / 오답·환각 0점
    "o": 1.0, "○": 1.0, "✅": 1.0, "1": 1.0,
    "△": 0.5, "0.5": 0.5,
    "x": 0.0, "✕": 0.0, "❌": 0.0, "0": 0.0,
}

BASE_COLUMNS = ["No", "질문", "정답 (요약)", "근거 페이지", "질문 유형", "채점 메모"]
COLUMNS = BASE_COLUMNS + [
    col for label in PHASES.values() for col in (f"{label} 답변", f"{label} 검색 페이지", label)
]


def fmt_pages(pages):
    return ", ".join(f"p.{p}" for p in pages)


def load_items():
    return json.loads(EVAL_SET_PATH.read_text(encoding="utf-8"))["items"]


def read_sheet():
    if not SHEET_PATH.exists():
        return {}
    for encoding in ("utf-8-sig", "cp949"):  # 엑셀에서 'CSV'로 저장하면 cp949가 됨
        try:
            with SHEET_PATH.open(encoding=encoding, newline="") as f:
                return {row["No"]: row for row in csv.DictReader(f)}
        except UnicodeDecodeError:
            continue
    sys.exit(f"{SHEET_PATH.name}의 인코딩을 읽지 못했습니다. 엑셀에서 'CSV UTF-8'로 다시 저장하세요.")


def write_sheet(rows):
    with SHEET_PATH.open("w", encoding="utf-8-sig", newline="") as f:  # BOM: 엑셀에서 한글 안 깨지게
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def retrieved_docs(runs):
    """답변 생성 중 최상위 retriever(MultiQueryRetriever 등)가 반환한 문서 목록"""
    docs = []
    stack = list(runs)
    while stack:
        run = stack.pop()
        if run.run_type == "retriever":
            docs.extend((run.outputs or {}).get("documents", []))
            continue  # 하위 retriever 결과는 상위 결과에 이미 합쳐져 있음
        stack.extend(run.child_runs)
    return docs


def run_question(question, answer_fn):
    start = time.perf_counter()
    try:
        with collect_runs() as cb:
            text = ""
            for text in answer_fn(question):  # 스트리밍을 끝까지 소비
                pass
        docs = retrieved_docs(cb.traced_runs)
        error = None
    except Exception as e:  # API 오류 등: 기록하고 다음 문항으로
        text, docs, error = "", [], repr(e)
    return {
        "answer": f"(에러: {error})" if error else text,
        "retrieved_pages": sorted({doc.metadata.get("page", -1) + 1 for doc in docs}),  # metadata는 0부터
        "contexts": [doc.page_content for doc in docs],
        "error": error,
        "seconds": round(time.perf_counter() - start, 1),
    }


def generate(phase, nos, force):
    label = PHASES[phase]
    items = load_items()
    targets = [it for it in items if not nos or it["no"] in nos]
    sheet = read_sheet()

    graded = [it["no"] for it in targets if sheet.get(str(it["no"]), {}).get(label, "").strip()]
    if graded and not force:
        sys.exit(f"'{label}' 채점이 이미 있는 문항: {graded}\n"
                 "답변을 다시 만들면 채점이 지워집니다. 그래도 하려면 --force를 붙이세요.")
    if SHEET_PATH.exists():
        try:
            SHEET_PATH.open("a").close()  # 엑셀에서 열려 있으면 여기서 PermissionError
        except PermissionError:
            sys.exit(f"{SHEET_PATH.name}이 다른 프로그램에서 열려 있습니다. 닫고 다시 실행하세요.")

    os.chdir(ROOT)  # app.py가 data/manual.pdf를 상대 경로로 읽음
    sys.path.insert(0, str(ROOT))
    import app  # import 시점에 매뉴얼 임베딩·DB 생성 (OpenAI 임베딩 API 호출)

    if app.DB is None:
        app.DB,_ = app.build_db(["./data/manual.pdf"],app.OPENAI_API_KEY)
        print("Wait 10 second")
        time.sleep(10)
        if app.DB is None:
            sys.exit(f"매뉴얼 DBza 생성 실패: {app.LOAD_STATUS}")

    def answer_fn(question):
        return app.answer(question, app.OPENAI_API_KEY, app.DB)

    results = {}
    for item in targets:
        r = run_question(item["question"], answer_fn)
        results[item["no"]] = r
        status = "에러" if r["error"] else f"검색 {fmt_pages(r['retrieved_pages'])}"
        print(f"{item['no']:>2}. [{item['type']}] {r['seconds']:>5}s  {status}")

    # 원본 기록 (검색된 청크 원문, 설정값): 채점할 때 근거 확인용
    RESULTS_DIR.mkdir(exist_ok=True)
    raw_path = RESULTS_DIR / f"{phase}_{datetime.now():%Y%m%d_%H%M%S}.json"
    config = {k: getattr(app, k, None) for k in ("CHUNK_SIZE", "CHUNK_OVERLAP")}
    raw_path.write_text(
        json.dumps({"phase": label, "config": config,
                    "results": [{"no": no, **r} for no, r in results.items()]},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    rows = []
    for item in items:
        row = {col: sheet.get(str(item["no"]), {}).get(col, "") for col in COLUMNS}
        row.update({  # 질문·정답은 항상 eval_set.json 기준으로 갱신
            "No": str(item["no"]),
            "질문": item["question"],
            "정답 (요약)": item["answer"],
            "근거 페이지": fmt_pages(item["pages"]) or "-",
            "질문 유형": item["type"],
            "채점 메모": item.get("notes", ""),
        })
        if item["no"] in results:
            r = results[item["no"]]
            row.update({f"{label} 답변": r["answer"],
                        f"{label} 검색 페이지": fmt_pages(r["retrieved_pages"]),
                        label: ""})  # 답변이 바뀌었으니 채점 초기화
        rows.append(row)
    write_sheet(rows)

    print(f"\n'{label}' 답변 {len(results)}개 기록: {SHEET_PATH}")
    print(f"원본 기록: {raw_path}")
    print(f"'{label}' 열에 O(정확) / △(부분 정답) / X(오답·환각)를 입력하세요.")


def score():
    sheet = read_sheet()
    if not sheet:
        sys.exit(f"{SHEET_PATH.name}이 없습니다. 먼저 'before'를 실행하세요.")
    items = load_items()

    table = {}  # label -> {type: [점수 합, 문항 수]}, 미채점·잘못된 표기 목록
    for label in PHASES.values():
        sums = {t: [0.0, 0] for t in TYPES}
        ungraded, invalid = [], []
        for item in items:
            sums[item["type"]][1] += 1
            mark = sheet.get(str(item["no"]), {}).get(label, "").strip().lower()
            if not mark:
                ungraded.append(item["no"])
            elif mark not in MARKS:
                invalid.append(f"{item['no']}({mark})")
            else:
                sums[item["type"]][0] += MARKS[mark]
        table[label] = (sums, ungraded, invalid)

    # 마크다운 표: 보고서(노션 등)에 그대로 붙여넣기용
    labels = list(PHASES.values())
    print("| 유형 | " + " | ".join(labels) + " |")
    print("|---|" + "---|" * len(labels))
    for t in TYPES:
        n = table[labels[0]][0][t][1]
        print(f"| {t} ({n}개) | " + " | ".join(f"{table[label][0][t][0]:.1f} / {n}" for label in labels) + " |")
    total_n = len(items)
    totals = [sum(s for s, _ in table[label][0].values()) for label in labels]
    print(f"| **전체 ({total_n}개)** | " + " | ".join(f"**{t:.1f} / {total_n} ({t / total_n:.1%})**" for t in totals) + " |")

    for label in labels:
        _, ungraded, invalid = table[label]
        if ungraded:
            print(f"\n'{label}' 미채점 {len(ungraded)}개: {ungraded}")
        if invalid:
            print(f"'{label}' 알 수 없는 표기 (O / △ / X 로 입력): {invalid}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")  # Windows 콘솔 한글 깨짐 방지
    parser = argparse.ArgumentParser(description="매뉴얼 챗봇 평가 (사람 채점)")
    parser.add_argument("command", choices=[*PHASES, "score"])
    parser.add_argument("nos", nargs="*", type=int, help="일부 문항 번호만 (생략 시 전체)")
    parser.add_argument("--force", action="store_true", help="이미 채점한 문항도 답변을 다시 생성")
    args = parser.parse_args()

    if args.command == "score":
        score()
    else:
        generate(args.command, args.nos, args.force)


if __name__ == "__main__":
    main()
