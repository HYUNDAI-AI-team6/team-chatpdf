import csv
import logging
import os
import textwrap
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")
logging.getLogger("pypdf").setLevel(logging.ERROR)

PDF_PATH = PROJECT_ROOT / "data" / "manual.pdf"
OUTPUT_DIR = PROJECT_ROOT / "experiments"

# 이 두 값을 바꾼 뒤 `python experiment.py`를 실행하세요.
CHUNK_SIZE = 100
CHUNK_OVERLAP = 20

TEST_QUESTIONS = [
    {
        "question_id": "Q1",
        "question": "예약 청소는 최대 몇 시간 뒤까지 설정할 수 있나요?",
    },
    {
        "question_id": "Q2",
        "question": "전원은 어떻게 끄나요?",
    },
    {
        "question_id": "Q3",
        "question": "먼지통 청소 방법을 알려주세요.",
    },
    {
        "question_id": "Q4",
        "question": "음성인식 기능은 어떻게 켜고 끄나요?",
    },
    {
        "question_id": "Q5",
        "question": "오늘 서울 날씨는 어떤가요?",
    },
]

PROMPT = ChatPromptTemplate.from_template(
    "당신은 가전제품 사용 설명서 도우미야.\n"
    "아래 [메뉴얼]을 보고 질문에 답변해줘.\n"
    "[메뉴얼]에 없는 질문은 '메뉴얼에 없습니다'라고 답해.\n\n"
    "[메뉴얼]\n{context}\n\n"
    "[질문]\n{question}"
)

CSV_COLUMNS = [
    "index_page_count",
    "index_chunk_count",
    "index_build_seconds",
    "question_id",
    "question",
    "answer",
    "retrieved_pages",
    "retrieved_chunks",
    "answer_seconds",
    "score",
    "notes",
]


def build_db(pdf_path: Path, api_key: str) -> tuple[Chroma, int, int]:
    pages = PyPDFLoader(str(pdf_path)).load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(pages)
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=api_key,
    )
    db = Chroma.from_documents(
        chunks,
        embeddings,
        collection_name=f"experiment_{uuid.uuid4().hex}",
    )
    return db, len(pages), len(chunks)


def display_page(metadata: dict) -> str | None:
    page = metadata.get("page")
    if isinstance(page, int):
        return str(page + 1)
    if page is not None:
        return str(page)
    return None


def csv_text(value: str) -> str:
    return " ".join(value.split())


def next_output_path(output_dir: Path) -> Path:
    existing_numbers = []
    for path in output_dir.glob("chunk_tuning_*.csv"):
        number = path.stem.removeprefix("chunk_tuning_")
        if number.isdigit():
            existing_numbers.append(int(number))
    next_number = max(existing_numbers, default=0) + 1
    return output_dir / f"chunk_tuning_{next_number:02d}.csv"


def write_results_csv(output_path: Path, results: list[dict[str, object]]) -> None:
    with output_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.writer(csv_file, lineterminator="\n")
        writer.writerow([f"청크사이즈 : {CHUNK_SIZE}"])
        writer.writerow([f"청크오버랩 : {CHUNK_OVERLAP}"])
        writer.writerow([])
        writer.writerow(CSV_COLUMNS)
        for result in results:
            answer = result.get("answer")
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError(
                    f"{result.get('question_id', '알 수 없는 질문')} 답변이 비어 있어 "
                    "CSV에 저장하지 못했습니다."
                )
            writer.writerow([result.get(column, "") for column in CSV_COLUMNS])


def main() -> None:
    if CHUNK_SIZE <= 0:
        raise ValueError("CHUNK_SIZE는 0보다 커야 합니다.")
    if CHUNK_OVERLAP < 0 or CHUNK_OVERLAP >= CHUNK_SIZE:
        raise ValueError("CHUNK_OVERLAP은 0 이상이고 CHUNK_SIZE보다 작아야 합니다.")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(f"{PROJECT_ROOT / '.env'}에 OPENAI_API_KEY를 설정하세요.")
    if not PDF_PATH.is_file():
        raise FileNotFoundError(f"매뉴얼 PDF 파일을 찾을 수 없습니다: {PDF_PATH}")

    index_started = time.perf_counter()
    db, page_count, chunk_count = build_db(PDF_PATH, api_key)
    index_build_seconds = time.perf_counter() - index_started

    retrieval_llm = ChatOpenAI(temperature=0, openai_api_key=api_key)
    retriever = MultiQueryRetriever.from_llm(
        retriever=db.as_retriever(),
        llm=retrieval_llm,
    )
    answer_llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=api_key,
    )
    answer_chain = PROMPT | answer_llm | StrOutputParser()

    results = []
    for item in TEST_QUESTIONS:
        started = time.perf_counter()
        documents = retriever.invoke(item["question"])
        context = "\n\n".join(document.page_content for document in documents)
        answer = answer_chain.invoke(
            {"context": context, "question": item["question"]}
        )
        answer = str(answer).strip()
        if not answer:
            raise RuntimeError(
                f"{item['question_id']}에 대해 모델이 빈 답변을 반환했습니다."
            )
        elapsed_seconds = time.perf_counter() - started

        pages = list(
            dict.fromkeys(
                page
                for document in documents
                if (page := display_page(document.metadata)) is not None
            )
        )
        results.append(
            {
                "index_page_count": page_count,
                "index_chunk_count": chunk_count,
                "index_build_seconds": round(index_build_seconds, 3),
                "question_id": item["question_id"],
                "question": csv_text(item["question"]),
                "answer": textwrap.fill(
                    csv_text(answer),
                    width=40,
                    break_long_words=True,
                    break_on_hyphens=False,
                ),
                "retrieved_pages": ", ".join(pages),
                "retrieved_chunks": " ||| ".join(
                    csv_text(document.page_content) for document in documents
                ),
                "answer_seconds": round(elapsed_seconds, 3),
                "score": "",
                "notes": "",
            }
        )
        print(f"{item['question_id']} 완료 ({elapsed_seconds:.2f}초)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = next_output_path(OUTPUT_DIR)
    write_results_csv(output_path, results)

    print(f"PDF: {page_count}페이지, {chunk_count}청크")
    print(f"인덱스 생성 시간: {index_build_seconds:.2f}초")
    print(f"CSV 저장 완료: {output_path}")


if __name__ == "__main__":
    main()
