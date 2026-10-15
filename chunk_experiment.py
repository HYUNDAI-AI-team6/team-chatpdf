"""실행: python chunk_experiment.py

아래 CHUNK_CONFIGS와 EXAMPLE_QUESTIONS를 수정해 실험합니다.
OpenAI API를 호출하므로 임베딩 및 답변 생성 비용이 발생합니다.
결과는 experiments/ 아래 실행별 CSV에 질문과 답변을 각각 별도 행으로 저장합니다.
"""

import csv
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from time import perf_counter

from dotenv import load_dotenv
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / "data/manual.pdf"
OUTPUT_DIR = BASE_DIR / "experiments"

# (청크 크기, 청크 오버랩): 문자가 기준이며 여러 설정을 추가할 수 있습니다.
CHUNK_CONFIGS = [
    (100, 20),
    # (500, 100),
    # (1000, 200),
]

EXAMPLE_QUESTIONS = [
    "예약 청소는 최대 몇 시간 뒤까지 설정할 수 있나요?",
    "전원은 어떻게 끄나요?",
    "먼지통 청소 방법을 알려주세요.",
    "음성인식 기능은 어떻게 켜고 끄나요?",
    "충전이 되지 않을 때는 어떻게 해야 하나요?",
]

EMBEDDING_MODEL = "text-embedding-3-large"
# 기존 앱의 ChatOpenAI 기본 모델을 명시하여 실험 조건을 고정합니다.
QUERY_MODEL = "gpt-3.5-turbo"
ANSWER_MODEL = "gpt-4o-mini"
RETRIEVAL_K = 4


def create_result_file():
    """기존 최대 번호 다음 파일을 생성하며, 기존 결과를 덮어쓰지 않습니다."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    numbers = [
        int(match.group(1))
        for path in OUTPUT_DIR.iterdir()
        if (match := re.fullmatch(r"chunk_tuning_(\d+)\.csv", path.name))
    ]
    number = max(numbers, default=0) + 1
    while True:
        path = OUTPUT_DIR / f"chunk_tuning_{number:02d}.csv"
        try:
            return path, path.open("x", newline="", encoding="utf-8-sig")
        except FileExistsError:
            number += 1


def make_answer_chain(db, api_key):
    retriever = MultiQueryRetriever.from_llm(
        retriever=db.as_retriever(search_kwargs={"k": RETRIEVAL_K}),
        llm=ChatOpenAI(model=QUERY_MODEL, temperature=0, api_key=api_key),
    )
    prompt = ChatPromptTemplate.from_template(
        "당신은 가전제품 사용 설명서 도우미야.\n"
        "아래 [메뉴얼]을 보고 질문에 답변해줘.\n"
        "[메뉴얼]에 없는 질문은 '메뉴얼에 없습니다'라고 답해.\n\n"
        "[메뉴얼]\n{context}\n\n[질문]\n{question}"
    )
    generator = (
        prompt
        | ChatOpenAI(model=ANSWER_MODEL, temperature=0, api_key=api_key)
        | StrOutputParser()
    )

    def answer(question):
        docs = retriever.invoke(question)
        return generator.invoke(
            {"context": "\n\n".join(doc.page_content for doc in docs),
             "question": question}
        )

    return answer


def main():
    load_dotenv(BASE_DIR / ".env")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(".env에 OPENAI_API_KEY를 설정해주세요.")
    if not PDF_PATH.is_file():
        raise FileNotFoundError(f"PDF 파일이 없습니다: {PDF_PATH}")
    if not CHUNK_CONFIGS:
        raise ValueError("CHUNK_CONFIGS에 실험 설정을 추가해주세요.")
    for size, overlap in CHUNK_CONFIGS:
        if size <= 0 or not 0 <= overlap < size:
            raise ValueError(f"청크 설정 오류: {(size, overlap)} (0 <= overlap < size)")
    if len(EXAMPLE_QUESTIONS) != 5 or any(not q.strip() for q in EXAMPLE_QUESTIONS):
        raise ValueError("비어 있지 않은 예시 질문을 정확히 5개 입력해주세요.")

    # 앱과 동일하게 PDF를 로드합니다. PDF 로딩 시간은 인덱스 생성 시간에서 제외합니다.
    pages = PyPDFLoader(str(PDF_PATH)).load_and_split()
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=api_key)
    run_id = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    output_path, output_file = create_result_file()
    # utf-8-sig는 Excel에서 한글을 열 때도 호환됩니다.
    with output_file as file:
        writer = csv.writer(file)
        writer.writerow(["항목", "내용"])
        for size, overlap in CHUNK_CONFIGS:
            chunks = RecursiveCharacterTextSplitter(
                chunk_size=size, chunk_overlap=overlap,
                length_function=len, is_separator_regex=False,
            ).split_documents(pages)
            if not chunks:
                raise ValueError("PDF에서 텍스트 청크를 생성하지 못했습니다.")
            print(f"청크 크기={size}, 오버랩={overlap}, 총 청크={len(chunks)}")
            db = None
            try:
                start = perf_counter()
                db = Chroma.from_documents(
                    chunks, embeddings,
                    collection_name=f"experiment_{uuid.uuid4().hex}",
                )
                index_seconds = perf_counter() - start
                answer = make_answer_chain(db, api_key)
                row = {
                    "run_id": run_id, "chunk_size": size,
                    "chunk_overlap": overlap, "total_chunks": len(chunks),
                    "index_creation_seconds": round(index_seconds, 6),
                    "embedding_model": EMBEDDING_MODEL,
                    "query_model": QUERY_MODEL, "answer_model": ANSWER_MODEL,
                    "retrieval_k": RETRIEVAL_K,
                }
                writer.writerows(row.items())
                file.flush()
                for number, question in enumerate(EXAMPLE_QUESTIONS, start=1):
                    response, error = "", ""
                    try:
                        response = answer(question)
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {exc}"
                        print(f"  질문 {number} 실패: {error}")
                    # 줄바꿈을 공백으로 바꿔 CSV를 텍스트로 열어도 한 행으로 보이게 합니다.
                    writer.writerow([f"질문{number}", " ".join(question.splitlines())])
                    writer.writerow([f"답변{number}", " ".join(response.splitlines())])
                    if error:
                        writer.writerow([f"오류{number}", " ".join(error.splitlines())])
                    file.flush()
                    print(f"  질문 {number}/5 처리 완료")
                writer.writerow([])
                file.flush()
                print("  설정별 결과 저장 완료")
            finally:
                if db is not None:
                    db.delete_collection()
    print(f"결과 저장 완료: {output_path}")


if __name__ == "__main__":
    main()
