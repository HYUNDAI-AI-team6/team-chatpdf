"""
실행: python chunk_experiment.py

CHUNK_CONFIGS와 EXAMPLE_QUESTIONS를 수정하여 실험합니다.

OpenAI API를 호출하므로 임베딩 및 답변 생성 비용이 발생합니다.

결과는 experiments/ 폴더에 CSV 파일로 저장됩니다.

CSV 출력 형식:
- Chunk Size
- Chunk Overlap
- 전체 청크 수
- 인덱스 생성 시간
- 질문 번호 및 질문
- 답변 (40자마다 줄바꿈)
- 검색된 청크 수
"""

import csv
import os
import re
import uuid
import textwrap

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


# ===== 1. 기본 경로 설정 =====

BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / "data/manual.pdf"
OUTPUT_DIR = BASE_DIR / "experiments"


# ===== 2. 실험 설정 =====

# (청크 크기, 청크 오버랩)
# 여러 설정을 추가하여 한 번에 실험할 수 있습니다.
CHUNK_CONFIGS = [
    (1000, 40),
    (300, 90),
    (300, 10),
]

# 평가 질문
# 질문 개수는 자유롭게 변경할 수 있습니다.
EXAMPLE_QUESTIONS = [
    "로보킹의 예약 청소는 최대 몇 시간 후까지 설정할 수 있나요?",
    "로보킹에서 원터치연결(WPS) 기능을 실행하려면 어떤 버튼을 얼마나 오래 눌러야 하나요?",
    "로보킹이 충전대에 붙어 있을 때 선택할 수 없는 청소 모드는 무엇인가요?",
    "로보킹의 스마트 진단 기능을 시작하는 방법과 진단 결과를 다시 듣는 방법을 알려주세요.",
    "로보킹에서 예약 청소를 설정할 수 없는 경우와 예약이 취소되는 경우를 알려주세요.",
    "로보킹의 자동 청소가 끝났을 때와 배터리가 부족할 때 충전은 어떻게 진행되나요?",
    "로보킹의 먼지통과 먼지통 필터는 각각 어떻게 청소하고 관리해야 하나요?",
    "로보킹을 원격으로 사용하기 위해 필요한 네트워크 환경과 사전 설정 절차를 설명해 주세요.",
    "로보킹의 음성 인식 기능을 켜거나 끄는 방법과 음성 인식 기능을 사용하는 방법을 설명해 주세요.",
    "로보킹에 카메라를 추가로 장착하면 4K 해상도로 영상을 녹화할 수 있나요?",
]

# 모델 설정
EMBEDDING_MODEL = "text-embedding-3-large"
QUERY_MODEL = "gpt-3.5-turbo"
ANSWER_MODEL = "gpt-4o-mini"

# 검색 설정
RETRIEVAL_K = 4


# ===== 3. CSV 파일 생성 =====

def create_result_file():
    """
    기존 CSV 파일의 최대 번호 다음 번호로 생성합니다.
    기존 실험 결과는 덮어쓰지 않습니다.
    """

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    numbers = [
        int(match.group(1))
        for path in OUTPUT_DIR.iterdir()
        if (
            match := re.fullmatch(
                r"chunk_tuning_(\d+)\.csv",
                path.name
            )
        )
    ]

    number = max(numbers, default=0) + 1

    while True:
        path = OUTPUT_DIR / f"chunk_tuning_{number:02d}.csv"

        try:
            return path, path.open(
                "x",
                newline="",
                encoding="utf-8-sig"
            )

        except FileExistsError:
            number += 1


# ===== 4. 답변 줄바꿈 함수 =====

def format_answer(response, width=40):
    """
    답변을 40자마다 줄바꿈합니다.
    기존 문단 및 번호 목록의 줄바꿈은 유지합니다.
    """

    lines = []

    for line in response.splitlines():

        # 기존 빈 줄 유지
        if not line:
            lines.append("")
            continue

        # 40자 단위로 줄바꿈
        wrapped_lines = textwrap.wrap(
            line,
            width=width,
            break_long_words=True,
            break_on_hyphens=False,
            replace_whitespace=False,
            drop_whitespace=False,
        )

        lines.extend(wrapped_lines)

    return "\n".join(lines)


# ===== 5. RAG 체인 생성 =====

def make_answer_chain(db, api_key):

    # MultiQueryRetriever 생성
    retriever = MultiQueryRetriever.from_llm(
        retriever=db.as_retriever(
            search_kwargs={"k": RETRIEVAL_K}
        ),
        llm=ChatOpenAI(
            model=QUERY_MODEL,
            temperature=0,
            api_key=api_key
        ),
    )

    # 프롬프트 설정
    prompt = ChatPromptTemplate.from_template(
        "당신은 가전제품 사용 설명서 도우미야.\n"
        "아래 [메뉴얼]을 보고 질문에 답변해줘.\n"
        "[메뉴얼]에 없는 질문은 '메뉴얼에 없습니다'라고 답해.\n\n"
        "[메뉴얼]\n{context}\n\n"
        "[질문]\n{question}"
    )

    # 답변 생성 체인
    generator = (
        prompt
        | ChatOpenAI(
            model=ANSWER_MODEL,
            temperature=0,
            api_key=api_key
        )
        | StrOutputParser()
    )

    def answer(question):

        # 관련 청크 검색
        docs = retriever.invoke(question)

        # 검색된 청크를 하나의 문맥으로 결합
        context = "\n\n".join(
            doc.page_content for doc in docs
        )

        # LLM 답변 생성
        response = generator.invoke(
            {
                "context": context,
                "question": question
            }
        )

        # 답변과 검색된 청크 반환
        return response, docs

    return answer


# ===== 6. 실험 실행 =====

def main():

    load_dotenv(BASE_DIR / ".env")

    api_key = os.getenv("OPENAI_API_KEY")

    # API 키 확인
    if not api_key:
        raise ValueError(
            ".env에 OPENAI_API_KEY를 설정해주세요."
        )

    # PDF 파일 확인
    if not PDF_PATH.is_file():
        raise FileNotFoundError(
            f"PDF 파일이 없습니다: {PDF_PATH}"
        )

    # 청크 설정 확인
    if not CHUNK_CONFIGS:
        raise ValueError(
            "CHUNK_CONFIGS에 실험 설정을 추가해주세요."
        )

    for size, overlap in CHUNK_CONFIGS:
        if size <= 0 or not 0 <= overlap < size:
            raise ValueError(
                f"청크 설정 오류: {(size, overlap)}"
            )

    # 질문 확인
    if not EXAMPLE_QUESTIONS or any(
        not q.strip() for q in EXAMPLE_QUESTIONS
    ):
        raise ValueError(
            "비어 있지 않은 평가 질문을 1개 이상 입력해주세요."
        )

    # ===== PDF 로드 =====

    pages = PyPDFLoader(str(PDF_PATH)).load()

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=api_key
    )

    # ===== CSV 생성 =====

    output_path, output_file = create_result_file()

    with output_file as file:

        writer = csv.writer(file, delimiter=":")

        # ===== 청크 설정별 실험 =====

        for size, overlap in CHUNK_CONFIGS:

            print("\n==============================")
            print(f"Chunk Size    : {size}")
            print(f"Chunk Overlap : {overlap}")
            print("==============================")

            # ===== 청크 분할 =====

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=size,
                chunk_overlap=overlap,
                length_function=len,
                is_separator_regex=False,
            )

            chunks = text_splitter.split_documents(pages)

            if not chunks:
                raise ValueError(
                    "PDF에서 텍스트 청크를 생성하지 못했습니다."
                )

            print(f"전체 청크 수 : {len(chunks)}")

            # ===== CSV 공통 정보 저장 =====

            writer.writerow(["Chunk Size", size])
            writer.writerow(["Chunk Overlap", overlap])
            writer.writerow(["전체 청크 수", len(chunks)])
            writer.writerow([])

            file.flush()

            db = None

            try:

                # ===== Chroma DB 생성 =====

                start = perf_counter()

                db = Chroma.from_documents(
                    chunks,
                    embeddings,
                    collection_name=(
                        f"experiment_{uuid.uuid4().hex}"
                    ),
                )

                index_seconds = perf_counter() - start

                print(
                    f"인덱스 생성 시간 : "
                    f"{index_seconds:.2f}초"
                )

                # ===== 인덱스 생성 시간 CSV 저장 =====

                writer.writerow([
                    "인덱스 생성 시간",
                    f"{index_seconds:.6f}초"
                ])

                writer.writerow([])
                file.flush()

                # ===== RAG 체인 생성 =====

                answer = make_answer_chain(db, api_key)

                response_times = []

                # ===== 질문별 실험 =====

                for number, question in enumerate(
                    EXAMPLE_QUESTIONS,
                    start=1
                ):

                    response = ""
                    error = ""
                    docs = []
                    response_seconds = 0.0

                    # ===== 질문 및 답변 생성 =====

                    start = perf_counter()

                    try:

                        response, docs = answer(question)

                        response_seconds = (
                            perf_counter() - start
                        )

                        response_times.append(
                            response_seconds
                        )

                    except Exception as exc:

                        response_seconds = (
                            perf_counter() - start
                        )

                        error = (
                            f"{type(exc).__name__}: {exc}"
                        )

                        print(
                            f"질문 {number} 실패: {error}"
                        )

                    # ===== 검색된 청크 수 =====

                    retrieved_count = len(docs)

                    # ===== 답변 40자 줄바꿈 =====

                    formatted_response = format_answer(
                        response,
                        width=40
                    )

                    # ===== CSV 질문별 결과 저장 =====

                    writer.writerow([
                        f"질문 {number}",
                        question
                    ])

                    writer.writerow([
                        "답변",
                        formatted_response
                    ])

                    writer.writerow([
                        "검색된 청크 수",
                        retrieved_count
                    ])

                    # 오류가 발생한 경우에만 기록
                    if error:
                        writer.writerow([
                            "오류",
                            error
                        ])

                    # 질문별 구분을 위한 빈 행
                    writer.writerow([])

                    file.flush()

                    print(
                        f"질문 {number}/"
                        f"{len(EXAMPLE_QUESTIONS)} "
                        f"처리 완료 "
                        f"({response_seconds:.2f}초)"
                    )

                # ===== 설정별 결과 요약 =====

                if response_times:

                    avg_response_time = (
                        sum(response_times)
                        / len(response_times)
                    )

                    print("\n[실험 결과]")
                    print(f"전체 청크 수 : {len(chunks)}")

                    print(
                        f"평균 응답 시간 : "
                        f"{avg_response_time:.2f}초"
                    )

                print("설정별 결과 저장 완료")

                # 여러 설정을 실행할 경우 구분
                writer.writerow([])
                file.flush()

            finally:

                # Chroma 컬렉션 정리
                if db is not None:
                    db.delete_collection()

    print("\n==============================")
    print("전체 실험 완료")
    print(f"결과 저장 위치 : {output_path}")
    print("==============================")


if __name__ == "__main__":
    main()
