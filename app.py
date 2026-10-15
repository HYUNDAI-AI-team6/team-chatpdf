import logging
import os
import uuid

import gradio as gr
from dotenv import load_dotenv
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import glob

load_dotenv()
logging.getLogger("pypdf").setLevel(logging.ERROR)  # PDF 경고 메시지 숨기기
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


#data 폴더의 pdf 형식의 파일들을 읽어옵니다.
PDF_DIR = "data"
PDF_PATHS = sorted(glob.glob(os.path.join(PDF_DIR, "*.pdf")))
#PDF_PATH = "data/manual.pdf"
DOC_NAME = ", ".join(os.path.splitext(os.path.basename(p))[0] for p in PDF_PATHS)

#드랍다운 선택지
DOCS = {os.path.basename(p): p for p in PDF_PATHS}   # {"manual.pdf": "data/manual.pdf", ...}
#드랍다운 없이 쓸땐 그냥 요대로
ALL_DOCS = "전체 (모든 PDF에서 검색)"

# [수정] 청크 튜닝 실험 결과에 맞게 변경
CHUNK_SIZE = 100
CHUNK_OVERLAP = 20


# ===== 1. PDF 처리 (Load → Split → Embed → Store) =====
def build_db(pdf_paths, openai_key):
    if not openai_key:
        return None, "⚠️ OpenAI API 키를 먼저 입력한 뒤 PDF를 다시 올려주세요."
    if pdf_paths is None:
        return None, ""

    #읽어온 pdf 파일들을 등록합니다.
    pages = []
    for path in pdf_paths:
        pages += PyPDFLoader(path).load_and_split()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )
    texts = text_splitter.split_documents(pages)

    embeddings_model = OpenAIEmbeddings(
        model="text-embedding-3-large", openai_api_key=openai_key
    )

    db = Chroma.from_documents(
        texts, embeddings_model, collection_name=f"pdf_{uuid.uuid4().hex}"
    )
    return db, f"✅ 처리 완료: {len(pages)}페이지"
    #return db, f"✅ 처리 완료: {len(pages)}페이지 → {len(texts)}개 청크"

# ===== 2. 질문 처리 (Retrieve → Generate) =====
def answer(question, openai_key, db):
    if db is None:
        raise gr.Error("먼저 API 키를 입력하고 PDF를 업로드하세요.")
    if not question:
        raise gr.Error("질문을 입력하세요.")

    llm = ChatOpenAI(temperature=0, openai_api_key=openai_key)
    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=db.as_retriever(), llm=llm
    )
    prompt = ChatPromptTemplate.from_template(
         "당신은 가전제품 사용 설명서 도우미야.\n"
         "아래 [메뉴얼]을 보고 질문에 답변해줘.\n"
         "[메뉴얼]에 없는 질문은 '메뉴얼에 없습니다'라고 답해.\n\n"
         "[메뉴얼]\n{context}\n\n"
         "[질문]\n{question}"     
    )

    generate_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=openai_key)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever_from_llm | format_docs, "question": RunnablePassthrough()}
        | prompt
        | generate_llm
        | StrOutputParser()
    )

    text = ""
    for chunk in rag_chain.stream(question):
        text += chunk
        yield text

# ===== 3. 앱 시작 시 매뉴얼 불러오기 =====
def load_manual(paths=PDF_PATHS):
    if not OPENAI_API_KEY:
        return None, "⚠️ `.env`에 `OPENAI_API_KEY`가 없습니다."
    if not paths:
        return None, f"⚠️ `{PDF_DIR}` 폴더에 PDF 파일이 없습니다."
    try:
        return build_db(paths, OPENAI_API_KEY)
    except Exception as e:
        print("매뉴얼 로드 실패:", e)
        return None, "⚠️ 매뉴얼에서 글자를 읽지 못했습니다. PDF가 손상됐거나 스캔본인지 확인하세요."


#DB, LOAD_STATUS = load_manual()
DB, LOAD_STATUS = None, "PDF를 업로드해 주세요."

#새로운 마크다운 불러올 때
def select_doc(doc_name):
    global DB, LOAD_STATUS, DOC_NAME
    paths = PDF_PATHS if doc_name == ALL_DOCS else [DOCS[doc_name]]
    DB, LOAD_STATUS = load_manual(paths)
    DOC_NAME = doc_name
    return f"# 💬 가전제품 매뉴얼 상담봇\n📘 {DOC_NAME} · {LOAD_STATUS}", [], []

#파일올리는 부분.
def on_upload(files):
    global DB, LOAD_STATUS, DOC_NAME
    paths = [f if isinstance(f, str) else f.name for f in (files or [])]
    if not paths:
        DB, LOAD_STATUS, DOC_NAME = None, "PDF를 업로드해 주세요.", ""
    else:
        try:
            DB, LOAD_STATUS = build_db(paths, OPENAI_API_KEY)
        except Exception as e:
            print("매뉴얼 로드 실패:", e)
            DB, LOAD_STATUS = None, "⚠️ 매뉴얼에서 글자를 읽지 못했습니다. PDF가 손상됐거나 스캔본인지 확인하세요."
        DOC_NAME = ", ".join(os.path.splitext(os.path.basename(p))[0] for p in paths)
    return f"# 💬 가전제품 매뉴얼 상담봇\n📘 {DOC_NAME} · {LOAD_STATUS}", [], []


def ask(question, history):
    if DB is None:
        raise gr.Error("매뉴얼을 불러오지 못했습니다. 화면 위쪽 안내를 확인하세요.")
    if not question or not question.strip():
        raise gr.Error("질문을 입력하세요.")

    history = history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": "..."},
    ]
    yield history, history, ""

    # [수정] answer()의 매개변수가 바뀌면 여기 호출도 같은 순서로 변경
    result = answer(question, OPENAI_API_KEY, DB)
    if isinstance(result, str):  # answer()가 문자열을 반환하는 경우
        result = [result]
    for text in result:          # 문자열 또는 yield 스트리밍 모두 처리
        history[-1] = {"role": "assistant", "content": text}
        yield history, history, ""


def clear_chat():
    return [], [], ""

# ===== 4. 화면 구성 =====
THEME = gr.themes.Soft(primary_hue="teal", secondary_hue="emerald", neutral_hue="slate").set(
    button_primary_background_fill="#3CC8A8",
    button_primary_background_fill_hover="#2FB394",
    button_primary_text_color="white",
    button_primary_border_color="#3CC8A8",
)

CSS = """
.gradio-container { max-width: 820px !important; margin: 0 auto !important; }
#header { text-align: center; }
#chat { border: 1px solid #CDEFE6 !important; border-radius: 14px !important; }
#chat .message.user { background: #E6F8F3 !important; border-color: #9EE3D2 !important; }
#question textarea, #question input { border: 1px solid #9EE3D2 !important; border-radius: 10px !important; background: white !important; }
"""

EXAMPLE_QUESTIONS = [
    "예약 청소는 최대 몇 시간 뒤까지 설정할 수 있나요?",
    "전원은 어떻게 끄나요?",
    "먼지통 청소 방법을 알려주세요.",
    "음성인식 기능은 어떻게 켜고 끄나요?",
]

with gr.Blocks(title="가전제품 매뉴얼 도우미") as demo:
    header = gr.Markdown(
        f"# 💬 가전제품 매뉴얼 상담봇\n📘 {DOC_NAME} · {LOAD_STATUS}",
        elem_id="header",
    )
    history_state = gr.State([])

    # pdf 여러개 등록하기.
    files = gr.File(file_count="multiple", file_types=[".pdf"], label="PDF 업로드")

    chatbot = gr.Chatbot(
        elem_id="chat",
        height=460,
        show_label=False,
        placeholder="**무엇이든 물어보세요!**<br>사용설명서를 대신 읽고 답해드려요.",
    )

    #파일 정보 챗봇에 입력.
    files.upload(on_upload, inputs=files, outputs=[header, chatbot, history_state])
    with gr.Row():
        question = gr.Textbox(
            elem_id="question",
            placeholder="궁금한 점을 입력하세요 (Enter로 전송)",
            show_label=False,
            container=False,
            scale=5,
        )
        send_btn = gr.Button("보내기", variant="primary", scale=1, min_width=90)
    gr.Examples(examples=EXAMPLE_QUESTIONS, inputs=question, label="자주 묻는 질문")
    clear_btn = gr.Button("🗑️ 대화 지우기", size="sm", variant="secondary")

    # ===== 5. 이벤트 연결 =====
    send_btn.click(ask, inputs=[question, history_state], outputs=[chatbot, history_state, question])
    question.submit(ask, inputs=[question, history_state], outputs=[chatbot, history_state, question])
    clear_btn.click(clear_chat, outputs=[chatbot, history_state, question])
    chatbot.clear(clear_chat, outputs=[chatbot, history_state, question])  # 채팅창 휴지통 아이콘

if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS)