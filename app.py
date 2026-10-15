import base64
import json
import logging
import os
import re
import tempfile
import uuid
from datetime import datetime

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

load_dotenv()
logging.getLogger("pypdf").setLevel(logging.ERROR)  # PDF 경고 메시지 숨기기
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

PDF_PATH = "data/manual.pdf"
AVATAR_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "avatar.png")
ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
WALKERS = [  # 걷는 캐릭터 순서대로 (앞장서는 캐릭터 먼저). 각 캐릭터는 있는 파일 중 첫 번째를 사용
    ["walker.webp", "walker.png"],
    ["walker2.webp"],
]
SCENE_JS_PATH = os.path.join(ASSETS_DIR, "scene.js")
DOC_NAME = "LG 로보킹 청소로봇 (VR6180VMNC) 사용설명서"

# [수정] 청크 튜닝 실험 결과에 맞게 변경
CHUNK_SIZE = 100
CHUNK_OVERLAP = 20


# ===== 1. PDF 처리 (Load → Split → Embed → Store) =====
def build_db(pdf_path, openai_key):
    if not openai_key:
        return None, "⚠️ OpenAI API 키를 먼저 입력한 뒤 PDF를 다시 올려주세요."
    if pdf_path is None:
        return None, ""
    if not os.path.exists(pdf_path):
        print(f"⚠️ `{PDF_PATH}` 파일이 없습니다. 매뉴얼 PDF를 이 경로에 넣어주세요.")
        return None, ""

    try:
        pages = PyPDFLoader(pdf_path).load_and_split()
    except Exception as e:
        print("매뉴얼 로드 실패:", e)
        return None, "⚠️ 매뉴얼에서 글자를 읽지 못했습니다. PDF 경로를 확인하시고, PDF가 손상됐거나 스캔본인지 확인하세요."

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
    return db, f"✅ 처리 완료: {len(pages)}페이지 → {len(texts)}개 청크"

# ===== 2. 질문 처리 (Retrieve → Generate) =====
def answer(question, openai_key, db):
    if db is None:
        raise gr.Error("먼저 API 키를 입력하고 PDF를 업로드하세요.")
    if not question:
        raise gr.Error("질문을 입력하세요.")

    mmr_retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 3})

    llm = ChatOpenAI(temperature=0, openai_api_key=openai_key)
    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=mmr_retriever, llm=llm
    )
    
    NO_ANSWER_FLAG = "NO_ANSWER"
    
    NOT_FOUND_MSG = "해당 내용은 매뉴얼에서 확인할 수 없습니다."
    
    prompt = ChatPromptTemplate.from_template(
         """
당신은 가전제품 사용 설명서를 전문적으로 안내하는 AI 도우미입니다.

[핵심 원칙]
- 반드시 제공된 매뉴얼을 근거로 답변합니다.
- 매뉴얼에 없는 정보는 추측하거나 만들어내지 않습니다.
- 사용자의 질문과 관련된 내용만 답변합니다.
- 답변은 친절하고 이해하기 쉬운 한국어 존댓말로 작성합니다.

[답변 판단 기준]
1. 매뉴얼에서 명확한 답을 찾은 경우:
   - 해당 내용을 정확하게 설명합니다.

2. 매뉴얼에 일부 정보만 존재하는 경우:
   - 확인 가능한 부분만 답변합니다.
   - 확인되지 않은 내용은 매뉴얼에서 확인할 수 없다고 안내합니다.

3. 매뉴얼에서 관련 정보를 찾을 수 없는 경우:
   - 다른 말은 하지 말고 정확히 NO_ANSWER 만 출력합니다

[답변 형식]
- 단순한 질문은 짧고 명확하게 답변합니다.
- 사용 방법이나 조작 순서에 관한 질문은 번호를 붙여 설명합니다.
- 주의사항이 필요한 경우 매뉴얼에 있는 안전 정보를 함께 안내합니다.
- 불필요한 설명이나 반복적인 문장은 제외합니다.

[금지사항]
- 매뉴얼에 없는 기능을 있다고 설명하지 마세요.
- 확인되지 않은 제품 사양을 임의로 작성하지 마세요.
- 매뉴얼 내용을 넘어서는 해결 방법을 추측하지 마세요.

[매뉴얼]
{context}

[사용자 질문]
{question}

[답변]
"""   
    )

    generate_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=openai_key)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    docs = retriever_from_llm.invoke(question)      # "context" 갈래의 앞부분

    context = format_docs(docs)                     # "context" 갈래의 뒷부분

    rag_chain = prompt | generate_llm | StrOutputParser() #딕셔너리를 뺀 나머지 체인


    text = ""
    found = None #None 판단 전, True 답변 있음 ,False 메뉴얼에 없음
    for chunk in rag_chain.stream({"context": context, "question": question}):
        text += chunk
        if found is None:
            head = text.lstrip()
            if head.startswith(NO_ANSWER_FLAG):
                found = False       # 메뉴얼에 없다고 판단했기 때문에 청크를 더 받을 필요 없음
                break           
            if NO_ANSWER_FLAG.startswith(head):
                continue            # 신호의 앞부분이면 화면에 보내지 않고 대기
            found = True            # 답변이 있으니 정상 답변
        yield text

    if found is not True:           # found에 답변이 없으면 "해당 내용은 매뉴얼에서 확인할 수 없습니다."라고 출력
        yield NOT_FOUND_MSG
        return
    
    pages = set()
    for doc in docs:
        page = doc.metadata.get("page")
        try:
            pages.add(int(page) + 1)    # "3"같은 문자열도 정수로 변환
        except (TypeError, ValueError):
            continue                    # None(TypeError), "Sec2:30"(ValueError)처럼 숫자로 못 바꾸면 건너뛰기
        
    if pages:
        page_list = [f"p.{n}" for n in sorted(pages)]
        text += "\n\n📄 출처: " + ", ".join(page_list)
        yield text


    # rag_chain = (
    #     {"context": retriever_from_llm | format_docs, "question": RunnablePassthrough()}
    #     | prompt
    #     | generate_llm
    #     | StrOutputParser()
    # )

    # text = ""
    # for chunk in rag_chain.stream(question):
    #     text += chunk
    #     yield text

        # print(text, end="", flush=True)

# PDF_PATH = "./data/manual.pdf"
# if not os.path.exists(PDF_PATH):
#     print(f"⚠️ `{PDF_PATH}` 파일이 없습니다. 매뉴얼 PDF를 이 경로에 넣어주세요.")
# else:
#     db,_ = build_db(PDF_PATH, OPENAI_API_KEY)
#     # build_db_text= build_db("./pdf/제품사용설명서.pdf", OPENAI_API_KEY)
#     # print(build_db_text)
#     prev = ""
#     for text in answer("예약 청소는 최대 몇 시간 뒤까지 설정할 수 있나요?", OPENAI_API_KEY, db):
#         print(text[len(prev):], end="", flush=True)   # 새로 추가된 부분만 출력
#         prev = text
#     print()

# ===== 3. 앱 시작 시 매뉴얼 불러오기 =====
def load_manual():
    if not OPENAI_API_KEY:
        return None, "⚠️ `.env`에 `OPENAI_API_KEY`가 없습니다."
    if not os.path.exists(PDF_PATH):
        return None, f"⚠️ `{PDF_PATH}` 파일이 없습니다. 매뉴얼 PDF를 이 경로에 넣어주세요."
    try:
        return build_db(PDF_PATH, OPENAI_API_KEY)
    except Exception as e:
        print("매뉴얼 로드 실패:", e)
        return None, "⚠️ 매뉴얼에서 글자를 읽지 못했습니다. PDF가 손상됐거나 스캔본인지 확인하세요."


DB, LOAD_STATUS = load_manual()


def ask(question, history):
    if DB is None:
        raise gr.Error("매뉴얼을 불러오지 못했습니다. 화면 위쪽 안내를 확인하세요.")
    if not question or not question.strip():
        raise gr.Error("질문을 입력하세요.")

    history = strip_options(history) + [
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


def export_chat(history):
    """대화 기록을 txt 파일로 만들어 다운로드 버튼에 연결"""
    if not history:
        return gr.DownloadButton(value=None, interactive=False)

    now = datetime.now()
    lines = [
        "가전제품 매뉴얼 상담 기록",
        f"문서: {DOC_NAME}",
        f"저장 시각: {now:%Y-%m-%d %H:%M}",
        "=" * 40,
        "",
    ]
    for msg in history:
        speaker = "질문" if msg["role"] == "user" else "답변"
        lines += [f"[{speaker}]", msg["content"], ""]

    path = os.path.join(tempfile.gettempdir(), f"매뉴얼상담_{now:%Y%m%d_%H%M%S}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return gr.DownloadButton(value=path, interactive=True)



# ===== 추천 질문 생성 =====
N_SUGGEST = 3

SUGGEST_PROMPT = ChatPromptTemplate.from_template(
    "너는 가전제품 사용설명서 도우미야.\n"
    "사용자가 방금 한 질문과 답변, 관련 [메뉴얼] 내용을 보고 "
    "사용자가 이어서 궁금해할 만한 질문 3개를 만들어줘.\n"
    "- [메뉴얼]에서 답을 찾을 수 있는 질문만 만들 것\n"
    "- 방금 한 질문과 같은 질문은 제외할 것\n"
    "- 한국어로, 각 질문은 30자 이내\n"
    "- 번호나 기호 없이 한 줄에 하나씩\n\n"
    "[질문]\n{question}\n\n[답변]\n{answer}\n\n[메뉴얼]\n{context}"
)


def strip_options(history):
    """이전 답변에 붙어 있던 추천 질문 버튼 제거"""
    return [{"role": m["role"], "content": m["content"]} for m in history]


def pick_suggestion(evt: gr.SelectData):
    """답변 아래 추천 질문을 누르면 그 질문을 입력창에 넣음"""
    return evt.value


def suggest_questions(history):
    """마지막 질문·답변과 관련 매뉴얼 내용을 보고, 이어서 물어볼 질문을 답변 바로 아래에 붙임"""
    if DB is None or len(history) < 2:
        return history, history
    last_q = history[-2]["content"]
    last_a = history[-1]["content"]

    try:
        docs = DB.similarity_search(last_q, k=4)
        context = "\n\n".join(doc.page_content for doc in docs)
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, openai_api_key=OPENAI_API_KEY)
        text = (SUGGEST_PROMPT | llm | StrOutputParser()).invoke(
            {"question": last_q, "answer": last_a, "context": context}
        )
    except Exception as e:
        print("추천 질문 생성 실패:", e)
        return history, history

    asked = {m["content"] for m in history if m["role"] == "user"}  # 이미 한 질문은 제외
    questions = []
    for line in text.splitlines():
        q = re.sub(r"^\s*(?:[-•*]|\d+[.)])\s*", "", line).strip()
        if q and q not in asked and q not in questions:
            questions.append(q)
    questions = questions[:N_SUGGEST]
    if not questions:
        return history, history

    options = [{"value": q, "label": f"💡 {q}"} for q in questions]
    history = history[:-1] + [{**history[-1], "options": options}]
    return history, history


def scene_head():
    """구름·언덕·걷는 캐릭터 배경 (assets/scene.js). 파일이 없으면 생략"""
    if not os.path.exists(SCENE_JS_PATH):
        return ""
    images = []
    for candidates in WALKERS:
        for name in candidates:
            path = os.path.join(ASSETS_DIR, name)
            if os.path.exists(path):
                mime = "image/webp" if name.endswith(".webp") else "image/png"
                with open(path, "rb") as f:
                    images.append(f"data:{mime};base64," + base64.b64encode(f.read()).decode())
                break
    with open(SCENE_JS_PATH, encoding="utf-8") as f:
        js = f.read()
    return f"<script>window.SCENE_IMAGES = {json.dumps(images)};</script><script>{js}</script>"


# ===== 4. 화면 구성 =====
THEME = gr.themes.Soft(primary_hue="teal", secondary_hue="emerald", neutral_hue="slate").set(
    button_primary_background_fill="#3CC8A8",
    button_primary_background_fill_hover="#2FB394",
    button_primary_text_color="white",
    button_primary_border_color="#3CC8A8",
)

CSS = """
.gradio-container { max-width: 820px !important; margin: 0 auto !important; padding-bottom: 60px !important; }
#header { text-align: center; }
#chat { border: 1px solid #CDEFE6 !important; border-radius: 14px !important; }
html { background: #FFFFFF !important; }
body, gradio-app, .gradio-container, .main, .app { background: transparent !important; }  /* 구름이 내용 뒤에 보이도록 */
button.secondary:disabled { opacity: 1 !important; color: #A0AEC0 !important; }  /* 비활성 버튼도 불투명하게 */
.gallery-item { background: #FFFFFF !important; }  /* 예시 질문 버튼을 불투명하게: 배경 캐릭터가 비치지 않도록 */
#scene-sky { position: fixed; top: 0; left: 0; width: 100%; height: 150px; overflow: hidden; pointer-events: none; z-index: -1; }
.scene-cloud { position: absolute; left: 100%; animation: cloud-drift linear infinite; }
@keyframes cloud-drift { from { transform: translateX(0); } to { transform: translateX(calc(-100vw - 200px)); } }
#scene-hills { position: fixed; left: 0; bottom: 0; pointer-events: none; z-index: -1; }  /* 배경: 모든 내용 뒤 */
.scene-walker { position: fixed; height: 96px; pointer-events: none; z-index: -1; transform-origin: 50% 100%; }
@media (prefers-reduced-motion: reduce) { .scene-cloud { animation: none; } }
#chat .avatar-container { width: 52px !important; height: 52px !important; }
#chat .avatar-container img { border-radius: 50% !important; border: 1px solid #9EE3D2 !important; object-fit: cover !important; }
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
    gr.Markdown(
        f"# 💬 가전제품 매뉴얼 상담봇\n📘 {DOC_NAME} · {LOAD_STATUS}",
        elem_id="header",
    )
    history_state = gr.State([])

    chatbot = gr.Chatbot(
        elem_id="chat",
        height="calc(100vh - 440px)",  # 화면 높이에 맞춰 채팅창 크기 조절 (스크롤 방지)
        min_height=280,
        show_label=False,
        placeholder="**무엇이든 물어보세요!**<br>사용설명서를 대신 읽고 답해드려요.",
        avatar_images=(None, AVATAR_PATH),  # (사용자, 챗봇) 프로필 이미지
    )
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
    with gr.Row():
        clear_btn = gr.Button("🗑️ 대화 지우기", size="sm", variant="secondary")
        download_btn = gr.DownloadButton("💾 대화 저장", size="sm", variant="secondary", interactive=False)

    # ===== 5. 이벤트 연결 =====
    chat_io = dict(fn=ask, inputs=[question, history_state], outputs=[chatbot, history_state, question])

    def after_answer(event):
        """답변 후: 대화 저장 파일 갱신 → 답변 아래에 추천 질문 붙이기"""
        (
            event.success(export_chat, inputs=history_state, outputs=download_btn)
            .success(suggest_questions, inputs=history_state, outputs=[chatbot, history_state])
        )

    after_answer(send_btn.click(**chat_io))
    after_answer(question.submit(**chat_io))
    after_answer(chatbot.option_select(pick_suggestion, outputs=question).then(**chat_io))  # 추천 질문 클릭

    for clear_event in (clear_btn.click, chatbot.clear):  # 대화 지우기 버튼, 채팅창 휴지통 아이콘
        clear_event(clear_chat, outputs=[chatbot, history_state, question]).then(
            export_chat, inputs=history_state, outputs=download_btn
        )

if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS, head=scene_head())