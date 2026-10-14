import os                                                   # 운영체제 환경변수를 다루는 기본 모듈
from dotenv import load_dotenv                              # .env 파일을 읽어 주는 함수
from langchain_core.prompts import ChatPromptTemplate           # 프롬프트 '틀'을 만드는 클래스
from langchain_core.output_parsers import StrOutputParser   # 모델 응답에서 글자만 뽑아 주는 파서
from langchain_core.runnables import RunnablePassthrough    # 값을 통과시키며 새 키를 붙이는 부품
from langchain_openai import ChatOpenAI,OpenAIEmbeddings                       # OpenAI 채팅 모델 클래스
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever
import uuid

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")




def build_db(pdf_path, openai_key):
    if not openai_key:
        return None, "⚠️ OpenAI API 키를 먼저 입력한 뒤 PDF를 다시 올려주세요."
    if pdf_path is None:
        return None, ""

    pages = PyPDFLoader(pdf_path).load_and_split()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100, chunk_overlap=20,
        length_function=len, is_separator_regex=False,
    )
    texts = text_splitter.split_documents(pages)

    embeddings_model = OpenAIEmbeddings(
        model="text-embedding-3-large", openai_api_key=openai_key,
    )

    db = Chroma.from_documents(
        texts, embeddings_model, collection_name=f"pdf_{uuid.uuid4().hex}"
    )
    return db, f"✅ 처리 완료:{len(pages)}페이지 →{len(texts)}개 청크"





def answer(question, openai_key, db):
    # if db is None:
    #     raise gr.Error("먼저 API 키를 입력하고 PDF를 업로드하세요.")
    # if not question:
    #     raise gr.Error("질문을 입력하세요.")

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

    # print(rag_chain.invoke(question))

    #여기부터 재개
    text = ""
    for chunk in rag_chain.stream(question):
        text += chunk
        yield text

        # print(text, end="", flush=True)



db,_ = build_db("./pdf/제품사용설명서.pdf", OPENAI_API_KEY)
# build_db_text= build_db("./pdf/제품사용설명서.pdf", OPENAI_API_KEY)
# print(build_db_text)
prev = ""
for text in answer("키보드 청소하는 법", OPENAI_API_KEY, db):
    print(text[len(prev):], end="", flush=True)   # 새로 추가된 부분만 출력
    prev = text
print()