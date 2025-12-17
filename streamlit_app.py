import streamlit as st
import os
import tempfile
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
import langchain
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# 設定頁面資訊
st.set_page_config(page_title="RAG 課程助理 (Gemini版)", page_icon="📚")
st.title("📚 RAG 學術課程助理")
st.caption("基於 Google Gemini 與 LangChain 的檢索增強生成系統")

# Sidebar: API Key 設定
with st.sidebar:
    st.header("設定")
    google_api_key = st.text_input("輸入 Google Gemini API Key", type="password")
    st.markdown("[取得 Google API Key](https://aistudio.google.com/app/apikey)")
    st.markdown("---")
    st.write("本系統由生成式 AI 課程專題實作延伸。")

# 初始化 Session State
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "你好！請上傳一份 PDF 講義或論文，我可以回答相關問題。"}]

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# 處理檔案上傳
uploaded_file = st.file_uploader("上傳 PDF 文件", type=["pdf"])

def process_pdf(uploaded_file, api_key):
    if not api_key:
        st.error("請先輸入 API Key")
        return None
    
    with st.spinner("正在分析文件..."):
        # 暫存檔案
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        # 讀取 PDF
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        
        # 切割文本
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""]
        )
        texts = text_splitter.split_documents(documents)
        
        # 建立向量庫
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        vector_store = FAISS.from_documents(texts, embeddings)
        
        os.remove(tmp_path) # 刪除暫存
        return vector_store

if uploaded_file and st.session_state.vector_store is None:
    if google_api_key:
        st.session_state.vector_store = process_pdf(uploaded_file, google_api_key)
        st.success("文件分析完成！請開始提問。")
    else:
        st.warning("請在左側輸入 Google API Key 以開始分析。")

# 顯示對話歷史
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 處理使用者輸入
if prompt := st.chat_input():
    if not google_api_key:
        st.info("請先輸入 API Key")
        st.stop()
        
    if st.session_state.vector_store is None:
        st.info("請先上傳 PDF 文件")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # RAG 鏈
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=google_api_key, temperature=0.3)
    
    # 建立 Chain
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=st.session_state.vector_store.as_retriever(search_kwargs={"k": 3}),
        memory=memory
    )
    
    # 生成回答
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            # 這裡為了簡單演示，不完全使用 memory chain 的歷史功能來避免複雜的 token 問題
            # 直接使用 retriever 找答案
            docs = st.session_state.vector_store.similarity_search(prompt, k=3)
            context = "\n".join([doc.page_content for doc in docs])
            
            # 組裝 Prompt
            system_prompt = f"你是一個專業的學術助教。請根據以下的上下文內容回答使用者的問題。如果上下文中沒有答案，請誠實說不知道。\n\n上下文：{context}\n\n問題：{prompt}"
            response = llm.invoke(system_prompt)
            
            st.write(response.content)
            st.session_state.messages.append({"role": "assistant", "content": response.content})
