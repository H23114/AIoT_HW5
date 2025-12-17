# AIoT HW5 - 個人化學術文獻 RAG 助理 (MyStudy RAG Agent)

**學生姓名：** 洪慧珊
**學號：** 7114056078
**作業主題：** 【生成式 AI】 07. 檢索增強生成(RAG)的原理及實作
**網址：** https://aiothw5-3itlsghnk3qtmx4ub9ttrj.streamlit.app/

---

## 1. 專案簡介 (Abstract)
本專案參考課程中關於檢索增強生成（RAG）的原理，開發了一個基於 Streamlit 的互動式網頁應用程式。系統整合了 LangChain 框架與 Google Gemini 模型，旨在解決大型語言模型在面對私有或特定領域文件時的幻覺問題。

使用者僅需上傳 PDF 格式的學術論文或講義，系統即會透過 `text-embedding-004` 模型進行向量化並儲存於 FAISS 資料庫。當使用者提問時，系統會檢索相關文獻片段作為 Context，並透過 `gemini-pro` 生成精確的回答。

## 2. 部署網址 (Demo URL)
專案已成功部署至 Streamlit Community Cloud，可直接訪問以下連結進行測試：
👉 **[https://aiothw5-3itlsghnk3qtmx4ub9ttrj.streamlit.app/](https://aiothw5-3itlsghnk3qtmx4ub9ttrj.streamlit.app/)**

> **注意：** 使用時請在左側欄位輸入有效的 Google Gemini API Key (可於 Google AI Studio 免費申請)。

## 3. 使用技術棧 (Tech Stack)
* **前端介面：** Streamlit
* **LLM 框架：** LangChain (v0.2 穩定版)
* **語言模型：** Google Gemini Pro (`gemini-pro`)
* **Embedding 模型：** Google Gemini Embeddings (`models/text-embedding-004`)
* **向量資料庫：** FAISS (Facebook AI Similarity Search)
* **PDF 處理：** PyPDFLoader

## 4. 檔案結構說明
* `streamlit_app.py`: 主程式碼，包含 UI 設計、RAG 流程邏輯與對話記憶功能。
* `requirements.txt`: 專案相依套件清單（已鎖定版本以確保雲端部署相容性）。
* `README.md`: 專案說明文件。

6. 功能特點
* **支援中文與英文 PDF**：針對中文語境優化了 Text Splitter 的切分符號。
* **對話記憶 (Memory)**：具備多輪對話能力，能根據上下文回答後續問題。
* **即時引用**：回答問題時會參考實際文件內容，減少 AI 胡說八道的機率。
* **雲端部署優化**：解決了 Streamlit Cloud 與 LangChain 新舊版本間的相容性問題。
