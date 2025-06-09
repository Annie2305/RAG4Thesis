from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from pypdf import PdfReader
import os
import re

# 自動抓取作者（從 PDF 首頁/前兩頁找）
def extract_author(text, filename):
    author_patterns = [
        r"(研究生|博士生)\s*[:：]?\s*([^\n（(]+)",     # 中文(碩/博士論文)
        r"Author\s*[:：]?\s*([^\n（(]+)",              # 英文論文
        r"學生\s*[:：]?\s*([^\n（(]+)",                # 學生
        r"學號\s*[:：]?\s*.*?\n([^\n（(]+)",           # 學號下一行
    ]
    for pattern in author_patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(2).strip() if len(match.groups()) > 1 else match.group(1).strip()
    return os.path.splitext(filename)[0]

# 文件切割並加入作者metadata
def load_and_split_documents(folder_path):
    all_docs = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            filepath = os.path.join(folder_path, filename)
            reader = PdfReader(filepath)
            full_text = ""
            first_pages = ""
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    full_text += page_text + "\n"
                    if i < 2:
                        first_pages += page_text + "\n"
            author = extract_author(first_pages, filename)
            # chunking
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=100,
                separators=['\n\n', '\n', '.']
            )
            chunks = text_splitter.split_text(full_text)
            # 建立 Document 並存作者與來源
            from langchain.schema import Document
            docs = [Document(page_content=chunk, metadata={"author": author, "source": filename}) for chunk in chunks]
            all_docs.extend(docs)
            print(f"Processed: {filename} (author: {author}, {len(chunks)} chunks)")
    return all_docs

# Embedding 並存入 vector store
def save_to_vectorstore(docs):
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device': 'cpu'}
    )
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory='./vector_db'
    )
    print("向量資料庫儲存完成，共處理", len(docs), "段文本")

if __name__ == "__main__":
    docs = load_and_split_documents("./docs")
    print(f"共分成 {len(docs)} 段")
    save_to_vectorstore(docs)
