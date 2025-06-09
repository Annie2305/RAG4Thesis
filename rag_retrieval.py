from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

def get_all_authors(vectorstore):
    docs = vectorstore._collection.get(include=["metadatas"], limit=9999)["metadatas"]
    authors = set()
    for md in docs:
        author = md.get("author", None)
        if author:
            authors.add(author)
    return list(authors)

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device': 'cpu'}
    )
    vectorstore = Chroma(
        persist_directory='./vector_db',
        embedding_function=embeddings
    )
    return vectorstore

def run_rag_chat(question):
    vectorstore = load_vectorstore()
    authors = get_all_authors(vectorstore)
    llm = OllamaLLM(model="mistral-nemo")

    formatted_context = ""
    by_source = {}
    for author in authors:
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 3, "filter": {"author": author}}
        )
        docs = retriever.get_relevant_documents(question)
        texts = [doc.page_content for doc in docs]
        by_source[author] = texts
        if texts:
            all_text = '\n'.join(texts)
            formatted_context += f"{author}: {all_text}\n"
        else:
            formatted_context += f"{author}: No relevant content found.\n"

    # 英文版 prompt，強制每位作者都回答
    answer = llm.invoke(
        f"""
            You are an AI assistant that must answer **strictly based on the provided context grouped by author**. Do not use any prior knowledge. Carefully organize the answer by author—every author listed in the context must have their own entry, even if the answer is simply 'No relevant content found.'

            For each author, combine all meaningful, relevant content from the context into a clear and complete answer. Be detailed; do not summarize with a single sentence unless that's all the information available. If there is no relevant information, state explicitly: "No relevant content found."

            Use the following format:
            1. [Author Name]:
            - [Findings/description based only on the context]
            2. ...

            Context:
            {formatted_context}

            Question:
            {question}

            Answer:
        """
    )

    return {
        "answer": answer,
        "by_source": by_source
    }

if __name__ == "__main__":
    while True:
        question = input("請輸入問題: ")
        if question.lower() in ["exit", "quit", "q"]:
            break
        result = run_rag_chat(question)
        print('\nAnswer:')
        print(result['answer'])
        print('\nSources:')
        for author, texts in result['by_source'].items():
            print(f"\nAuthor: {author}")
            for text in texts:
                print(f"- {text[:1000]}")
                print('-' * 40)