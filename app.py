import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI

st.set_page_config(page_title="Sports-Niti", page_icon="🏅")
st.title("🏅 Sports-Niti")
st.caption("Khelo India Scheme & Talent Guidance Assistant")

@st.cache_resource
def load_system():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    store = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name="sports_niti",
        url=st.secrets["QDRANT_URL"],
        api_key=st.secrets["QDRANT_API_KEY"],
    )
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        google_api_key=st.secrets["GEMINI_API_KEY"],
        temperature=0.2,
    )
    return store.as_retriever(search_kwargs={"k": 4}), llm

retriever, llm = load_system()

def ask(question):
    found = retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in found)
    prompt = f"""You are Sports-Niti, an assistant for the Khelo India Scheme and talent guidance.
Answer ONLY using the context below. If the answer is not in the context, say:
"I couldn't find this in the Khelo India documents."

Context:
{context}

Question: {question}
Answer:"""
    resp = llm.invoke(prompt).content
    if isinstance(resp, list):
        answer = "".join(b.get("text", "") for b in resp if isinstance(b, dict) and b.get("type") == "text")
    else:
        answer = resp
    sources = sorted({f"{d.metadata.get('source', 'document')} (page {d.metadata.get('page', 0) + 1})" for d in found})
    return answer, sources

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if q := st.chat_input("Ask about Khelo India, wrestling or boxing selection..."):
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            answer, sources = ask(q)
        st.markdown(answer)
        st.caption("Sources: " + ", ".join(sources))
    st.session_state.messages.append({"role": "assistant", "content": answer})
