import os, re, json
import pandas as pd
import streamlit as st
import altair as alt
from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv


# Load environment

load_dotenv()

MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = os.getenv("MYSQL_PORT", "3306")
MYSQL_DB = os.getenv("MYSQL_DB", "classicmodels")
MYSQL_URI = f"mysql+mysqlconnector://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"

CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
EMBED_MODEL = os.getenv("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")
MAX_ROWS = int(os.getenv("MAX_ROWS", "500"))
TOP_K = int(os.getenv("TOP_K", "6"))
SQL_FILE_PATH = "mysqlsampledatabase.sql"

# Initialize Chroma

chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
COLLECTION_NAME = "classicmodels_schema"
collection_names = [c.name for c in chroma_client.list_collections()]
collection = (
    chroma_client.get_collection(COLLECTION_NAME)
    if COLLECTION_NAME in collection_names
    else chroma_client.create_collection(name=COLLECTION_NAME)
)


# Embedding model

embed_model = SentenceTransformer(EMBED_MODEL)
def embed_texts(texts): return embed_model.encode(texts, show_progress_bar=False).tolist()


# SQL ingestion utilities

CREATE_TABLE_RE = re.compile(r"CREATE\s+TABLE\s+`?([a-zA-Z0-9_]+)`?\s*\((.*?)\)\s*;?", re.S | re.I)
INSERT_RE = re.compile(r"INSERT\s+INTO\s+`?([a-zA-Z0-9_]+)`?\s*\((.*?)\)\s*VALUES\s*\((.*?)\)\s*;?", re.S | re.I)

def extract_schema_docs(sql_text):
    docs = []
    for m in CREATE_TABLE_RE.finditer(sql_text):
        table = m.group(1)
        snippet = "\n".join([ln.strip() for ln in m.group(2).splitlines()[:40]])
        docs.append({"id": f"schema::{table}", "table": table, "type": "schema",
                     "content": f"CREATE TABLE {table} (\n{snippet}\n)"})
    for m in INSERT_RE.finditer(sql_text):
        table = m.group(1)
        snippet = f"INSERT INTO {table} ({m.group(2)}) VALUES ({m.group(3)})"
        docs.append({"id": f"insert::{table}::{abs(hash(snippet))%1000000}",
                     "table": table, "type": "insert", "content": snippet})
    return docs

def ingest_sql_to_chroma(path):
    sql = open(path, "r", encoding="utf-8", errors="ignore").read()
    docs = extract_schema_docs(sql)
    if not docs: return 0
    texts = [d["content"] for d in docs]
    ids = [d["id"] for d in docs]
    metas = [{"table": d["table"], "type": d["type"]} for d in docs]
    embeddings = embed_texts(texts)
    collection.add(documents=texts, metadatas=metas, ids=ids, embeddings=embeddings)
    return len(docs)

# Retrieval

def retrieve_context(query, k=TOP_K):
    q_emb = embed_model.encode([query])[0].tolist()
    res = collection.query(query_embeddings=[q_emb], n_results=k, include=["documents","metadatas"])
    docs = res["documents"][0]; metas = res["metadatas"][0]
    return [{"doc": d, "meta": m} for d, m in zip(docs, metas)]

# Gemini LLM

def get_gemini():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("Missing GOOGLE_API_KEY in environment.")
        st.stop()
    return ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=api_key, temperature=0.0)

# SQL safety + execution
ALLOWED_SQL_RE = re.compile(r"^\s*(WITH|SELECT)\b", re.I)
BANNED_WORDS = ("DROP","DELETE","UPDATE","INSERT","ALTER","TRUNCATE","CREATE","REPLACE")

def validate_sql(sql):
    if ";" in sql.strip().rstrip(";"): return False
    if not ALLOWED_SQL_RE.match(sql): return False
    if any(re.search(rf"\b{b}\b", sql, re.I) for b in BANNED_WORDS): return False
    return True

engine = create_engine(MYSQL_URI, pool_pre_ping=True)
def run_query_safe(sql, limit=MAX_ROWS):
    wrapped = f"SELECT * FROM ({sql.strip().rstrip(';')}) AS _q LIMIT :__limit"
    with engine.connect() as conn:
        res = conn.execute(text(wrapped), {"__limit": limit})
        df = pd.DataFrame(res.fetchall(), columns=res.keys())
    return df

# Prompt & parsing
PROMPT_TEMPLATE = """You are a SQL assistant for the MySQL classicmodels database.
Rules:
- Return only a single-line SELECT query (no markdown or commentary).
- Use only schema provided.
- If unsure, return SQL_NONE.

Schema Context:
{context}

Conversation History:
{history}

User Question:
{question}

Output:"""

def build_prompt(context_docs, question, history):
    ctx = "\n\n".join([f"/*{d['meta']['table']}*/\n{d['doc']}" for d in context_docs])
    hist_str = "\n".join([f"Q: {h['user']}\nA: {h['sql']}" for h in history[-5:]]) if history else "None"
    return PROMPT_TEMPLATE.format(context=ctx, question=question, history=hist_str)

def parse_response(txt):
    t = txt.strip()
    if "SQL_NONE" in t: return "SQL_NONE"
    for ln in t.splitlines():
        ln = ln.strip()
        if ln.upper().startswith(("SELECT","WITH")): return ln
    return t.splitlines()[0] if t else ""

# Streamlit UI

st.set_page_config("RAG Chat SQL Assistant + Viz", layout="wide", page_icon="📊")
st.title("📊 RAG Chat SQL Assistant (Gemini + MySQL + Visualization)")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar
with st.sidebar:
    st.header("⚙️ Controls")
    if st.button("🧠 Ingest Schema"):
        count = ingest_sql_to_chroma(SQL_FILE_PATH)
        st.success(f"Ingested {count} schema docs.")
    if st.button("🧹 Clear Chat"):
        st.session_state.chat_history = []
        st.success("Chat cleared.")

# Chat Input
user_input = st.chat_input("Ask a question about Database (e.g., Top 5 customers by payments in 2004)")

if user_input:
    st.session_state.chat_history.append({"user": user_input, "sql": "", "result": None})
    with st.spinner("Generating SQL with Gemini..."):
        ctx = retrieve_context(user_input)
        prompt = build_prompt(ctx, user_input, st.session_state.chat_history)
        llm = get_gemini()
        try:
            resp_object = llm.invoke(prompt)
            resp = resp_object.content
        except Exception as e:
            st.error(f"Gemini error: {e}")
            st.stop()
        sql = parse_response(resp)

    if sql == "SQL_NONE" or not validate_sql(sql):
        st.session_state.chat_history[-1]["sql"] = "❌ Invalid or no SQL generated"
        st.warning("No valid SQL could be generated for this question.")
    else:
        st.session_state.chat_history[-1]["sql"] = sql
        with st.spinner("Executing SQL..."):
            try:
                df = run_query_safe(sql)
                st.session_state.chat_history[-1]["result"] = df
            except Exception as e:
                st.session_state.chat_history[-1]["result"] = f"⚠️ Query failed: {e}"

# Display chat 
for turn in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(turn["user"])
    with st.chat_message("assistant"):
        st.code(turn["sql"], language="sql")

        if isinstance(turn["result"], pd.DataFrame):
            df = turn["result"]
            st.dataframe(df.head(50))


        elif isinstance(turn["result"], str):
            st.warning(turn["result"])
