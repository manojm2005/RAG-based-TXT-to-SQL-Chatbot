# 📊 RAG Chat SQL Assistant

An intelligent **Retrieval-Augmented Generation (RAG)**–powered SQL assistant that lets you **chat with your MySQL database** using natural language.
This project uses **Google Gemini**, **LangChain**, **ChromaDB**, and **SentenceTransformer** embeddings to dynamically retrieve schema context and generate safe SQL queries — all within an interactive **Streamlit** interface.

---

## 🧠 Overview

RAG Chat SQL Assistant allows users to ask natural language questions (e.g., *“Show me the top 5 customers by payment in 2004”*) and automatically:

1. Retrieves relevant database schema context from ChromaDB.
2. Generates SQL queries safely using Google Gemini.
3. Executes validated SQL queries against a MySQL database.
4. Displays tabular results and insights in Streamlit.

This project demonstrates how **LLMs + RAG + SQL** can be combined for **enterprise-level database intelligence**.

---

## 🧹 Tech Stack

| Component                  | Technology                                              |
| -------------------------- | ------------------------------------------------------- |
| **Frontend / UI**          | [Streamlit](https://streamlit.io/)                      |
| **LLM Engine**             | [Google Gemini (Generative AI)](https://ai.google.dev/) |
| **RAG Vector Store**       | [ChromaDB](https://www.trychroma.com/)                  |
| **Embeddings Model**       | SentenceTransformer (`all-MiniLM-L6-v2`)                |
| **Database**               | MySQL (ClassicModels schema)                            |
| **Frameworks / Libraries** | LangChain, SQLAlchemy, Altair, Pandas                   |
| **Environment**            | Python 3.11+                                            |

---

## 🧱 Project Structure

```
rag-sql-assistant/
│
├── app/
│   ├── main.py                   # Streamlit app (core logic)
│   ├── mysqlsampledatabase.sql   # ClassicModels sample schema
│   ├── requirements.txt          # Python dependencies
│   ├── .env.example              # Environment variable template
│
├── .gitignore
├── README.md
```

---

## 🚀 Features

✅ **Natural language to SQL** using Gemini Pro
✅ **Schema-aware retrieval** via Chroma + embeddings
✅ **Safe SQL execution** with strict validation (no DDL/DML)
✅ **Interactive chat UI** built with Streamlit
✅ **MySQL integration** via SQLAlchemy
✅ **Persistent vector store** for schema retrieval context

---

## ⚙️ Setup & Installation

### 1️⃣ Clone this repository

```bash
git clone https://github.com/<your-username>/rag-sql-assistant.git
cd rag-sql-assistant/app
```

### 2️⃣ Create and activate a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Configure environment variables

Copy the example file and update values:

```bash
cp .env.example .env
```

Example `.env`:

```bash
MYSQL_USER=root
MYSQL_PASSWORD=yourpassword
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_DB=classicmodels
CHROMA_DIR=./chroma_db
EMBED_MODEL_NAME=all-MiniLM-L6-v2
GOOGLE_API_KEY=your_api_key_here
MAX_ROWS=500
TOP_K=6
```

> ⚠️ **Do not commit `.env`** — it contains sensitive credentials and API keys.

---

## 🧠 Running the App

Start the Streamlit server:

```bash
streamlit run main.py
```

Open your browser at:
🔗 **[http://localhost:8501](http://localhost:8501)**

---

## 🧹 How It Works

1. **Schema Ingestion**

   * Click “Ingest Schema” to embed the MySQL schema into ChromaDB.
   * Embeddings are generated using SentenceTransformer.

2. **Query Understanding**

   * Your question is converted to a prompt with schema context.

3. **SQL Generation**

   * Google Gemini generates a secure, read-only SQL query.

4. **Query Validation & Execution**

   * The SQL is checked for unsafe operations (e.g., DROP, DELETE).
   * Executed on MySQL through SQLAlchemy.

5. **Visualization**

   * Results are displayed interactively with Streamlit and Altair.

---

## 🧹 Example Query

**Input:**

> "Show top 10 customers by total payments in 2004"

**Generated SQL:**

```sql
SELECT c.customerName, SUM(p.amount) AS total_payments
FROM customers c
JOIN payments p ON c.customerNumber = p.customerNumber
WHERE YEAR(p.paymentDate) = 2004
GROUP BY c.customerName
ORDER BY total_payments DESC
LIMIT 10;
```

**Output:**
Interactive table visualization of top 10 customers.

---

## 🔐 Security & Best Practices

* 🔒 All SQL queries are **validated** against destructive commands (`DROP`, `DELETE`, `UPDATE`, etc.).
* 🧱 Environment variables are stored securely in `.env`.
* 🧠 ChromaDB embeddings are persisted locally for reusability.
* 🚫 Sensitive files (`.env`, `__pycache__`, etc.) are ignored via `.gitignore`.

---

## 💡 Future Enhancements

* [ ] Add Docker and Docker Compose support
* [ ] Integrate caching for repeated queries
* [ ] Add charting options for analytical queries
* [ ] Support for multiple databases / schemas

---

## 🧮 Troubleshooting

| Issue                      | Solution                                        |
| -------------------------- | ----------------------------------------------- |
| `Missing GOOGLE_API_KEY`   | Add your Gemini API key to `.env`               |
| `SQL_NONE returned`        | Query not understood — try rephrasing           |
| `MySQL connection refused` | Verify your DB credentials and host             |
| Chroma not found           | Create folder path in `.env` or rerun ingestion |

---

## 🧑‍💻 Author

**👨‍💻 MANOJ M**
AI & ML Developer | AI&DS Student


---

## 🛪️ License

This project is licensed under the **MIT License** — you’re free to use, modify, and distribute it with attribution.

---

## ⭐ Support

If you find this project useful:

* ⭐ Star this repository on GitHub
* 💬 Share feedback or open an issue
* 🤝 Contribute with pull requests

> *Made with ❤️ using Python, Streamlit, and Gemini AI*
