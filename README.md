# 🧠 AI Text Summarizer 

A **Streamlit-based AI summarization app** that uses **Google’s Gemini 2.0 Flash model** via **LangChain** to summarize text or PDF files intelligently.
It supports **chunked summarization** for long documents and provides a simple, interactive UI.

---

## 🚀 Features

✅ Summarize large text or PDF documents

✅ Uses **Google Gemini 2.0 Flash** for fast, efficient summarization

✅ Supports **chunked summarization** (splits long documents automatically)

✅ Adjustable **summary length** and **temperature**

✅ User-friendly **Streamlit interface**

---

## 🛠️ Tech Stack

* **Python 3.10+**
* **Streamlit** — Web UI framework
* **LangChain** — LLM orchestration
* **Google Generative AI (Gemini 2.0 Flash)** — LLM backend
* **PyMuPDF (fitz)** — PDF text extraction
* **python-dotenv** — Environment variable management

---

## 📦 Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/<your-username>/ai-text-summarizer.git
cd ai-text-summarizer
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

If you don’t have a `requirements.txt`, create one with:

```txt
streamlit
langchain
google-generativeai
python-dotenv
PyMuPDF
```

---

## 🔑 Setup API Key

1. Get your **Google Gemini API key** from the [Google AI Studio](https://makersuite.google.com/app/apikey).
2. Create a `.env` file in the project root:

   ```bash
   GEMINI_API_KEY=your_api_key_here
   ```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

*(Rename your file if needed, e.g., `streamlit_langchain_gemini_summarizer.py` → `app.py`)*

Then open the link Streamlit provides, usually:

```
http://localhost:8501
```

---

## 📘 How It Works

1. **User Input:** Paste text or upload a `.txt`/`.pdf` file.
2. **LangChain Prompting:** The app uses a custom prompt to instruct Gemini 2.0 Flash for summarization.
3. **Chunk Handling:** For long texts, the document is split into chunks, summarized individually, and merged into a final concise summary.
4. **Display:** The summarized result appears on-screen instantly.

---

## ⚙️ Configuration Options

| Setting                   | Description                                     |
| ------------------------- | ----------------------------------------------- |
| **Summary length**        | Adjust number of sentences per chunk            |
| **Temperature**           | Controls creativity (0 = factual, 1 = creative) |
| **Chunked summarization** | Automatically summarizes long documents         |

---

## 📄 Example Output

**Input:**

> Artificial Intelligence (AI) is transforming industries through automation and predictive analytics...

**Summary:**

> AI automates tasks, improves efficiency, and offers predictive insights while raising ethical concerns about bias and privacy.

---

## 🧠 Model Info

* **Model Used:** `gemini-2.0-flash`
* **Provider:** [Google Generative AI](https://ai.google.dev/gemini-api/docs)
* **Benefits:** Fast response, low latency, cost-effective for summarization tasks.

---

## 🤝 Contributing

Contributions are welcome!
To contribute:

1. Fork this repo
2. Create a new branch
3. Make your changes
4. Submit a pull request 🚀

---

## 📜 License

This project is licensed under the **MIT License** — feel free to use and modify it.

---



