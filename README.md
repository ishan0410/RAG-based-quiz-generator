# RAG-Based Quiz Generator

A production-ready system that ingests PDF documents and generates interactive multiple-choice quizzes using Retrieval-Augmented Generation (RAG). Upload any PDF, and the app retrieves the most relevant passages to create context-aware questions with explanations — all through a clean Gradio web interface.

## Features

- **PDF Processing** — extracts and chunks text with LangChain's recursive splitter
- **Dual Retrieval** — switch between FAISS (semantic / dense) and TF-IDF (keyword / sparse) at any time
- **LLM Quiz Generation** — produces MCQs with four options, a correct answer, and an explanation via OpenAI
- **Difficulty Levels** — Easy, Medium, and Hard question sets
- **Interactive UI** — real-time answer grading, score tracking, and per-question feedback
- **Quiz History** — sessions are automatically saved to `quiz_history/` as JSON

## Project Structure

```
RAG-based-quiz-generator/
├── app.py               # Gradio frontend and callback logic
├── pdf_loader.py        # PDF extraction and chunking
├── retriever.py         # FAISS + TF-IDF retrieval with unified manager
├── quiz_generator.py    # LLM-powered MCQ generation and scoring
├── requirements.txt     # Python dependencies
├── .env.example         # Template for environment variables
├── quiz_history/        # Auto-created directory for saved sessions
└── README.md
```


## Tech Stack

- **Python 3.10+**
- **LangChain** — document loading, text splitting, embeddings
- **FAISS** — dense vector similarity search
- **scikit-learn** — TF-IDF sparse retrieval baseline
- **OpenAI API** — GPT-4o-mini for quiz generation
- **Gradio** — interactive web UI
- **python-dotenv** — environment variable management
