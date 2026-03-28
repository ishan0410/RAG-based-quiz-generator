"""
app.py — Gradio Frontend for the RAG-Based Quiz Generator

Launch with:
    python app.py

The UI provides PDF upload, retriever selection, difficulty control,
interactive MCQ answering with real-time feedback, and score tracking.
"""

import logging
import os
import sys
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

# Load .env before any module tries to read OPENAI_API_KEY
load_dotenv()

from pdf_loader import PDFLoader
from retriever import RetrieverManager
from quiz_generator import QuizGenerator, QuizSession

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application state (module-level singletons)
# ---------------------------------------------------------------------------
HISTORY_DIR = os.path.join(os.path.dirname(__file__), "quiz_history")

pdf_loader = PDFLoader(chunk_size=1000, chunk_overlap=200)
retriever_mgr = RetrieverManager(default_method="faiss")
quiz_gen = QuizGenerator(history_dir=HISTORY_DIR)

# Mutable state shared across callbacks
_state: dict = {
    "session": None,          # QuizSession
    "current_idx": 0,         # which question is displayed
    "pdf_loaded": False,
    "index_stats": None,
}


# =========================================================================
# Callback helpers
# =========================================================================

def _status(msg: str, is_error: bool = False) -> str:
    prefix = "ERROR: " if is_error else ""
    return f"{prefix}{msg}"


def process_pdf(file, retriever_method):
    """Handle PDF upload → extract → chunk → index."""
    if file is None:
        return _status("Please upload a PDF file.", is_error=True)

    file_path = file.name if hasattr(file, "name") else file
    try:
        result = pdf_loader.load_pdf(file_path)
    except (FileNotFoundError, ValueError) as exc:
        return _status(str(exc), is_error=True)

    retriever_mgr.set_method(retriever_method.lower())
    stats = retriever_mgr.index_documents(result.chunks)
    _state["pdf_loaded"] = True
    _state["index_stats"] = stats

    return _status(
        f"PDF processed — {result.page_count} pages, "
        f"{stats['total_chunks']} chunks indexed. "
        f"Active retriever: {stats['active_method'].upper()}."
    )


def generate_quiz(num_questions, difficulty, retriever_method):
    """Retrieve context + generate MCQs → show first question."""
    if not _state["pdf_loaded"]:
        return (
            _status("Upload and process a PDF first.", is_error=True),
            "", "", gr.update(choices=[], value=None), "", "",
        )

    retriever_mgr.set_method(retriever_method.lower())

    try:
        # Use a generic educational query to pull diverse chunks
        chunks = retriever_mgr.retrieve(
            "key concepts, definitions, and important information",
            top_k=8,
        )
        questions = quiz_gen.generate_questions(
            context_chunks=chunks,
            num_questions=int(num_questions),
            difficulty=difficulty.lower(),
            shuffle_options=True,
        )
    except Exception as exc:
        logger.exception("Quiz generation failed")
        return (
            _status(f"Generation failed: {exc}", is_error=True),
            "", "", gr.update(choices=[], value=None), "", "",
        )

    session = quiz_gen.create_session(questions)
    _state["session"] = session
    _state["current_idx"] = 0

    return _render_question(0, session)


def submit_answer(selected):
    """Grade the current answer, show feedback, advance pointer."""
    session: QuizSession = _state.get("session")
    if session is None:
        return (
            _status("Generate a quiz first.", is_error=True),
            "", "", gr.update(choices=[], value=None), "", "",
        )

    idx = _state["current_idx"]
    if selected is None:
        return _render_question(idx, session, extra_feedback="Please select an answer.")

    # Extract label from the radio text  ("A. …" → "A")
    label = selected.split(".")[0].strip()
    result = quiz_gen.answer_question(session, idx, label)

    correct_text = next(
        o.text for o in result.question.options if o.label == result.question.correct_label
    )

    if result.is_correct:
        feedback = f"Correct! The answer is {result.question.correct_label}. {correct_text}"
    else:
        feedback = (
            f"Incorrect. You chose {label}. "
            f"The correct answer is {result.question.correct_label}. {correct_text}"
        )
    feedback += f"\n\nExplanation: {result.question.explanation}"

    score_text = f"Score: {session.score} / {session.total_answered}  ({session.score_pct:.0f}%)"

    # Advance to next question (if any)
    _state["current_idx"] = idx + 1

    if idx + 1 < session.total_questions:
        status, q_text, progress, radio, _, _ = _render_question(idx + 1, session)
        return status, q_text, progress, radio, feedback, score_text
    else:
        # Quiz complete — save history
        quiz_gen.save_session(session)
        final_status = _status(
            f"Quiz complete! Final score: {session.score}/{session.total_questions} "
            f"({session.score_pct:.0f}%)"
        )
        return (
            final_status, "All questions answered!", "",
            gr.update(choices=[], value=None), feedback, score_text,
        )


def _render_question(idx, session, extra_feedback=""):
    """Build Gradio outputs for question at *idx*."""
    q = session.questions[idx]
    q_text = f"**Q{idx + 1}.** {q.question}"
    if q.difficulty:
        q_text += f"  _(Difficulty: {q.difficulty})_"

    choices = [f"{o.label}. {o.text}" for o in q.options]
    progress = f"Question {idx + 1} of {session.total_questions}"
    status = _status(f"Showing question {idx + 1}")
    score_text = (
        f"Score: {session.score} / {session.total_answered}"
        if session.total_answered
        else ""
    )

    return (
        status,
        q_text,
        progress,
        gr.update(choices=choices, value=None),
        extra_feedback,
        score_text,
    )


# =========================================================================
# Gradio UI
# =========================================================================

def build_app() -> gr.Blocks:
    with gr.Blocks(
        title="RAG Quiz Generator",
        theme=gr.themes.Soft(),
        css="""
            .main-title { text-align:center; margin-bottom:0.2em; }
            .subtitle   { text-align:center; color:#666; margin-bottom:1.5em; }
        """,
    ) as app:

        gr.Markdown("<h1 class='main-title'>RAG-Based Quiz Generator</h1>")
        gr.Markdown(
            "<p class='subtitle'>Upload a PDF, generate context-aware quizzes, "
            "and test your knowledge interactively.</p>"
        )

        # ---- Sidebar / Controls -------------------------------------------
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Settings")
                pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
                retriever_dd = gr.Dropdown(
                    choices=["FAISS", "TFIDF"],
                    value="FAISS",
                    label="Retriever",
                )
                num_q_slider = gr.Slider(
                    minimum=1, maximum=20, value=5, step=1,
                    label="Number of Questions",
                )
                diff_dd = gr.Dropdown(
                    choices=["Easy", "Medium", "Hard"],
                    value="Medium",
                    label="Difficulty",
                )
                process_btn = gr.Button("Process PDF", variant="secondary")
                generate_btn = gr.Button("Generate Quiz", variant="primary")
                status_box = gr.Textbox(label="Status", interactive=False, lines=2)

            # ---- Main quiz area -------------------------------------------
            with gr.Column(scale=2):
                gr.Markdown("### Quiz")
                progress_md = gr.Markdown("")
                question_md = gr.Markdown("")
                answer_radio = gr.Radio(
                    choices=[], label="Select your answer", interactive=True,
                )
                submit_btn = gr.Button("Submit Answer", variant="primary")
                feedback_box = gr.Textbox(
                    label="Feedback", interactive=False, lines=4,
                )
                score_md = gr.Markdown("")

        # ---- Wire callbacks ------------------------------------------------
        process_btn.click(
            fn=process_pdf,
            inputs=[pdf_input, retriever_dd],
            outputs=[status_box],
        )

        generate_btn.click(
            fn=generate_quiz,
            inputs=[num_q_slider, diff_dd, retriever_dd],
            outputs=[status_box, question_md, progress_md, answer_radio, feedback_box, score_md],
        )

        submit_btn.click(
            fn=submit_answer,
            inputs=[answer_radio],
            outputs=[status_box, question_md, progress_md, answer_radio, feedback_box, score_md],
        )

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
