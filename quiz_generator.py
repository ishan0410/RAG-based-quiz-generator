"""
quiz_generator.py — LLM-Powered Quiz Generation Engine

Generates multiple-choice questions from retrieved document chunks,
with support for difficulty levels, option shuffling, scoring, and
persistent quiz history.
"""

import json
import logging
import os
import random
import time
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Literal

from groq import Groq

logger = logging.getLogger(__name__)

Difficulty = Literal["easy", "medium", "hard"]


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class MCQOption:
    label: str          # "A", "B", "C", "D"
    text: str

@dataclass
class MCQuestion:
    question: str
    options: List[MCQOption]
    correct_label: str  # e.g. "B"
    explanation: str
    difficulty: Difficulty
    source_page: Optional[int] = None

@dataclass
class QuizResult:
    question: MCQuestion
    user_answer: Optional[str] = None   # label chosen
    is_correct: Optional[bool] = None

@dataclass
class QuizSession:
    questions: List[MCQuestion] = field(default_factory=list)
    results: List[QuizResult] = field(default_factory=list)
    score: int = 0
    total_answered: int = 0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def total_questions(self) -> int:
        return len(self.questions)

    @property
    def score_pct(self) -> float:
        return (self.score / self.total_answered * 100) if self.total_answered else 0.0


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert quiz creator. You generate high-quality multiple-choice "
    "questions from educational material. Every question must be grounded in "
    "the provided context — never invent facts."
)

def _build_user_prompt(context: str, num_questions: int, difficulty: Difficulty) -> str:
    diff_guidance = {
        "easy": "Focus on basic recall and simple definitions.",
        "medium": "Test understanding, comparisons, and application of concepts.",
        "hard": "Require synthesis, critical thinking, or multi-step reasoning.",
    }

    return f"""Based **only** on the following context, generate exactly {num_questions} \
multiple-choice question(s) at the **{difficulty}** difficulty level.

{diff_guidance[difficulty]}

### Context
\"\"\"
{context}
\"\"\"

### Output format
Return a JSON array. Each element must follow this schema exactly:
{{
  "question": "...",
  "options": {{
    "A": "...",
    "B": "...",
    "C": "...",
    "D": "..."
  }},
  "correct": "A" | "B" | "C" | "D",
  "explanation": "..."
}}

Rules:
- Only ONE correct answer per question.
- All four options must be plausible.
- Explanations should reference the context.
- Return ONLY the JSON array — no markdown fences, no extra text.
"""


# ---------------------------------------------------------------------------
# Generator class
# ---------------------------------------------------------------------------

class QuizGenerator:
    """
    Uses OpenAI chat completions to create MCQs from context chunks.

    Parameters
    ----------
    model : str
        OpenAI model name (default gpt-4o-mini for cost efficiency).
    temperature : float
        Sampling temperature — lower = more deterministic.
    max_retries : int
        Retries on transient API / parse failures.
    history_dir : str | None
        Directory to persist quiz sessions as JSON. None disables persistence.
    """

    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.7,
        max_retries: int = 2,
        history_dir: Optional[str] = None,
    ):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY not set. Add it to your .env file or export it."
            )

        self.client = Groq(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.history_dir = history_dir

        if self.history_dir:
            os.makedirs(self.history_dir, exist_ok=True)

    # ---- core generation ---------------------------------------------------

    def generate_questions(
        self,
        context_chunks: list,
        num_questions: int = 5,
        difficulty: Difficulty = "medium",
        shuffle_options: bool = True,
    ) -> List[MCQuestion]:
        """
        Generate *num_questions* MCQs from *context_chunks*.

        Parameters
        ----------
        context_chunks : list[Document]
            LangChain Document objects (or anything with .page_content).
        num_questions : int
            How many questions to produce.
        difficulty : str
            "easy", "medium", or "hard".
        shuffle_options : bool
            If True, randomly reorder A-D after generation.

        Returns
        -------
        list[MCQuestion]
        """
        if not context_chunks:
            raise ValueError("No context provided — upload a PDF first")

        # Merge chunk texts into one context string (truncate to ~12k chars
        # so we stay well within model context limits)
        merged = "\n\n".join(
            getattr(c, "page_content", str(c)) for c in context_chunks
        )[:12_000]

        raw = self._call_llm(merged, num_questions, difficulty)
        questions = self._parse_response(raw, difficulty)

        # Attach source page from first chunk if available
        first_page = None
        if context_chunks and hasattr(context_chunks[0], "metadata"):
            first_page = context_chunks[0].metadata.get("page")

        for q in questions:
            q.source_page = first_page

        if shuffle_options:
            questions = [self._shuffle_question(q) for q in questions]

        return questions[:num_questions]

    # ---- LLM call with retries --------------------------------------------

    def _call_llm(self, context: str, num_questions: int, difficulty: Difficulty) -> str:
        last_err = None
        for attempt in range(1, self.max_retries + 2):
            try:
                logger.info(
                    "LLM call attempt %d (model=%s, n=%d, diff=%s)",
                    attempt, self.model, num_questions, difficulty,
                )
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": _build_user_prompt(context, num_questions, difficulty),
                        },
                    ],
                )
                return response.choices[0].message.content.strip()
            except Exception as exc:
                last_err = exc
                logger.warning("LLM attempt %d failed: %s", attempt, exc)
                if attempt <= self.max_retries:
                    time.sleep(2 ** attempt)  # exponential back-off

        raise RuntimeError(f"LLM call failed after {self.max_retries + 1} attempts: {last_err}")

    # ---- JSON parsing ------------------------------------------------------

    @staticmethod
    def _parse_response(raw: str, difficulty: Difficulty) -> List[MCQuestion]:
        # Strip markdown fences if the model wraps them
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1]
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]
        cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM returned invalid JSON: {exc}\nRaw output:\n{raw[:500]}")

        if not isinstance(data, list):
            raise ValueError("Expected a JSON array of questions")

        questions: List[MCQuestion] = []
        for item in data:
            opts = item.get("options", {})
            options = [MCQOption(label=k, text=v) for k, v in opts.items()]
            questions.append(
                MCQuestion(
                    question=item["question"],
                    options=options,
                    correct_label=item["correct"],
                    explanation=item.get("explanation", ""),
                    difficulty=difficulty,
                )
            )
        return questions

    # ---- Option shuffling --------------------------------------------------

    @staticmethod
    def _shuffle_question(q: MCQuestion) -> MCQuestion:
        """Shuffle option order while keeping correct_label accurate."""
        pairs = [(o.label, o.text) for o in q.options]
        correct_text = next(o.text for o in q.options if o.label == q.correct_label)
        random.shuffle(pairs)

        new_labels = ["A", "B", "C", "D"]
        new_options = []
        new_correct = q.correct_label
        for new_label, (_, text) in zip(new_labels, pairs):
            new_options.append(MCQOption(label=new_label, text=text))
            if text == correct_text:
                new_correct = new_label

        return MCQuestion(
            question=q.question,
            options=new_options,
            correct_label=new_correct,
            explanation=q.explanation,
            difficulty=q.difficulty,
            source_page=q.source_page,
        )

    # ---- Quiz session helpers ----------------------------------------------

    def create_session(self, questions: List[MCQuestion]) -> QuizSession:
        session = QuizSession(
            questions=questions,
            results=[QuizResult(question=q) for q in questions],
        )
        return session

    def answer_question(
        self, session: QuizSession, question_index: int, user_answer: str
    ) -> QuizResult:
        """Record a user's answer and return the graded result."""
        if question_index < 0 or question_index >= len(session.results):
            raise IndexError("Question index out of range")

        result = session.results[question_index]
        result.user_answer = user_answer
        result.is_correct = user_answer == result.question.correct_label

        # Recompute totals
        answered = [r for r in session.results if r.user_answer is not None]
        session.total_answered = len(answered)
        session.score = sum(1 for r in answered if r.is_correct)

        return result

    # ---- Persistence -------------------------------------------------------

    def save_session(self, session: QuizSession) -> Optional[str]:
        """Persist a QuizSession to disk as JSON. Returns the file path."""
        if not self.history_dir:
            return None

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.history_dir, f"quiz_{ts}.json")

        # Convert dataclass tree to dicts
        payload = {
            "created_at": session.created_at,
            "score": session.score,
            "total_answered": session.total_answered,
            "total_questions": session.total_questions,
            "score_pct": round(session.score_pct, 1),
            "results": [],
        }
        for r in session.results:
            payload["results"].append({
                "question": r.question.question,
                "options": {o.label: o.text for o in r.question.options},
                "correct": r.question.correct_label,
                "explanation": r.question.explanation,
                "difficulty": r.question.difficulty,
                "user_answer": r.user_answer,
                "is_correct": r.is_correct,
            })

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        logger.info("Quiz session saved → %s", path)
        return path
