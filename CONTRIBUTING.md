# Contributing to DocuMind AI

Thank you for your interest in contributing to **DocuMind AI — Intelligent Document Q&A System using Retrieval-Augmented Generation (RAG)**!

---

## 1. Development Environment Setup

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/hnimje14/DocuMind-AI.git
   cd DocuMind-AI
   ```

2. **Create and activate a virtual environment:**
   - **Windows (PowerShell):**
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
   - **Linux / macOS:**
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables:**
   ```bash
   cp .env.example .env
   ```

---

## 2. Running the Test Suite

Before submitting a pull request, ensure all automated unit and integration tests pass:

```bash
pytest -v
```

To inspect test coverage across backend modules:

```bash
pytest --cov=backend.app --cov-report=term-missing
```

---

## 3. Coding & Architecture Guidelines

- **Modular Boundaries:** Keep FastAPI route controllers (`backend/app/api/`) separate from business logic in `ingestion`, `chunking`, `embeddings`, `vectorstore`, `retrieval`, `generation`, `memory`, and `evaluation`.
- **Type Safety & Validation:** Use Pydantic v2 schemas (`backend/app/schemas/models.py`) and Python type annotations for all public interfaces.
- **Security & Privacy:**
  - Never hardcode API keys, tokens, or credentials.
  - Never commit `.env` files, personal PDF documents, or runtime SQLite/FAISS databases (`data/uploads/*.pdf`, `data/*.db`, `data/vectorstore/*.bin`).
  - Use synthetic documents for all automated tests and benchmarks.

---

## 4. Submitting a Pull Request

1. Create a descriptive feature branch (`git checkout -b feat/your-feature-name` or `fix/issue-description`).
2. Add or update unit/integration tests in `backend/tests/` covering your changes.
3. Verify that `pytest -v` passes with zero errors or warnings.
4. Open a Pull Request with a clear summary of the problem and solution.
