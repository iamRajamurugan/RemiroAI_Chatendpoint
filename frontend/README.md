# Remiro AI Frontend (Streamlit)

This is a simple Streamlit frontend for the Remiro AI multi-agent career copilot.
It uses Supabase Auth for email/password login and talks to the backend
orchestration defined in `graph.py`.

## Prerequisites

- Python 3.10+
- A configured `.env` file at the project root with:
  - `GOOGLE_API_KEY` (Gemini)
  - `SUPABASE_URL`
  - `SUPABASE_ANON_KEY`
  - `SERPER_API_KEY`
- Supabase project with the expected tables:
  - `profiles`
  - `chat_sessions`
  - `messages`

## Install dependencies

From the project root (where `requirements.txt` lives):

```bash
pip install -r requirements.txt
```

## Run the frontend

From the project root:

```bash
streamlit run frontend/app.py
```

Then open the URL shown in the terminal (typically `http://localhost:8501`).

You can sign up or log in with email/password, then start chatting with the
career copilot. Each conversation is stored as a separate session in Supabase
and can be revisited from the sidebar.
