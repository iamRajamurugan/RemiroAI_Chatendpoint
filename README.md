# Remiro AI – Career Co‑Pilot

Remiro AI is a **multi‑agent, career‑focused AI assistant** that helps professionals clarify their identity, design long‑term strategies, grow skills, navigate workplace dynamics, and communicate their personal brand.

It is built as a **full-stack, production-style app**:

- A **multi‑agent reasoning graph** orchestrated with LangGraph & LangChain.
- A set of **specialist agents** (identity, motivation, strategy, skills, workplace, branding).
- **Long‑term memory** and **multi‑session chat history** stored in Supabase.
- A polished **Streamlit frontend** with authentication and session management.
- Live **web search integration** (Serper) for up‑to‑date market context.
- Runs on **Google Gemini** via `langchain-google-genai`.

This repo is intended both as a **portfolio‑grade project** and a realistic foundation for building advanced AI career copilots.

---

## ✨ Features

- **Holistic Career Coaching**
  - 6 specialist agents:
    - Core Identity Architect
    - Purpose & Motivation Navigator
    - Grand Strategy Director
    - Capability Growth Engineer
    - Workplace Dynamics & Culture Coach
    - Chief Marketing Officer (Personal Brand)
  - A **router agent** decides which specialists to invoke.
  - A **response synthesizer** unifies their outputs into one coherent reply.

- **Multi‑Agent Orchestration**
  - Built with **LangGraph** (stateful graph) + **LangChain**.
  - Nodes for routing, optional web search, specialists, synthesis, profile updates, and history summarization.
  - Shared state tracks:
    - Conversation messages
    - User profile (long‑term traits, constraints, preferences)
    - Active agents
    - Web search results
    - Per‑agent outputs

- **Persistent Memory via Supabase**
  - **Supabase Auth** for email/password login.
  - Tables for:
    - `profiles` – long‑term user profile (JSON).
    - `chat_sessions` – per‑user chat sessions.
    - `messages` – full conversation history.
  - Automatic:
    - Session creation / selection.
    - Saving and re‑loading messages.
    - Incremental profile updates after each turn.

- **Web Search Integration**
  - Uses **Serper** (`GoogleSerperAPIWrapper`) to fetch current market information.
  - A dedicated `WebSearcher` agent:
    - Calls Serper.
    - Asks the LLM to summarize and ground other agents with fresh data.
  - Graceful error handling if the search API key is missing or quota is exceeded.

- **Modern Streamlit Frontend**
  - Clean **login / signup** screen with project description.
  - **Sidebar**:
    - Shows signed‑in user.
    - “New chat” button.
    - Session list with selection.
  - **Chat UI**:
    - `st.chat_message` interface.
    - Clear prompt about allowed questions (career‑only).
    - Streaming‑style interaction with all backend agents.

- **Strict Domain Boundary**
  - Remiro AI answers **only career and work‑related questions**:
    - No real‑time stock prices
    - No medical / legal advice
    - No generic trivia
  - For out‑of‑scope questions, it politely redirects the user back to career topics.

---

## 🧱 Architecture Overview

High‑level flow:

1. **User message** comes from the Streamlit chat.
2. **Backend graph** (in [graph.py](graph.py)) runs the following pipeline:
   - Router agent → chooses relevant specialists (and optional web search).
   - Optional web search node → fetches + summarizes live data.
   - Specialist agents → each produces a focused analysis.
   - Response synthesizer → merges all agent outputs into one answer.
   - Profile updater → updates long‑term structured user profile in Supabase.
   - History manager → summarizes older messages to keep context small.
3. **Final answer** is returned to the frontend, displayed in the chat, and stored in Supabase along with updated profile.

Key components:

- **Orchestration & State**: [graph.py](graph.py)
- **Specialist Agents**: [agents/](agents)
- **Supabase Client & Auth**: [supabase_client.py](supabase_client.py)
- **Frontend UI**: [frontend/app.py](frontend/app.py)

---

## 🧠 Agents

Located in [agents/](agents):

- **CoreIdentityArchitect** – analyzes personality, strengths, preferences, and constraints.
- **PurposeMotivationNavigator** – clarifies values, intrinsic motivation, and meaning.
- **GrandStrategyDirector** – designs realistic multi‑year career strategies and milestones.
- **CapabilityGrowthEngineer** – performs skill‑gap analysis and builds learning roadmaps.
- **WorkplaceDynamicsCultureCoach** – helps with culture fit, conflict, and burnout.
- **ChiefMarketingOfficer** – focuses on personal brand, portfolio, resume, and LinkedIn.
- **QueryParser** – routes incoming queries to the right specialists and web search.
- **ResponseSynthesizer** – merges all agent outputs into a single, empathetic response.
- **WebSearcher** – fetches and summarizes web data (e.g., market trends).
- **ProfileUpdater** – maintains a structured, evolving user profile over time.

Each agent exposes a `get_chain()` method returning a LangChain runnable chain bound to a shared Gemini model.

---

## 🧰 Tech Stack

**Backend / Orchestration**

- Python 3.10+
- [LangChain](https://python.langchain.com/)
- [LangGraph](https://langgraph.dev/)
- [langchain-google-genai](https://docs.langchain.com/oss/python/integrations/chat/google_generative_ai) (Gemini chat models)

**Data & Auth**

- [Supabase](https://supabase.com/) (Postgres, Auth, REST)
- Supabase Python client v2 (`supabase`)

**Web Search**

- [Serper](https://serper.dev/) (`GoogleSerperAPIWrapper` from `langchain_community`)

**Frontend**

- [Streamlit](https://streamlit.io/)

---

Remiro AI – Proprietary License
Version 1.0 – December 16, 2025

Copyright (c) 2025 * VishCraft's REMIRO AI *.
All rights reserved.

This software and associated documentation files (the “Software”) are proprietary and confidential.
By accessing, viewing, or using the Software, you agree to the following terms:

1. Ownership
   The Software is owned exclusively by Vishcraft and Analyka Insights. No ownership rights are transferred
   by granting access to this repository.

2. Permitted Use
   You may:
   - View and locally run the Software for evaluation or internal, non-commercial purposes only.

3. Prohibited Use
   You may NOT, without prior written permission from * VishCraft's REMIRO AI *:
   - Copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software.
   - Use the Software or any derivative works for any commercial purpose.
   - Remove, alter, or obscure any proprietary notices, trademarks, or attribution.

4. No Redistribution
   Redistribution of the Software, in whole or in part, in any form or by any means, is strictly
   prohibited without express written consent from * VishCraft's REMIRO AI *.

5. No Warranty
   THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
   BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
   NON-INFRINGEMENT.

6. Limitation of Liability
   IN NO EVENT SHALL * VishCraft's REMIRO AI * BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY,
   WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM, OUT OF, OR IN CONNECTION
   WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

7. Termination
   Any breach of these terms automatically terminates your permission to use the Software. Upon
   termination, you must cease all use and destroy any local copies.

For permission requests, please contact:
Rajamurugan ,
Email : the.rajamurugan@gmail.com
