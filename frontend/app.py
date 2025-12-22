import os
import sys
import time

import streamlit as st

# Ensure the project root (one level up) is on sys.path so we can
# import `graph` and `supabase_client` when this file is run via Streamlit.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from graph import run_session, list_user_sessions, get_session_messages
from supabase_client import sign_up_user, sign_in_user


st.set_page_config(
    page_title="Remiro AI – Career Co‑Pilot",
    layout="wide",
)


def init_state() -> None:
    if "user_id" not in st.session_state:
        st.session_state.user_id = None
    if "email" not in st.session_state:
        st.session_state.email = ""
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "auth_mode" not in st.session_state:
        st.session_state.auth_mode = "Login"


def logout() -> None:
    st.session_state.user_id = None
    st.session_state.email = ""
    st.session_state.session_id = None
    st.session_state.chat_history = []


def auth_description() -> None:
    """Render a short description of what Remiro AI does and its scope."""

    st.markdown(
        """
        ### Welcome to Remiro AI
        A holistic career advisory system that helps you:

        - Clarify your strengths, values, and long‑term direction.
        - Design realistic career strategies and learning roadmaps.
        - Navigate workplace dynamics and communicate your personal brand.

        Remiro AI focuses **only on career and work‑related questions** (not real‑time
        stock prices, medical, legal, or general trivia).
        """
    )


def render_auth_screen() -> None:
    left, right = st.columns([2.2, 1.3])

    with left:
        st.title("Remiro AI – Career Co‑Pilot")
        auth_description()

    with right:
        st.subheader("Sign in or create an account")

        mode = st.radio("Mode", ["Login", "Sign up"], key="auth_mode", horizontal=True)

        with st.form("auth_form", clear_on_submit=False):
            email = st.text_input("Email", key="auth_email")
            password = st.text_input("Password", type="password", key="auth_password")
            submit_label = "Log in" if mode == "Login" else "Create account"
            submitted = st.form_submit_button(submit_label)

        if submitted:
            if not email or not password:
                st.error("Please enter both email and password.")
                return

            try:
                if mode == "Login":
                    auth_result = sign_in_user(email, password)
                    st.success("Logged in successfully.")
                else:
                    auth_result = sign_up_user(email, password)
                    st.success("Account created and logged in.")

                st.session_state.user_id = auth_result["user_id"]
                st.session_state.email = email
                st.session_state.session_id = None
                st.session_state.chat_history = []
            except Exception as e:  # noqa: BLE001
                st.error(f"Authentication failed: {e}")


def render_sidebar() -> None:
    with st.sidebar:
        st.title("Remiro AI")
        st.caption("Career Co‑Pilot")

        if st.session_state.user_id:
            st.caption(f"Signed in as {st.session_state.email}")

            if st.button("Log out", use_container_width=True):
                logout()
                st.experimental_rerun()

            st.markdown("---")

            if st.button("New chat", use_container_width=True):
                st.session_state.session_id = None
                st.session_state.chat_history = []

            try:
                sessions = list_user_sessions(st.session_state.user_id)
            except Exception:
                # If loading sessions fails (for example, if the tables are not
                # yet initialized in Supabase), do not interrupt the user with
                # a red error popup. Instead, show a gentle hint and proceed.
                st.caption("(We couldn't load past conversations yet. You can still start a new chat.)")
                sessions = []

            if sessions:
                st.subheader("Your conversations")

                # Build a mapping from label to session id for a cleaner selection UI.
                labels = []
                id_by_label = {}
                for sess in sessions:
                    title = sess.get("title") or "Untitled session"
                    sid = sess.get("id")
                    label = f"{title}"
                    labels.append(label)
                    id_by_label[label] = sid

                current_id = st.session_state.session_id
                current_label = None
                for lbl, sid in id_by_label.items():
                    if sid == current_id:
                        current_label = lbl
                        break

                selected_label = st.radio(
                    "Select a session",
                    labels,
                    index=labels.index(current_label) if current_label in labels else 0,
                    key="session_selector",
                )

                selected_id = id_by_label.get(selected_label)
                if selected_id and selected_id != st.session_state.session_id:
                    st.session_state.session_id = selected_id
                    # Load past messages for this session into the local history
                    try:
                        msgs = get_session_messages(selected_id)
                        st.session_state.chat_history = [
                            m for m in msgs if m["role"] in ("user", "assistant")
                        ]
                    except Exception as load_err:  # noqa: BLE001
                        st.error(f"Could not load messages for this session: {load_err}")
            else:
                st.caption("No sessions yet. Start a new chat!")


def render_chat() -> None:
    st.title("Remiro AI – Career Conversation")
    st.caption(
        "Ask about your career, skills, learning path, workplace dynamics, and long‑term direction. "
        "Remiro AI is not designed for real‑time market data, medical, legal, or general trivia questions."
    )

    # Display existing chat history
    for msg in st.session_state.chat_history:
        role = msg.get("role", "assistant")
        content = msg.get("content", "")
        with st.chat_message(role):
            st.markdown(content)

    user_input = st.chat_input("Ask about your career, goals, or next moves...")

    if user_input:
        # Optimistically show the user message
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Get assistant reply via the orchestrated backend, then stream it
        # into the UI word-by-word for a more responsive feel.
        with st.chat_message("assistant"):
            placeholder = st.empty()
            with st.spinner("Thinking with all specialist agents..."):
                try:
                    result = run_session(
                        user_id=st.session_state.user_id,
                        user_input=user_input,
                        session_id=st.session_state.session_id,
                    )
                    reply = result.get("reply", "")
                    st.session_state.session_id = result.get("session_id", st.session_state.session_id)
                except Exception as e:  # noqa: BLE001
                    reply = f"There was an error processing your request: {e}"

            # Simple front-end streaming: progressively reveal the reply
            # once it is available from the backend.
            streamed = ""
            for word in reply.split(" "):
                streamed += (word + " ")
                placeholder.markdown(streamed)
                # Small delay to make the streaming perceptible but fast.
                time.sleep(0.01)

        st.session_state.chat_history.append({"role": "assistant", "content": reply})


def main() -> None:
    init_state()

    if not st.session_state.user_id:
        render_auth_screen()
    else:
        render_sidebar()
        render_chat()


if __name__ == "__main__":
    main()
