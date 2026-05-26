import streamlit as st

from pharmacy_ai import ask_ai

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="AI Agent Farmasi",
    page_icon="💊",
    layout="centered"
)

# =========================
# TITLE
# =========================

st.title("💊 AI Agent Farmasi")
st.write("Tanyakan informasi obat kepada AI.")

# =========================
# CHAT HISTORY
# =========================

if "messages" not in st.session_state:
    st.session_state.messages = []

# tampilkan histori chat
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# =========================
# INPUT USER
# =========================

prompt = st.chat_input(
    "Contoh: Obat untuk luka bakar ringan"
)

if prompt:

    # tampilkan user message
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    # =========================
    # AI RESPONSE
    # =========================

    with st.chat_message("assistant"):

        with st.spinner("AI sedang berpikir..."):

            response = ask_ai(prompt)

            st.markdown(response)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })