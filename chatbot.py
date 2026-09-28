import streamlit as st
import ollama

st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖"
)

st.title("My AI Chatbot")
st.caption("Powered by Ollama + Streamlit")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
prompt = st.chat_input("Type your message...")

if prompt:
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)

    # Save user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    # Generate response from Ollama
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = ollama.chat(
                    model="llama3.2",
                    messages=st.session_state.messages
                )

                answer = response["message"]["content"]
                st.markdown(answer)

            except Exception as e:
                answer = f"Error: {e}"
                st.error(answer)

    # Save assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })