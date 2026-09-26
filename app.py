import streamlit as st

st.title("AI CHATBOT")

st.text("Welcome to AI Chatbot")

user_input = st.text_input(
    "Enter Prompt",
    placeholder="User Input"
)

if st.button("Send"):

    if user_input:
        st.success("Entered successfully")
        st.write(user_input)

    else:
        st.error("Please enter a correct prompt")