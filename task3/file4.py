
# To run this code you need to install:
# pip install google-genai python-dotenv

import os
import streamlit as st
from google import genai
from google.genai import types
from dotenv import load_dotenv

def main():
    st.set_page_config(page_title="Data Science Chatbot")

    st.title("Data Science Assistant")
    st.caption("Powered by Gemini 2.5 Pro")

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        st.error("API key not found in .env file!")
        return

    client = genai.Client(api_key=api_key)
    model = "gemini-2.5-pro"

    # Maintain chat history in Streamlit session
    if "history" not in st.session_state:
        st.session_state.history = []

    # User input
    user_input = st.chat_input("Ask me anything about data science...")

    if user_input:
        # Add user message to chat history
        st.session_state.history.append({"role": "user", "content": user_input})

        # Create chat session each time
        response = client.models.generate_content(
            model=model,
            contents=[types.Content(
                role="user",
                parts=[types.Part.from_text(text=user_input)],
            )],
            config=types.GenerateContentConfig(
                temperature=2,
                tools=[types.Tool(googleSearch=types.GoogleSearch())],
                system_instruction=[
                    types.Part.from_text(
                        text="You are a data science engineer. Always converse in a formal tone and relate all conversations to data science."
                    )
                ],
            ),
        )

        bot_reply = response.text
        st.session_state.history.append({"role": "assistant", "content": bot_reply})

    # Display chat history
    for chat in st.session_state.history:
        if chat["role"] == "user":
            with st.chat_message("user"):
                st.markdown(chat["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(chat["content"])

if __name__ == "__main__":
    main()
