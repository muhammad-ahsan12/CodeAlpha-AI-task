from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
import streamlit as st

# Set up the environment
os.environ['GOOGLE_API_KEY'] = 'AIzaSyA0S7F21ExbBnR06YXkEi7aj94nWP5kJho'

# Define the prompt template
template = """You are a helpful assistant,
translate the following {speech} into {Language}."""
prompt = PromptTemplate(
    input_variables=["speech", "Language"],
    template=template
)

# Set up the Streamlit app
st.set_page_config(page_title="Language Translator", page_icon="🌐")
st.title("🌐 Language Translator")
st.write("Welcome! Translate your speech into the language of your choice. 🌍")

# Input speech and language
speech = st.text_area("✏️ Input your speech")
languages = [
    "Urdu", "Spanish", "French", "German", "Chinese", "Japanese", "Korean", "Italian",
    "Portuguese", "Russian", "Arabic", "Hindi", "Bengali", "Greek", "Dutch", "Swedish",
    "Turkish", "Thai", "Vietnamese", "Polish", "Hebrew", "Czech", "Danish", "Finnish",
    "Hungarian", "Indonesian", "Malay", "Norwegian", "Romanian", "Slovak"
]
language = st.selectbox("🌍 Select Language", languages)

# Spinner for loading state
if st.button("Translate"):
    with st.spinner("Translating... 🔄"):
        # Initialize the model
        llm = ChatGoogleGenerativeAI(model='gemini-1.5-pro')
        LM = LLMChain(llm=llm, prompt=prompt)
        
        # Run the translation
        translated_text = LM.run({"speech": speech, "Language": language})
        st.success("Translation completed! ✅")
        st.write(translated_text)

# Footer
st.write("Made with ❤️ by [Muhammad Ahsan]")