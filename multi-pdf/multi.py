import streamlit as st
from dotenv import load_dotenv
from utils import *
from htmlTemplates import bot_template, user_template, css

def handle_user_input(question):

    response = st.session_state.conversation({'question':question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = "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"
    st.set_page_config(page_title='Chat with Your own PDFs', page_icon=':books:')

    st.write(css, unsafe_allow_html=True)
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    st.header('Chat with Your own PDFs :books:')
    question = st.text_input("Ask anything to your PDF: ")

    if question:
        handle_user_input(question)
    

    with st.sidebar:
        st.subheader("Upload your Documents Here: ")
        pdf_files = st.file_uploader("Choose your PDF Files and Press OK", type=['pdf'], accept_multiple_files=True)

        if st.button("OK"):
            with st.spinner("Processing your PDFs..."):

                # Get PDF Text
                raw_text = get_pdf_text(pdf_files)

                # Get Text Chunks
                text_chunks = get_chunk_text(raw_text)
                

                # Create Vector Store
                
                # vector_store = get_vector_store(text_chunks, pdf_files.name[:-4])
                vector_store = get_vector_store(text_chunks)
                st.write("DONE")

                # Create conversation chain

                st.session_state.conversation =  get_conversation_chain(vector_store)


if __name__ == '__main__':
    main()