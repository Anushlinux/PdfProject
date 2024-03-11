import streamlit as st
import pypdfium2 as pdfium
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceInstructEmbeddings 
from langchain.llms import HuggingFaceHub
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
import os


def get_pdf_text(pdf_files):
    text = " "
    for pdf in pdf_files:
        pdf_reader = pdfium.PdfDocument(pdf)
        for i in range(len(pdf_reader)):
            page = pdf_reader.get_page(i)
            textpage = page.get_textpage()
            text += textpage.get_text_range()
            
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(separator = "\n", chunk_size = 5000, chunk_overlap = 500, length_function = len)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(embedding = embeddings, texts = text_chunks)
    return vector_store

def get_conversation_chain(vector_store):
    llms = ChatGoogleGenerativeAI(model = 'gemini-pro')
    memory = ConversationBufferMemory(memory_key="AIzaffa-Ouf6EEsdsfvOVdH44ujFEFBaqpCUzI1vPw", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llms,
        memory = memory,
        retriever= vector_store.as_retriever(),
    )
    return conversation_chain

def save_question_and_clear_prompt(ss):
    ss.user_question = ss.prompt_bar
    ss.prompt_bar = ""     


def handle_user_input(user_question):
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    response = st.session_state.conversation({
        'question': user_question,
        'chat_history': st.session_state.chat_history
    })
    
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():

    load_dotenv()
    print(os.getenv("GOOGLE_API_KEY"))
    st.set_page_config(page_title="Chat with multiple pdfs", page_icon=":smiley:", layout="wide", initial_sidebar_state="expanded")    
    
    st.write(css, unsafe_allow_html=True)

        
    st.header("Chat with multiple pdfs")
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
        
    user_question = st.text_input("Ask any Question")
    
    # if st.session_state.conversation is not None and user_question:
    #     handle_user_input(user_question)
            
    if  user_question:
        handle_user_input(user_question)
        
    st.write(user_template.replace("{{MSG}}", "Sup robot?"), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", "Sup Human"), unsafe_allow_html=True)
    
    with st.sidebar:
        st.subheader("Your pdfs")
        
        pdf_files = st.file_uploader("Upload pdfs and click on: 'process'", accept_multiple_files=True, type = "pdf")
        
          
        if st.button("Process") and pdf_files:
            with st.spinner("Processing"):
                
                # get text from pdfs
                raw_data = get_pdf_text(pdf_files)

                # get chunks of text
                text_chunks = get_text_chunks(raw_data)
                
                # create vector store
                vector_store = get_vector_store(text_chunks)
                
                #convo parts
                st.session_state.conversation = get_conversation_chain(vector_store)
                



if __name__ == "__main__":
    main()    

# import streamlit as st
# from dotenv import load_dotenv
# import pypdfium2 as pdfium  
# from langchain.text_splitter import CharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.vectorstores import FAISS
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain
# from htmlTemplates import css, bot_template, user_template



# def get_pdf_text(pdf_docs):
#     text = " "
#     for pdf in pdf_docs:
#         pdf_reader = pdfium.PdfDocument(pdf)
#         for i in range(len(pdf_reader)):
#             page = pdf_reader.get_page(i)
#             textpage = page.get_textpage()
#             text += textpage.get_text_range() + "\n"
#     return text


# def get_text_chunks(text):
#     text_splitter = CharacterTextSplitter(separator="\n", chunk_size=5000, chunk_overlap=500, length_function=len)
#     chunks = text_splitter.split_text(text)
#     return chunks


# def get_vectorstore(text_chunks):
#     embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001", google_api_key="AIzaSyA-Ouf6EEGJOVdH44ujFEFBwTpCUzI1vPw")
#     vector_store = FAISS.from_texts(embedding = embeddings, texts = text_chunks)
#     return vector_store

# def get_conversation_chain(vector_store):
#     llms = ChatGoogleGenerativeAI(model = "gemini-pro")
#     memory = ConversationBufferMemory(memory_key="gemini-pro", return_messages=True)
#     conversation_chain = ConversationalRetrievalChain.from_llm(
#         llm = llms,
#         memory = memory,
#         retriever= vector_store.as_retriever(),
#     )
#     return conversation_chain


# def save_question_and_clear_prompt(ss):
#     ss.user_question = ss.prompt_bar
#     ss.prompt_bar = " "  # clearing the prompt bar after clicking enter to prevent automatic re-submissions


# def write_chat(msgs):  # Write the Q&A in a pretty chat format
#     for i, msg in enumerate(msgs):
#         if i % 2 == 0:  # it's a question
#             st.write(user_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)
#         else:  # it's an answer
#             st.write(bot_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)


# def main():
#     load_dotenv()  # loads api keys
#     ss = st.session_state  # https://docs.streamlit.io/library/api-reference/session-state

#     # Page design
#     st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
#     st.write(css, unsafe_allow_html=True)
#     st.header("Chat with multiple PDFs :books:")

#     # Initializing session state variables
#     if "conversation_chain" not in ss:
#         ss.conversation_chain = None  # the main variable storing the llm, retriever and memory
#     if "prompt_bar" not in ss:
#         ss.prompt_bar = ""
#     if "user_question" not in ss:
#         ss.user_question = ""
#     if "docs_are_processed" not in ss:
#         ss.docs_are_processed = False

#     with st.sidebar:
#         st.subheader("Your documents")
#         pdf_docs = st.file_uploader("Upload your PDFs here and click 'Process'", accept_multiple_files=True, type="pdf")
#         if st.button("Process") and pdf_docs:
#             with st.spinner("Processing"):
#                 raw_text = get_pdf_text(pdf_docs)  # get pdf text
#                 text_chunks = get_text_chunks(raw_text)  # get the text chunks
#                 vectorstore = get_vectorstore(text_chunks)
#                 ss.conversation_chain = get_conversation_chain(vectorstore)  # create conversation chain
#                 ss.docs_are_processed = True
#         if ss.docs_are_processed:
#             st.text('Documents processed')

#     st.text_input("Ask a question here:", key='prompt_bar', on_change=save_question_and_clear_prompt(ss))

#     if ss.user_question:
#         ss.conversation_chain({'question': ss.user_question})  # This is what gets the response from the LLM!
#         if hasattr(ss.conversation_chain.memory, 'chat_memory'):
#             chat = ss.conversation_chain.memory.chat_memory.messages
#             write_chat(chat)

#     if hasattr(ss.conversation_chain, 'memory'):  # There is memory if the documents have been processed
#         if hasattr(ss.conversation_chain.memory, 'chat_memory'):  # There is chat_memory if questions have been asked
#             if st.button("Forget conversation"):  # adding a button
#                 ss.conversation_chain.memory.chat_memory.clear()  # clears the ConversationBufferMemory

#     # st.write(ss)  # use this when debugging for visualizing the session_state variables


# if __name__ == '__main__':
#     main()