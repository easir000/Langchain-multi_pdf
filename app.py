import os
import pickle
import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.memory import ConversationBufferMemory
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.callbacks import get_openai_callback
from PyPDF2 import PdfReader
from streamlit_extras.colored_header import colored_header
from dotenv import load_dotenv

VALID_CREDENTIALS = {"chalkrai": "ab12cd34", "rapidai": "123456"}

# load the environment variables
load_dotenv()
 

# Create the vectorstore directory if it doesn't exist
VECTORSTORE_DIR = "vectorstore"
os.makedirs(VECTORSTORE_DIR, exist_ok=True)


# set the page title and icon
st.set_page_config(page_title="Chalkrai Powered Document Chat", page_icon="🌍")


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def save_vectorstore(vectorstore, filename):
    with open(os.path.join(VECTORSTORE_DIR, filename), "wb") as file:
        pickle.dump(vectorstore, file)


def get_vectorstore(text_chunks, embeddings_selection):
    if embeddings_selection == "OpenAI":
        embeddings = OpenAIEmbeddings()
    else:
        embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")

    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def load_vectorstore(filename):
    with open(os.path.join(VECTORSTORE_DIR, filename), "rb") as file:
        vectorstore = pickle.load(file)
    return vectorstore

def authenticate():
    if "authenticated" not in st.session_state or not st.session_state.authenticated:
        st.subheader("Authentication")

        # Check if the user wants to register
        is_registering = st.checkbox("Register")

        if is_registering:
            new_username = st.text_input("New Username:")
            new_password = st.text_input("New Password:", type="password")
            confirm_password = st.text_input("Confirm Password:", type="password")

            if st.button("Register"):
                # Check if passwords match
                if new_password == confirm_password:
                    # Check if username already exists
                    if new_username not in st.session_state.users:
                        # Update session_state.users with the new user
                        st.session_state.users[new_username] = {"password": new_password}
                        st.success("Registration successful! You can now log in.")
                    else:
                        st.error("Username already exists. Please choose a different username.")
                else:
                    st.error("Passwords do not match. Please try again.")
        else:
            # Log in section remains the same
            username = st.text_input("Username:")
            password = st.text_input("Password:", type="password")

            if st.button("Login"):
                user_data = st.session_state.users.get(username)
                if user_data and user_data["password"] == password:
                    st.success("Authentication successful!")
                    st.session_state.authenticated = True
                    st.experimental_set_query_params(authenticated=True)
                else:
                    st.error("Authentication failed. Please try again.")
                    st.write("Debugging Info:")
                    st.write("Entered Username:", username)
                    st.write("Entered Password:", password)

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )

if "users" not in st.session_state:
    st.session_state.users = {}

def chain_setup(vectorstore, model_name="OpenAI"):
    template = """{question}
    """

    if model_name == "OpenAI":
        # initialize the LLM with api key
        llm = ChatOpenAI()

    elif model_name == "Falcon":
        prompt = PromptTemplate(template=template, input_variables=["question"])
        llm = HuggingFaceHub(
            repo_id="google/flan-t5-xxl",
            model_kwargs={"temperature": 0.5, "max_length": 512},
        )

    elif model_name == "OpenAssistant":
        template = """<|prompter|>{question}<|endoftext|>
        <|assistant|>"""
        prompt = PromptTemplate(template=template, input_variables=["question"])
        llm = HuggingFaceHub(
            repo_id="OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5",
            model_kwargs={"max_new_tokens": 1200},
        )

    else:
        raise ValueError(
            "Invalid model_name. Choose from 'OpenAI', 'Falcon', 'OpenAssistant'."
        )

    # Use the memory from the session state
    memory = st.session_state.memory

    # Create the LLM chain
    if model_name == "OpenAI":
        llm_chain = ConversationalRetrievalChain.from_llm(
            llm=llm, retriever=vectorstore.as_retriever(), memory=memory
        )
    else:
        llm_chain = LLMChain(llm=llm, prompt=prompt, memory=memory)
    return llm_chain


# generate response
def generate_response(question, llm_chain, llm_model_name):
    cost = 0.0

    # Get the response from the LLM
    if llm_model_name == "OpenAI":
        with get_openai_callback() as cb:
            response = llm_chain.run(question)

            if cb is not None:
                cost = round(cb.total_cost, 5)
    else:
        response = llm_chain.run(question)

    st.session_state.chat_history.append((question, response))
    return response, cost


# This function will create a new message in the chat
def render_message(sender: str, avatar_url: str, message: str, cost: float = 0.0):
    # Create a container for the message
    with st.container():
        # Create a column for the avatar
        col1, col2 = st.columns([1, 9])

        # Display the avatar
        with col1:
            st.image(avatar_url, width=50)
            if sender == "AI":
                st.write(cost)

        # Display the message
        with col2:
            # st.markdown(f"**{sender}**")
            if sender == "User":
                st.text_area("", value=message, height=50, max_chars=None, key=None)
            else:
                st.info(message)


# This function will get current vectorstore
def get_current_vectorstore():
    if st.session_state.vectorstore_selection == "Create New":
        st.warning(
            "Please upload PDFs to create a new vectorstore or select an existing vectorstore."
        )
        return None
    else:
        vectorstore = load_vectorstore(st.session_state.vectorstore_selection)
        return vectorstore


def main():
    llm_selection = None  # Initialize llm_selection at the beginning

    # Only show authentication if not authenticated
    if "authenticated" not in st.session_state or not st.session_state.authenticated:
        authenticate()
        return

    # Sidebar contents
    with st.sidebar:
        st.subheader(":gear: Options")


        
    

        # Let the user choose the models
        llm_selection = st.selectbox(
            "🚀 Choose a Large Language Model",
            options=["OpenAI", "Falcon", "OpenAssistant"],
        )
        embeddings_selection = st.selectbox(
            "🌞 Choose an Embeddings Model",
            options=["OpenAI", "HuggingFaceInstruct"],
        )

        # Let the user choose a vector store file, or create a new one
        vectorstore_files = ["Create New"] + os.listdir(VECTORSTORE_DIR)
        st.session_state.vectorstore_selection = st.selectbox(
            "📥 Choose a Vector Store File", options=vectorstore_files
        )

        # Handle file upload
        pdf_docs = st.file_uploader(
            "📑Upload your PDFs here and click on 'Process'",
            type=["pdf", "txt"],
            accept_multiple_files=True,
        )
        if st.button("Process"):
            with st.spinner("Processing"):
                # Get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # Get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # Create or load vector store
                if (
                    st.session_state.vectorstore_selection == "Create New"
                    or not os.path.exists(
                        os.path.join(
                            VECTORSTORE_DIR, st.session_state.vectorstore_selection
                        )
                    )
                ):
                    vectorstore = get_vectorstore(text_chunks, embeddings_selection)
                    vectorstore_filename = f"{llm_selection}_{embeddings_selection}_{len(os.listdir(VECTORSTORE_DIR))}.pkl"
                    save_vectorstore(vectorstore, vectorstore_filename)
                    st.session_state.vectorstore_selection = vectorstore_filename  # update the current selection to the new file
                else:
                    vectorstore = load_vectorstore(
                        st.session_state.vectorstore_selection
                    )
                    # vectorstore.update(text_chunks)

                # Get the current vectorstore
                current_vectorstore = get_current_vectorstore()

                # Create conversation chain
                if current_vectorstore is not None:
                    st.session_state.conversation = chain_setup(
                        vectorstore, llm_selection
                    )

        if st.button("Clear Chat"):
            st.session_state.user = []
            st.session_state.generated = []
            st.session_state.cost = []



        # Logout button
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.experimental_set_query_params(authenticated=False)

    st.header("Your Personal Assistant 💼")

    # Generate empty lists for generated and user.
    # Assistant Response
    if "generated" not in st.session_state:
        st.session_state["generated"] = ["I'm Assistant, \n \n How may I help you?"]

    # user question
    if "user" not in st.session_state:
        st.session_state["user"] = ["Hi!"]

    # chat history
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Generate empty list for cost history
    if "cost" not in st.session_state:
        st.session_state["cost"] = [0.0]

    # Layout of input/response containers
    response_container = st.container()
    colored_header(label="", description="", color_name="blue-30")
    input_container = st.container()

    # get user input
    def get_text():
        input_text = st.text_input("You: ", "", key="input")
        return input_text

    # Applying the user input box
    with input_container:
        user_input = get_text()

    # load LLM
    if user_input:
        current_vectorstore = get_current_vectorstore()
        if current_vectorstore is None:
            return
        llm_chain = chain_setup(current_vectorstore, llm_selection)

    # main loop
    with response_container:
        if user_input:
            response, cost = generate_response(user_input, llm_chain, llm_selection)
            st.session_state.user.append(user_input)
            st.session_state.generated.append(response)
            st.session_state.cost.append(cost)

        if st.session_state["generated"]:
            for i in range(len(st.session_state["generated"])):
                render_message(
                    "user",
                    "https://i.ibb.co/R6MK14P/pngtree-man.png",
                    st.session_state["user"][i],
                )

                render_message(
                    "AI",
                    "https://i.ibb.co/k6TrSVX/chalkr-ai-2.jpg",
                    st.session_state["generated"][i],
                    st.session_state["cost"][i],
                )


if __name__ == "__main__":
    main()
