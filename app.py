
import streamlit as st
import phase5_rag_agent as rag
from pypdf import PdfReader
from docx import Document
from io import BytesIO

st.set_page_config(
    page_title="StudyBuddy AI",
    page_icon="📚",
    layout="wide"
)


# ----------------------------------
# FILE READING FUNCTIONS
# ----------------------------------

def read_uploaded_file(uploaded_file):

    file_name = uploaded_file.name.lower()

    if file_name.endswith(".txt"):
        return uploaded_file.getvalue().decode("utf-8", errors="replace")

    elif file_name.endswith(".pdf"):
        reader = PdfReader(BytesIO(uploaded_file.getvalue()))

        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")

        return "\n".join(pages)

    elif file_name.endswith(".docx"):
        document = Document(BytesIO(uploaded_file.getvalue()))

        paragraphs = [
            paragraph.text
            for paragraph in document.paragraphs
            if paragraph.text.strip()
        ]

        return "\n".join(paragraphs)

    return ""


# ----------------------------------
# LOAD DEFAULT NOTES
# ----------------------------------

@st.cache_data
def get_chunks():
    return rag.load_chunks()


if "default_chunks" not in st.session_state:
    st.session_state.default_chunks = get_chunks()

if "uploaded_chunks" not in st.session_state:
    st.session_state.uploaded_chunks = []

if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = ""


# ----------------------------------
# CHAT HISTORIES
# ----------------------------------

if "all_messages" not in st.session_state:
    st.session_state.all_messages = {
        "Study Notes": [],
        "General AI": []
    }

if "all_histories" not in st.session_state:
    st.session_state.all_histories = {
        "Study Notes": [],
        "General AI": []
    }


# ----------------------------------
# GENERAL AI
# ----------------------------------

def general_answer(question, history):

    contents = []

    for message in history[-10:]:
        role = "model" if message["role"] == "assistant" else "user"

        contents.append({
            "role": role,
            "parts": [{"text": message["content"]}]
        })

    contents.append({
        "role": "user",
        "parts": [{"text": question}]
    })

    response = rag.client.models.generate_content(
        model=rag.MODEL,
        contents=contents,
        config=rag.types.GenerateContentConfig(
            system_instruction=(
                "You are StudyBuddy, a friendly and helpful AI study assistant. "
                "Answer greetings naturally and explain educational topics "
                "in simple, clear language. Give examples when helpful."
            ),
            temperature=0.4
        )
    )

    return (response.text or "Sorry, I could not generate an answer.").strip()


# ----------------------------------
# SIDEBAR
# ----------------------------------

with st.sidebar:

    st.title("📚 StudyBuddy AI")
    st.caption("Your Personal Study Assistant")
    st.divider()

    st.subheader("Choose AI Mode")

    mode = st.radio(
        "Select mode:",
        ["General AI", "Study Notes"]
    )

    st.divider()

    if mode == "Study Notes":

        st.subheader("📂 Upload Study Notes")

        uploaded_file = st.file_uploader(
            "Upload PDF, TXT or DOCX",
            type=["pdf", "txt", "docx"],
            key="study_file"
        )

        if uploaded_file is not None:

            if st.button("📥 Process Uploaded File", use_container_width=True):

                with st.spinner("Reading and processing your file..."):

                    try:
                        text = read_uploaded_file(uploaded_file)

                        if not text.strip():
                            st.error(
                                "No readable text found. "
                                "If this is a scanned PDF, OCR is required."
                            )

                        else:
                            st.session_state.uploaded_chunks = rag.chunk_text(text)

                            st.session_state.uploaded_file_name = uploaded_file.name

                            st.session_state.all_messages["Study Notes"] = []
                            st.session_state.all_histories["Study Notes"] = []

                            st.success("File processed successfully!")

                    except Exception as e:
                        st.error(f"File reading error: {e}")

        if st.session_state.uploaded_chunks:
            st.success(
                f"Active file: {st.session_state.uploaded_file_name}"
            )

            st.caption(
                f"{len(st.session_state.uploaded_chunks)} chunks ready."
            )

            if st.button("Remove Uploaded File", use_container_width=True):
                st.session_state.uploaded_chunks = []
                st.session_state.uploaded_file_name = ""
                st.session_state.all_messages["Study Notes"] = []
                st.session_state.all_histories["Study Notes"] = []
                st.rerun()

        st.divider()

        st.info(
            "Study Notes Mode answers using your uploaded file "
            "or default my_notes.txt notes."
        )

    else:
        st.info(
            "Ask anything! Definitions, concepts, greetings and more."
        )

    st.divider()

    if st.button("🗑️ Clear Current Chat", use_container_width=True):

        st.session_state.all_messages[mode] = []
        st.session_state.all_histories[mode] = []
        st.rerun()

    st.caption("Powered by Google Gemini AI")


# ----------------------------------
# SELECT ACTIVE NOTES
# ----------------------------------

if st.session_state.uploaded_chunks:
    chunks = st.session_state.uploaded_chunks
else:
    chunks = st.session_state.default_chunks


# ----------------------------------
# MAIN PAGE
# ----------------------------------

messages = st.session_state.all_messages[mode]
history = st.session_state.all_histories[mode]

st.title("📚 StudyBuddy AI")

if mode == "General AI":

    st.subheader("Your General AI Assistant")
    st.write("Ask me anything about your studies or other topics!")

else:

    st.subheader("Your Study Notes Assistant")

    if st.session_state.uploaded_chunks:
        st.write(
            f"Currently answering from: "
            f"**{st.session_state.uploaded_file_name}**"
        )
    else:
        st.write("Currently answering from your my_notes.txt file.")

    st.caption(f"Available chunks: {len(chunks)}")

st.divider()


# ----------------------------------
# WELCOME MESSAGE
# ----------------------------------

if not messages:

    if mode == "General AI":
        st.info("👋 Hello! I'm StudyBuddy. How can I help you today?")

    else:
        st.info(
            "👋 Upload your lecture notes or ask questions "
            "about your existing notes."
        )


# ----------------------------------
# DISPLAY CHAT
# ----------------------------------

for message in messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message.get("chunks"):

            with st.expander("📖 View Retrieved Notes"):

                for number, score, text in message["chunks"]:

                    st.markdown(
                        f"**Chunk {number} | Score: {score:.0f}**"
                    )

                    st.write(text)


# ----------------------------------
# CHAT INPUT
# ----------------------------------

question = st.chat_input("Type your question here...")

if question:

    messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("StudyBuddy is thinking..."):

            try:

                hits = []

                if mode == "General AI":

                    reply = general_answer(question, history)

                else:

                    hits = rag.retrieve(question, chunks)

                    if not hits:

                        reply = "I couldn't find that in your notes."

                    else:

                        prompt = rag.build_augmented_prompt(question, hits)

                        context_history = history[-rag.MAX_HISTORY:]

                        context_history = context_history + [{
                            "role": "user",
                            "content": prompt
                        }]

                        reply = rag.chat(context_history)

                st.markdown(reply)

                if mode == "Study Notes" and hits:

                    with st.expander("📖 View Retrieved Notes"):

                        for number, score, text in hits:

                            st.markdown(
                                f"**Chunk {number} | Score: {score:.0f}**"
                            )

                            st.write(text)

                history.append({
                    "role": "user",
                    "content": question
                })

                history.append({
                    "role": "assistant",
                    "content": reply
                })

                messages.append({
                    "role": "assistant",
                    "content": reply,
                    "chunks": hits
                })

            except Exception as e:

                st.error(f"Something went wrong: {e}")
