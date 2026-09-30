import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb
import ollama


# -----------------------------
# Streamlit Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Mini RAG Q&A",
    page_icon="📚"
)

st.title("📚 Mini RAG Q&A")
st.write(
    "Paste a document, store it in ChromaDB, "
    "and ask questions about it."
)


# -----------------------------
# Load Embedding Model
# -----------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


embedding_model = load_embedding_model()


# -----------------------------
# Create ChromaDB Collection
# -----------------------------
client = chromadb.Client()

collection = client.get_or_create_collection(
    name="documents"
)


# -----------------------------
# Document Input
# -----------------------------
document = st.text_area(
    "📃 Paste your document here",
    height=250,
    placeholder="Paste your notes, article, syllabus, etc."
)


# -----------------------------
# Add Document
# -----------------------------
if st.button("➕ Add Document"):

    if not document.strip():
        st.warning("Please enter some text.")

    else:
        # Split document into chunks
        chunks = [
            document[i:i + 500]
            for i in range(0, len(document), 500)
        ]

        # Create embeddings
        embeddings = embedding_model.encode(chunks)

        # Create unique IDs
        ids = [
            f"chunk_{i}"
            for i in range(len(chunks))
        ]

        # Store in ChromaDB
        collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings.tolist()
        )

        st.success(
            f"Added {len(chunks)} chunk(s) to ChromaDB."
        )


# -----------------------------
# Question Input
# -----------------------------
question = st.text_input(
    "❓ Ask a question about your document"
)


# -----------------------------
# Ask AI
# -----------------------------
if st.button("🔍 Ask AI"):

    if not question.strip():

        st.warning("Please enter a question.")

    elif collection.count() == 0:

        st.warning("Please add a document first.")

    else:

        # Convert question into embedding
        question_embedding = embedding_model.encode(
            [question]
        )[0]

        # Search ChromaDB
        results = collection.query(
            query_embeddings=[
                question_embedding.tolist()
            ],
            n_results=min(3, collection.count())
        )

        # Get retrieved documents
        retrieved_chunks = results["documents"][0]

        # Combine retrieved chunks
        context = "\n\n".join(
            retrieved_chunks
        )

        # -----------------------------
        # Create RAG Prompt
        # -----------------------------
        prompt = f"""
You are a helpful AI assistant.

Answer the question ONLY using the context below.

Context:
{context}

Question:
{question}

If the answer is not present in the context,
say:

"I don't know based on the provided document."
"""

        # -----------------------------
        # Call Ollama
        # -----------------------------
        try:

            response = ollama.chat(
                model="llama3.2",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            # Display answer
            st.subheader("🤖 Answer")

            st.write(
                response["message"]["content"]
            )

            # Display retrieved context
            with st.expander("📄 Retrieved Context"):

                for i, chunk in enumerate(
                    retrieved_chunks
                ):

                    st.write(
                        f"**Chunk {i + 1}:**"
                    )

                    st.write(chunk)

        except Exception as e:

            st.error(
                "Could not connect to Ollama."
            )

            st.info(
                "Make sure Ollama is running "
                "and that the llama3.2 model is installed."
            )

            st.code(str(e))


# -----------------------------
# Footer
# -----------------------------
st.divider()

st.caption(
    "Python + Sentence Transformers + "
    "ChromaDB + Ollama + Streamlit"
)