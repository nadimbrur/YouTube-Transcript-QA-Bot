import os
import streamlit as st
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
import re
# Load API token
load_dotenv()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

# --- Streamlit UI ---
import re
st.title("📺 YouTube Transcript QA Bot")
url = st.text_input("Enter YouTube Video Link or ID", "Gfr50f6ZBvo")

match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", url)
video_id = match.group(1) if match else (url if len(url) == 11 else "")



if video_id:
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        transcript = " ".join(chunk["text"] for chunk in transcript_list)
        st.success("Transcript fetched successfully!")
    except TranscriptsDisabled:
        st.error("No captions available for this video.")
        st.stop()
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

    # Split transcript into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])
    cnt = 1
    if cnt:
      # Embed and store in vector store
      with st.spinner("Embedding transcript..."):
          embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
          vector_store = FAISS.from_documents(chunks, embeddings)
          retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
          st.success("Transcript embedded and ready for querying!")
          cnt = 0 
    # Show query input
    question = st.text_input("Ask a question based on the transcript:")

    if st.button("Get Answer") and question.strip():
        # Load LLM
        with st.spinner("Calling Hugging Face model..."):
            llm = HuggingFaceEndpoint(
                repo_id='HuggingFaceH4/zephyr-7b-beta',
                task='text-generation'
            )

            # Prompt
            prompt = PromptTemplate(
                template="""
                You are a helpful assistant.
                Use only the provided transcript context.
                If the context is insufficient, just say you don't know.

                Context:
                {context}

                Question:
                {question}
                """,
                input_variables=["context", "question"]
            )

            # Chain logic
            def format_docs(retrieved_docs):
                return "\n\n".join(doc.page_content for doc in retrieved_docs)

            parallel_chain = RunnableParallel({
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough()
            })

            main_chain = parallel_chain | prompt | llm | StrOutputParser()

            result = main_chain.invoke(question)
            st.success("Answer:")
            st.write(result)

