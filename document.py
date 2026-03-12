from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from google import genai

import chromadb
from uuid import uuid4
load_dotenv()


gemini_client= genai.Client()

# Document Loader
file_path="/Users/stackular/container/AI/1hr/document_reader/Tutorial_EDIT.pdf"
loader=PyPDFLoader(file_path)
document=loader.load()



# Text Splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=400)
texts = text_splitter.split_documents(document)
texts_content = [t.page_content for t in texts]


# Embeddings
result = gemini_client.models.embed_content(
    model="gemini-embedding-001",
    contents=texts_content,
    
)

embeddings=[e.values for e in result.embeddings]

client =chromadb.Client()

colection= client.create_collection('my_agent_collection')

colection.add(
    ids=[str(uuid4()) for _ in texts],
    documents=texts_content,
    embeddings=embeddings,
    
)

query_texts='What is The String format() Method'

query_embeding=gemini_client.models.embed_content(
    model="gemini-embedding-001",
    contents=[query_texts],
)

query_vector= query_embeding.embeddings[0].values

results= colection.query(
    query_embeddings=[query_vector],
    n_results=1,
 )

print('results-------------------------------->',results['documents'][0][0])