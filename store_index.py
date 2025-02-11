from src.helper import load_pdf_file, split_text ,download_hugging_face_embeddings
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os



load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

#Data Loading
extracted_data = load_pdf_file(data="Data/")
text_chunks = split_text(extracted_data)
embeddings = download_hugging_face_embeddings()


#Pinecone Initalization

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "curenow"

pc.create_index(
    name=index_name,
    dimension=384, # Replace with your model dimensions
    metric="cosine", # Replace with your model metric
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    ) 
)

from langchain.vectorstores import Pinecone
#Embed each chunk and upsert the embeddings into Your Pinecone index.
docresearch = Pinecone.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings,
)