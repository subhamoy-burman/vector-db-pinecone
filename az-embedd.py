import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_openai import AzureChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_openai import AzureOpenAIEmbeddings  # Correct class for Azure


load_dotenv()
if __name__ == '__main__':
   print("ingestion.py is being run directly")
   loader = TextLoader(r"C:\Users\sburman\Projects\py-vector-db\mediumblog1.txt", encoding='utf-8')
   document = loader.load()

   print("Document loaded")

   text_splitter = CharacterTextSplitter(chunk_size=1000,chunk_overlap=0)
   texts = text_splitter.split_documents(document)

   print("Texts split done")


   # Extract text content from each Document object
   # text_contents = [text.content for text in texts]
   
   # Use Azure OpenAI embeddings
   # Use Azure OpenAI embeddings
   embeddings = AzureOpenAIEmbeddings(
        model="text-embedding-ada-002",
        azure_deployment="subhamoy-text-embeddings"
    # dimensions: Optional[int] = None, # Can specify dimensions with new text-embedding-3 models
    # azure_endpoint="https://<your-endpoint>.openai.azure.com/", If not provided, will read env variable AZURE_OPENAI_ENDPOINT
    # api_key=... # Can provide an API key directly. If missing read env variable AZURE_OPENAI_API_KEY
    # openai_api_version=..., # If not provided, will read env variable AZURE_OPENAI_API_VERSION
    )

   PineconeVectorStore.from_documents(texts,embeddings, index_name=os.environ['INDEX_NAME'])
   print("Ingestion Done")