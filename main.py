from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_openai import AzureChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_openai import AzureOpenAIEmbeddings  # Correct class for Azure
from langchain_core.prompts import PromptTemplate
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
import os


load_dotenv()

if __name__ == '__main__':
    print("Retrieving data")

    embeddings = AzureOpenAIEmbeddings(
        model="text-embedding-ada-002",
        azure_deployment="subhamoy-text-embeddings"
    # dimensions: Optional[int] = None, # Can specify dimensions with new text-embedding-3 models
    # azure_endpoint="https://<your-endpoint>.openai.azure.com/", If not provided, will read env variable AZURE_OPENAI_ENDPOINT
    # api_key=... # Can provide an API key directly. If missing read env variable AZURE_OPENAI_API_KEY
    # openai_api_version=..., # If not provided, will read env variable AZURE_OPENAI_API_VERSION
    )

    llm = AzureChatOpenAI(
        temperature=0,
        stop=["\nObservation"],
        openai_api_key=os.environ['OPENAI_API_KEY'],
        openai_api_version="2024-08-01-preview",  # Specify API version
        azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],
        azure_deployment="gpt-4o",
        model="gpt-4o"
    )

    query = "What is pinecone in machine learning?"

    #chain = PromptTemplate.from_template(template=query) | llm
    #result = chain.invoke(input = {})
    #print(result.content)

    vectorstore = PineconeVectorStore(
        index_name=os.environ['INDEX_NAME'],
        embedding=embeddings
    )

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)

    retrival_chain = create_retrieval_chain(retriever=vectorstore.as_retriever(), combine_docs_chain = combine_docs_chain)

    result = retrival_chain.invoke(input={"input": query})

    print(result)

