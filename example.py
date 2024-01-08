from langchain.retrievers.web_research_v2 import WebResearchRetriever
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper


# Vectorstore
vectorstore = Chroma(
    embedding_function=OpenAIEmbeddings(), persist_directory="./chroma_db_oai"
)

# LLM
llm = ChatOpenAI(temperature=0)

# Search
search = DuckDuckGoSearchAPIWrapper()

# # Initialize
web_research_retriever = WebResearchRetriever.from_llm(
    vectorstore=vectorstore,
    llm=llm,
    search=search,
    num_search_results = 3,
)

from langchain.chains import RetrievalQAWithSourcesChain

user_input = "what is the capital of France?"
qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm, retriever=web_research_retriever
)
result = qa_chain({"question": user_input})

print(result['answer'])
