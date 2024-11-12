from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain.chains.summarize import load_summarize_chain


def create_db_from_url(url: str):
    loader = WebBaseLoader(url)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)[:4]
    return docs

def generate_summary(docs):
  llm = Ollama(model="phi3")
  chain = load_summarize_chain(llm,
                            chain_type="map_reduce",
                            verbose = True)
  output_summary = chain.invoke(docs)
  summary = output_summary['output_text']
  return summary


url = "https://arxiv.org/abs/2401.09334"


if url:
  docs = create_db_from_url(url)
  response = generate_summary(docs)
  print(response)