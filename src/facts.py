from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma

load_dotenv() #define your API key in a .env file

embeddings = OpenAIEmbeddings()

loader = TextLoader("facts.txt")
splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 200,
    chunk_overlap = 0,
)

docs = loader.load_and_split(text_splitter=splitter)

db = Chroma.from_documents(
    docs, 
    embedding = embeddings,
    persist_directory = "emb"
)

results = db.similarity_search_with_score("What is an interesting fact about the english language?")

for result in results:
    print("\n")
    print(result[1])
    print(result[0].page_content)