from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings

load_dotenv() #define your API key in a .env file

embeddings = OpenAIEmbeddings()

loader = TextLoader("facts.txt")
splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 200,
    chunk_overlap = 0,
)

docs = loader.load_and_split(text_splitter=splitter)

for doc in docs: 
    print(doc.page_content)
    print("\n")