from langchain.document_loaders import CSVLoader
from langchain.text_splitter import CharacterTextSplitter

def load_and_split(file_path):
    loader = CSVLoader(file_path)
    documents = loader.load()
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(documents)
