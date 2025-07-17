from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

class VectorStoreBuilder:

    def __init__(self, csv_file_path: str, persist_dir:str= "chroma_db"):
        """ Initializes the vector store builder with the path to the CSV file and the vector store path."""
        self.csv_file_path = csv_file_path
        self.persist_dir = persist_dir
        self.embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    def build_vector_store(self):
        try:
            # Load the CSV data
            loader = CSVLoader(file_path=self.csv_file_path, encoding='utf-8', metadata_columns=[])
            documents = loader.load()

            # Split the text into manageable chunks
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            split_docs = text_splitter.split_documents(documents)

            # Create a Chroma vector store
            vector_store = Chroma.from_documents(split_docs, self.embedding, persist_directory=self.persist_dir)

            return vector_store
        except Exception as e:
            print(f"An error occurred while building the vector store: {e}")
            return None
        

    def load_vector_store(self):
        """ Loads the vector store from the specified directory."""
        try:
            vector_store = Chroma(persist_directory=self.persist_dir, embedding_function=self.embedding)
            return vector_store
        except Exception as e:
            print(f"An error occurred while loading the vector store: {e}")
            return None