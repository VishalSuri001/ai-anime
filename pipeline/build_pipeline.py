from src.data_loader import AnimeDataLoader
from src.vector_store import VectorStoreBuilder
from src.recommender import AnnimeRecommender
from config.config import GROQ_API_KEY, MODEL_NAME
from utils.logger import get_logger
from utils.custom_exception import CustomException
from dotenv import load_dotenv

load_dotenv()

def main():
    logger = get_logger(__name__)
    logger.info("Pipeline init.")
    try:
        # Load and process the anime data
        data_loader = AnimeDataLoader(file_path="data/anime_with_synopsis.csv", processed_file_path="data/anime_updated.csv")
        processed_csv = data_loader.load_and_process_data()
        logger.info("Pipeline - try to load and process anime data.")
        if not processed_csv:
            raise CustomException("Failed to load and process anime data")

        # Build the vector store
        vector_store_builder = VectorStoreBuilder(csv_file_path=processed_csv)
        vector_store_builder.build_vector_store()                

        logger.info("Vector store built successfully.")

    except CustomException as ce:
        logger.error(f"Custom Exception: {ce}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")

if __name__=="__main__":
     main()