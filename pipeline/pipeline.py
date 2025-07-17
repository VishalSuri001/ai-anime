from src.vector_store import VectorStoreBuilder
from src.recommender import AnnimeRecommender
from config.config import GROQ_API_KEY, MODEL_NAME
from utils.logger import get_logger
from utils.custom_exception import CustomException

logger = get_logger(__name__)


class AnimeRecommenderPipeline:
    def __init__(self, persist_dir: str = "chroma_db"):
        """ Initializes the anime recommender pipeline with a vector store and model name."""
        try:
            logger.info("Anime Recommender Pipeline initialized successfully.")
            
            retriever = VectorStoreBuilder(csv_file_path="", persist_dir=persist_dir).load_vector_store().as_retriever()

            self.recommender = AnnimeRecommender(
                retriever=retriever,
                api_key=GROQ_API_KEY,
                model_name=MODEL_NAME
            )
        except Exception as e:
            logger.error(f"Error initializing Anime Recommender Pipeline: {str(e)}")
            raise CustomException("Failed to initialize Anime Recommender Pipeline", e)
        
    def recommend(self, question: str) -> str:
        """ Generates anime recommendations based on the user's question."""
        try:
            logger.info(f"Generating recommendations for question: {question}")
            response = self.recommender.get_recommendation(question)
            if response:
                logger.info("Recommendations generated successfully.")
                return response
            else:
                logger.warning("No recommendations found.")
                return "No recommendations found."
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            raise CustomException("Failed to generate recommendations", e)