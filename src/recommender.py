from langchain_groq import ChatGroq
from src.prompt_template import get_anime_prompt
from langchain_core.runnables import RunnableMap
from langchain_core.prompts import PromptTemplate

class AnnimeRecommender:
    def __init__(self, retriever, api_key: str, model_name: str):
        """Initializes the anime recommender with retriever and model."""
        self.model_name = model_name
        self.llm = ChatGroq(api_key=api_key, model_name=model_name, temperature=0.1)
        self.prompt_template: PromptTemplate = get_anime_prompt()

        # Define the new style LCEL chain
        self.qa_chain = (
            RunnableMap({
                "context": lambda x: retriever.get_relevant_documents(x["question"]),
                "question": lambda x: x["question"],
            }) 
            | self.prompt_template
            | self.llm
        )

    def get_recommendation(self, question: str) -> str:
        """Generates anime recommendations based on the user's question."""
        try:
            response = self.qa_chain.invoke({"question": question})
            return response.content
        except Exception as e:
            print(f"An error occurred while generating recommendations: {e}")
            return None
