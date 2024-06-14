from rag import RAGModel
from utils import Case, LawDomain
from prompts import PROMPT_SYSTEM, PROMPT_TEMPLATE

from dotenv import dotenv_values

def main() -> None:
    # Load the environment variables
    env_var = dotenv_values(".env")

    # Inject the environment variables inside the configuration
    config = {
        # rag knowledge path
        "knowledge_folder":"data/swiss_lex",
        # rag prompts
        "prompt_system":PROMPT_SYSTEM,
        "prompt_template": PROMPT_TEMPLATE,
        # splitter
        "chunk_size":300,
        "chunk_overlap":10,
        # embedding model
        "embedding_provider":"mistral",
        "embedding_model_deployment":env_var["MISTRAL_EMBED_MODEL_DEPLOYMENT"],
        "embedding_api_key":env_var["MISTRAL_EMBED_API_KEY"],
        "embedding_api_version":"",
        "embedding_endpoint":"",
        # retrieval parameters
        "n_results": 5,
        # completion model
        "completion_provider":"mistral",
        "completion_model_deployment":env_var["MISTRAL_SMALL_MODEL_DEPLOYMENT"],
        "completion_api_key":env_var["MISTRAL_SMALL_API_KEY"],
        "completion_api_version":"",
        "completion_endpoint":"",
        "temperature":0,
    }

    # Dummy Dataset
    ds_cases = [
        Case(
            description="I murdered my birthday cake. The event happened in my chalet in Valais, I was feeling very sad and I stabed the cake instead of making slices.",
            related_articles=[],
            domain=LawDomain.CRIMINAL,
            outcome=False,
        ),
        Case(
            description="I want to build a house on a proprety I bought last year but the neighbour are blocking the construction. Fancy house, with a swimming pool in a small village in the Canton of Bern.",
            related_articles=[],
            domain=LawDomain.PUBLIC,
            outcome=True,
        ),
        Case(
            description="I asked for the Swiss citizenship after living 6 years in the same commune. I have been living with my Swiss wife in the center of Geneva for the last 6 years, we have been married for 5 years.",
            related_articles=[],
            domain=LawDomain.CIVIL,
            outcome=True,
        ),
    ]

    model = RAGModel(
        expert_name="generic",
        config=config,
        force_collection_creation=False,
    )

    model.predict_from_dataset(ds_cases)

if __name__ == "__main__":
    main()
