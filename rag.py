from typing import List, Dict, Optional, Literal

import glob
import json
import uuid
from time import time
from bs4 import BeautifulSoup

import chromadb
from chromadb.api.types import QueryResult
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
from chromadb import Documents, EmbeddingFunction, Embeddings

from langchain_core.documents.base import Document
from langchain_community.document_loaders import BSHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from openai import AzureOpenAI
from openai.types.chat import ChatCompletionUserMessageParam

from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

from utils import Case
from prompts import CASE_PLACEHOLDER, SUPPORTING_CONTENT_PLACEHOLDER


class CompletionModel:
    def __init__(
            self,
            provider: Literal["mistral", "openai"],
            api_key: str,
            api_version: Optional[str],
            endpoint: str,
            model_deployment: str
        ):
        """Class interfacing with the model deployments"""
        self.provider = provider
        self.model = model_deployment

        if self.provider == "mistral":
            self.client = MistralClient(
                api_key=api_key,
            )
            
        elif provider == "openai":
            self.client = AzureOpenAI(
                api_version=api_version,
                api_key=api_key,
                azure_endpoint=endpoint,
            )
        else:
            raise ValueError("Model provider was not recognized, must be either 'mistral' or 'openai'.")

    def call(
        self,
        messages: List[Dict],
        temperature: Optional[float],
    ) -> str:
        """Send request to LLM
        
        Parameters:
        -----------
            messages: conversation with the LLM, can include system messages as well as the history
            temperature: impact the imagination and variability of the LLm answers

        Returns:
        --------
            answer: the content of the reponse of the LLM
        """
        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[ChatCompletionUserMessageParam(**msg) for msg in messages],
                max_tokens=None,
                temperature=temperature,
                user="unknown",
            )
        elif self.provider == "mistral":
            response = self.client.chat(
                model=self.model,
                messages=[ChatMessage(**msg) for msg in messages],
                max_tokens=None,
                temperature=temperature,
            )

        # TODO: might contain function call?
        answer = response.choices[0].message.content
        return answer if answer is not None else "Failed to complete"
    
class MistralEmbeddingFunction(EmbeddingFunction):
    def __init__(self, api_key: str, model_deployment: str):
        self.client = MistralClient(
                api_key=api_key,
            )
        self.model = model_deployment

    def __call__(self, input: Documents) -> Embeddings:
        embeddings_batch_response = self.client.embeddings(
            model=self.model,
            input=input
        )
        # TODO: make sure that the order is preserved?!
        return [entry.embedding for entry in embeddings_batch_response.data]

class EmbeddingModel:
    def __init__(self, model_deployment: str, api_key: str):
        """Use API calls to embed content"""
        self.embedding_fun = MistralEmbeddingFunction(
                api_key=api_key,
                model_deployment=model_deployment,
            )
        self.batch_size = 1

    def embed(self, input: Documents):
        nb_batches = len(input) // self.batch_size
        if len(input) % self.batch_size != 0:
            nb_batches += 1
        
        embeddings = []
        for batch_idx in range(nb_batches):
            idx_start = batch_idx * self.batch_size
            idx_end = (batch_idx + 1) * self.batch_size
            batch = input[idx_start:idx_end]
            embeddings += self.embedding_fun(batch)
        return embeddings

class RAGModel:
    def __init__(self, expert_name: str, config: Dict, force_collection_creation: bool = False):
        """Model responsible for consuming the data to build a knowledge database"""
        self.config = config
        self.knowledge_folder = config["knowledge_folder"]
        self.prompt_system = config["prompt_system"]
        self.prompt_template = config["prompt_template"]


        # Slice document into smaller chunks
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"],
            length_function=len,
        )

        # Embedding model, convert natural langage to vector
        self.embedding_model = EmbeddingModel(
            model_deployment=config["embedding_model_deployment"],
            api_key=config["embedding_api_key"],
        )

        # Vector database/Search index
        self.db_client = chromadb.PersistentClient(
            path="chroma",
            settings=Settings(anonymized_telemetry=False),
            tenant=DEFAULT_TENANT,
            database=DEFAULT_DATABASE,
        )
        # TODO: the name of the collection could actually be a parameter of predict, to allow the model to switch between vector db
        self.vectordb = self.db_client.get_or_create_collection(name=expert_name)
        # Empty collection, need to populate it
        if force_collection_creation or self.vectordb.count() == 0:
            self.create_vectordb()

        # Completion model, answer request based on supporting content
        self.completion_model = CompletionModel(
            provider=config["completion_provider"],
            api_key=config["completion_api_key"],
            api_version=config["completion_api_version"],
            endpoint=config["completion_endpoint"],
            model_deployment=config["completion_model_deployment"],
        )

    def create_vectordb(self):
        """Load local document stored as html and add them to the vector database (Chroma Collection)"""
        # Collect all the files to process
        all_files = glob.glob(f"{self.knowledge_folder}/**/**/**.html", recursive=True)

        # Load & slice the documents by section/articles
        chunks = []
        for f_path in all_files:
            # TODO: store metadata
            chunks += self._load_and_split_document(f_path)

        # Embedd the content
        vectors = self.embedding_model.embed(chunks)

        # Create ids for each chunk
        ids = [str(uuid.uuid4()) for _ in range(len(chunks))]

        # update the vectordb with the new chunks & vectors
        self.vectordb.add(
            embeddings=vectors,
            documents=chunks,
            ids=ids,
            metadatas=None,
        )


    def _load_and_split_document(self, f_path: str) -> List[str]:
        """Load the document (html page), extract the section tags."""
        with open(f_path, "r") as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
            sections = soup.find_all("section")

        chunks = []
        for sec in sections:
            # heuristic to keep only legal text
            if "Art" in sec.text and len(sec.text) > 10:
                if len(sec.text) > self.splitter._chunk_size:
                    chunks += self.splitter.split_text(sec.text)
                else:
                    chunks.append(sec.text)
        return chunks
            

    def predict(self, case: Case):
        """
        Execute all the steps of the RAG logic:
            1. embed the query
            2. retrieve the supporting content
            3. update the prompt with the information
            4. return the completion model reponse
        """
        relevant_chunks = self._retrieve_supporting_content(case.description)

        # convert relevant chunks to a list of string
        relevant_chunks_content = relevant_chunks["documents"]
        if relevant_chunks_content is not None:
            relevant_chunks_str = relevant_chunks_content[0]
        else:
            relevant_chunks_str = []

        completion_query = self._inject_content_prompt(
            case_description=case.description,
            supporting_content=relevant_chunks_str,
        )

        answer = self.completion_model.call(
            messages=[
                {"role": "system", "content": self.prompt_system},
                {"role": "user", "content": completion_query}
            ],
            temperature=self.config["temperature"]
        )
        return {
            "answer": answer,
            "support_content": relevant_chunks_str,
        }
    
    def predict_from_dataset(self, dataset: List[Case], export_predictions: bool = True):
        # Generate prediction for all the cases in the dataset
        predictions = [self.predict(case=entry) for entry in dataset]

        # Export the predictions
        if export_predictions:
            with open(f"{time()}_predictions.json", "w") as f:
                json.dump(predictions, f, indent=4)

    def _retrieve_supporting_content(self, query: str) -> QueryResult:
        # Embed the query
        vector_query = self.embedding_model.embed([query])

        # Retrieve relevant chunks
        relevant_chunks = self.vectordb.query(
            query_embeddings=vector_query,
            n_results=self.config["n_results"],
        )
        return relevant_chunks

    def _inject_content_prompt(
        self, case_description: str, supporting_content: List[str]
    ) -> str:
        completion_query = self.prompt_template

        # inject case description in the completion request
        if CASE_PLACEHOLDER in self.prompt_template:
            completion_query = completion_query.replace(CASE_PLACEHOLDER, case_description)
        else:
            raise ValueError("Could not find the query placeholder in the prompt template.")

        # inject supporting content in completion request
        if SUPPORTING_CONTENT_PLACEHOLDER in self.prompt_template:
            completion_query = completion_query.replace(
                SUPPORTING_CONTENT_PLACEHOLDER, "\n".join(supporting_content)
            )
        else:
            raise ValueError(
                "Could not find the supporting content placeholder in the prompt template."
            )
        return completion_query
