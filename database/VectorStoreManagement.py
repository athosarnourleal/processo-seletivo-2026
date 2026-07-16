# setup
import os
from dotenv import load_dotenv

load_dotenv()

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

# vectorstores
from langchain_core.documents import Document
import chromadb

# embeddings GOOGLE
from google.genai.types import ContentEmbedding
from google.genai.models import types
from google import genai

# embeddings OpenAI
from openai import OpenAI
from openai.types import Embedding

import tiktoken

# debug
from docs.TraceManager import addToTraceJson

def printBarProgress(current: int, total: int, label: str = "progress", width: int = 100) -> None:
    progress = int((current*width)/total)
    left = width - progress
    percent = int((current*width)/total)
    print(f"{label}: ","#"*progress,"-"*left,f"{percent}% done",sep = "")

class VectorStoreManager:
    """manages the chroma database"""

    def __init__(self,
        persist_directory: str= str(Path(__file__).parent.parent/'vector_store'),
        google_model="gemini-embedding-001",
        openai_model="text-embedding-ada-002",
    ) -> None:
        self.directory = persist_directory

        self.MODEL_GOOGLE = google_model
        self.MODEL_OPENAI = openai_model

        # setup clients
        self.google_client = genai.Client(
            api_key=os.getenv('GOOGLE_API_KEY')
        )
        self.openai_client = OpenAI(
            api_key=os.getenv('OPENAI_API_KEY')
        )
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)

        # create/connect to collection
        self.collection = self.chroma_client.get_or_create_collection(
            name="vectorStore",
            metadata={"hnsw:space": "cosine"} # make chroma use cosine similarity
        )

        self.tokens_spent = 0

    # MAIN FUNCTIONS

    def searchQuery(self, query: str):
        # embed query
        embedded_query = self.vectorizeTextOpenAi(text= query)

        # perform search
        response = self.collection.query(
            query_embeddings= embedded_query,
            n_results= 4
        )

        chunk_context = self.reformulateChunksIntoLLMContext(dict(response))

        addToTraceJson({"retrieved_chunks": chunk_context})

        return chunk_context

    def addToVectorStore(self, langchain_documents: list[Document]):
        print("number of documents:", len(langchain_documents))

        # extract, index and embed chunks
        ids = []
        docs = []
        metadatas = []
        embeddings = []
        for count,doc in enumerate(langchain_documents):
            ids.append(f"id{count+1}") # id1, id2, id3...
            docs.append(doc.page_content)
            metadatas.append(doc.metadata)
            embeddings.append(self.vectorizeTextOpenAi(text= doc.page_content))

            printBarProgress(current= count, total= len(langchain_documents), label="embedding")


        # print(f"ids:{len(ids)} \ndocs:{len(docs)} \nmetadatas:{len(metadatas)} \nvectors:{len(embeddings)}")

        if len(ids) != len(embeddings):
            raise "embedding number is different of the id number"

        # add documents to collection
        self.collection.add(
            documents= docs,
            metadatas= metadatas,
            ids= ids,
            embeddings= embeddings
        )

        # update the Json with the amount spent
        self.getTokensSpent()

        return self.collection

    def getTokensSpent(self) -> int:
        addToTraceJson({"tokens spent in ingestion": self.tokens_spent})
        return self.tokens_spent

    # UTILITY  FUNCTIONS

    def vectorizeTextOpenAi(self, text: str):
        """get a single embedding that represents the whole text"""

        embeddings: list[Embedding] = self.embedTextOpenAi(text)

        # extracts embeddings for each word in a list[list[float]]
        extracted_embeddings = [e.embedding for e in embeddings]

        text_vector = self.getEmbeddingsAverage(extracted_embeddings)

        return text_vector

    def vectorizeTextGoogle(self, text: str) -> list[float]:
        """get a single embedding that represents the whole text"""

        embeddings: list[ContentEmbedding] | None = self.embedTextGoogle(text)

        if embeddings is None:
            return []

        # extracts embeddings for each word in a list[list[float]]
        extracted_embeddings = [e.values for e in embeddings]

        text_vector = self.getEmbeddingsAverage(extracted_embeddings)

        return text_vector

    def embedTextGoogle(self, text: str):
        """get the embedding values of each word"""

        result: types.EmbedContentResponse = self.google_client.models.embed_content(
            model= self.MODEL_GOOGLE,
            contents= text
        )

        return result.embeddings # the embeddings for each word

    def embedTextOpenAi(self, text: str):
        """get the embedding values of each word"""

        response = self.openai_client.embeddings.create(
            input=text,
            model=self.MODEL_OPENAI
        )

        self.tokens_spent += self.num_tokens_from_string(string= text)

        return response.data

    @staticmethod
    def num_tokens_from_string(
            string: str,
            encoding_name: str = "cl100k_base"  # basic encoding for OpenAi
    ) -> int:
        """Returns the number of tokens in a text string."""
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens

    @staticmethod
    def getEmbeddingsAverage(unclean_embeddings: list[list[float]| None]) -> list[float]:
        embeddings = []

        # remove null(None) embeddings
        for emb in unclean_embeddings:
            if emb is not None:
                embeddings.append(emb)

        # get sum of embeddings
        text_embedding = embeddings[0]
        for emb in embeddings[1:]:
            for i in range(len(text_embedding)):
                text_embedding[i] += emb[i]

        # divide sum by total
        for i in range(len(text_embedding)):
            text_embedding[i] /= len(embeddings)

        return text_embedding

    @staticmethod
    def reformulateChunksIntoLLMContext(retrieval_response: dict):
        """adequate the retrieved chunks from chromadb to a more easily readable format"""

        texts = retrieval_response["documents"][0]
        metadatas = retrieval_response["metadatas"][0]

        chunks = ''

        for i in range(len(texts)):
            reformulated_chunk = f"\n{texts[i]}\nSOURCED FROM DOCUMENT: {metadatas[i]['source']}\n\n"
            chunks += reformulated_chunk

        return chunks

# instantiate in global scope
vector_store_manager = VectorStoreManager()


