import os

import nest_asyncio
from typing import Union, List
from dotenv import load_dotenv

from llama_index.llms.vllm.base import VllmServer
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.text_embeddings_inference import (
    TextEmbeddingsInference,
)
from llama_index.core.settings import Settings
from llama_index.core.workflow import Event, Context, Workflow, StartEvent, StopEvent, step
from llama_index.core.schema import NodeWithScore, MetadataMode
from llama_index.core.response_synthesizers import (
    ResponseMode,
    get_response_synthesizer,
)
from llama_index.core.text_splitter import SentenceSplitter
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.vector_stores.qdrant import QdrantVectorStore
import qdrant_client
from qdrant_client.async_qdrant_client import AsyncQdrantClient
from prompt import DEFAULT_CITATION_CHUNK_SIZE, DEFAULT_CITATION_CHUNK_OVERLAP, CITATION_QA_TEMPLATE, CITATION_REFINE_TEMPLATE

load_dotenv()

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

class RetrieverEvent(Event):
    """Result of running retrieval"""
    nodes: list[NodeWithScore]

class CreateCitationsEvent(Event):
    """Add citations to the nodes."""
    nodes: list[NodeWithScore]

class RAGWorkflow(Workflow):
    def __init__(self):
        super().__init__()
        
        # Initialize LLM and embedding model
        # self.llm = VllmServer(
        #     api_url=os.getenv("LLM_MODEL_URL"),
        #     max_new_tokens=1024,
        #     temperature=0,
        # )
        
        self.llm = OpenAI(model="gpt-4o-mini")
        self.embed_model = TextEmbeddingsInference(base_url=os.getenv("EMBEDDING_MODEL_URL"), model_name="thaile1809/Vietnamese_Embedding_fast")
        
        # Configure global settings
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        
        # Initialize async Qdrant client
        self.aclient = AsyncQdrantClient(
            url=os.getenv("QDRANT_URL_DB"),
            port=6333
        )
        vector_store = QdrantVectorStore(
            client=qdrant_client.QdrantClient(url=os.getenv("QDRANT_URL_DB"), port=6333),
            aclient=self.aclient,
            collection_name="paul_graham"
        )
        self.index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

    @step
    async def retrieve(self, ctx: Context, ev: StartEvent) -> Union[RetrieverEvent, None]:
        """Entry point for RAG retrieval."""
        query = ev.get("query")
        streaming = ev.get("streaming", False)  # Get streaming parameter
        index = ev.get("index") or self.index

        if not query:
            return None

        if index is None:
            print("Index is empty, load some documents before querying!")
            return None

        retriever = index.as_retriever(similarity_top_k=2)
        nodes = await retriever.aretrieve(query)
        await ctx.set("query", query)
        await ctx.set("streaming", streaming)  # Set streaming in context
        return RetrieverEvent(nodes=nodes)

    @step
    async def create_citation_nodes(
        self, ev: RetrieverEvent
    ) -> CreateCitationsEvent:
        """
        Modify retrieved nodes to create granular sources for citations.

        Takes a list of NodeWithScore objects and splits their content
        into smaller chunks, creating new NodeWithScore objects for each chunk.
        Each new node is labeled as a numbered source, allowing for more precise
        citation in query results.

        Args:
            nodes (List[NodeWithScore]): A list of NodeWithScore objects to be processed.

        Returns:
            List[NodeWithScore]: A new list of NodeWithScore objects, where each object
            represents a smaller chunk of the original nodes, labeled as a source.
        """
        nodes = ev.nodes

        new_nodes: List[NodeWithScore] = []

        text_splitter = SentenceSplitter(
            chunk_size=DEFAULT_CITATION_CHUNK_SIZE,
            chunk_overlap=DEFAULT_CITATION_CHUNK_OVERLAP,
        )
        for node in nodes:
            text_chunks = text_splitter.split_text(
                node.node.get_content(metadata_mode=MetadataMode.NONE)
            )

            for text_chunk in text_chunks:
                text = f"Source {len(new_nodes)+1}:\n{text_chunk}\n"

                new_node = NodeWithScore(
                    node=TextNode.model_validate(node.node), score=node.score
                )
                new_node.node.text = text
                new_nodes.append(new_node)
        return CreateCitationsEvent(nodes=new_nodes)
    
    @step
    async def synthesize(self, ctx: Context, ev: CreateCitationsEvent) -> StopEvent:
        """Generate a response using retrieved nodes with citations."""
        
        query = await ctx.get("query", default=None)
        streaming = await ctx.get("streaming", default=False)
        print("Synthesize step - streaming:", streaming)  # Debug log
        
        synthesizer = get_response_synthesizer(
            llm=self.llm,
            text_qa_template=CITATION_QA_TEMPLATE,
            refine_template=CITATION_REFINE_TEMPLATE,
            response_mode=ResponseMode.COMPACT,
            streaming=streaming,
            use_async=True,
        )

        if streaming:
            print("Using streaming response")  # Debug log
            response_gen = await synthesizer.asynthesize(query, nodes=ev.nodes)
            return StopEvent(result={"response_gen": response_gen})
        else:
            print("Using non-streaming response")  # Debug log
            response = await synthesizer.asynthesize(query, nodes=ev.nodes)
            return StopEvent(result={"response": response})

    async def query_stream(self, query_text: str):
        """Helper method to perform a streaming RAG query."""
        if self.index is None:
            raise ValueError("No documents have been ingested. Call ingest_documents first.")
        
        print("Starting streaming query")  # Debug log
        result = await self.run(query=query_text, index=self.index, streaming=True)
        print("Got streaming result")  # Debug log
        response_gen = result["response_gen"]
        async for text in response_gen.response_gen:
            yield text

    async def query(self, query_text: str):
        """Helper method to perform a non-streaming RAG query.
        
        Args:
            query_text (str): The query text to process
            
        Returns:
            str: The complete response as a string.
        """
        if self.index is None:
            raise ValueError("No documents have been ingested. Call ingest_documents first.")
        
        result = await self.run(query=query_text, index=self.index, streaming=False)
        return str(result["response"])

# Example usage
async def main():
    # Initialize the workflow
    workflow = RAGWorkflow()
    
    # Example of streaming query
    async for chunk in workflow.query_stream("Maybe they'd be able to avoid the worst of the mistakes we'd made"):
        print(chunk, end="", flush=True)

    # Example of non-streaming query
    response = await workflow.query("Maybe they'd be able to avoid the worst of the mistakes we'd made")
    print(response)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())