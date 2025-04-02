import asyncio
from dotenv import load_dotenv
from rag import RAGWorkflow
from mcp.server.fastmcp import FastMCP

load_dotenv()

mcp = FastMCP('rag-server')
rag_workflow = RAGWorkflow()

@mcp.tool()
async def rag(query: str, streaming: bool = False):
    """Use RAG workflow to answer queries using documents from data directory.
    
    Args:
        query (str): The query text to process
        streaming (bool, optional): Whether to stream the response. Defaults to False.
    """
    if streaming:
        async for chunk in rag_workflow.query_stream(query):
            yield chunk
    else:
        response = await rag_workflow.query(query)
        yield response

if __name__ == "__main__":
    asyncio.run(rag_workflow.ingest_documents("data"))
    mcp.run(transport="stdio")