from rag_score.core.types import RetrievedChunk

async def my_retriever(query: str, top_k: int = 1):
    """A simple retriever that returns a fixed document."""
    # Ignore the query and return a fixed document
    return [
        RetrievedChunk(doc_id="doc1", text="Paris is the capital of France."),
        RetrievedChunk(doc_id="doc2", text="Berlin is the capital of Germany."),
    ][:top_k]

async def my_generator(query: str, context: list[RetrievedChunk]) -> str:
    """A simple generator that returns a fixed answer."""
    # Ignore the query and context, return a fixed answer
    return "The capital is Paris."