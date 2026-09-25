async def my_retriever(query, top_k=5):
    from rag_score.core.types import RetrievedChunk
    return [RetrievedChunk(doc_id='doc_1', text='This is a test document about apples.')]

async def my_generator(query, context):
    return "The answer is apples."
