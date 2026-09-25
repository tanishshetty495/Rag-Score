from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import List

class FakeRetriever(BaseRetriever):
    def _get_relevant_documents(self, query: str, *, run_manager) -> List[Document]:
        # Return a fixed document
        return [
            Document(page_content="This is a test document about apples.", metadata={"id": "1"}),
            Document(page_content="Another test document about oranges.", metadata={"id": "2"})
        ]
