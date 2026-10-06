from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


class FakeRetriever(BaseRetriever):
    def _get_relevant_documents(self, query: str, *, run_manager) -> list[Document]:
        # Return a fixed document
        return [
            Document(page_content="This is a test document about apples.", metadata={"id": "1"}),
            Document(page_content="Another test document about oranges.", metadata={"id": "2"}),
        ]
