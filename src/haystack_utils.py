from typing import Any, Dict, List

from haystack import Document, component


@component
class PassThroughDocuments:
    """
    A component for normalizing the input and output of the pipeline
    """

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]):
        return {"documents": documents}

@component
class DocumentToList:
    """
    A component for normalizing the input and output of the pipeline
    """

    @component.output_types(document=Document, documents=List[Document])
    def run(self, document: Document):
        return {"document" : document, "documents": [document]}

@component
class PassThroughText:
    """
    A component for normalizing the input and output of the pipeline
    """

    @component.output_types(text=str)
    def run(self, text: str):
        return {"text": text}


@component
class AddLLMMetadata:
    """
    A component for adding an object to a document's metadata
    """

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document], replies: List[str]) -> Dict[str, Any]:
        for doc in documents:
            doc.meta["llm_extracted_info"] = "\n".join(replies)
        return {"documents": documents}
