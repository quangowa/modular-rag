from typing import List

from haystack import Document, component


@component
class PassThroughDocumentsComponent:
    """
    A component for normalizing the input and output of the pipeline
    """

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]):
        return {"documents": documents}

@component
class PassThroughTextComponent:
    """
    A component for normalizing the input and output of the pipeline
    """

    @component.output_types(text=str)
    def run(self, text: str):
        return {"text": text}
