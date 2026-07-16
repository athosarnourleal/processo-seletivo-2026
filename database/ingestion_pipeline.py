import sys
from pathlib import Path
import pypdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

sys.path.append(str(Path(__file__).resolve().parent.parent))

from database.VectorStoreManagement import vector_store_manager


def loadTxtFile(doc_path: Path):
    return doc_path.read_text(encoding='utf-8')

def loadPDFText(doc_path: Path):
    reader = pypdf.PdfReader(str(doc_path))

    text = ''
    for page in reader.pages:
        text+= page.extract_text()

    return text

def loadTextsFromDirectory(directory_name: str) -> list[Document]:

    # search for package in root
    documents_directory = Path(__file__).parent.parent / directory_name

    if not documents_directory.exists():
        print(f"directory {documents_directory} does not exist")
        return []

    # add all found documents inside the designated package into the 'all_documents' list
    all_documents = []
    for cur_path in documents_directory.iterdir():
        extracted_text = ''

        if cur_path.suffix == '.pdf':
            extracted_text = loadPDFText(cur_path)
        elif cur_path.suffix == '.txt':
            extracted_text = loadTxtFile(cur_path)

        if extracted_text != '':
            doc = Document(
                page_content=extracted_text,
                metadata={"source": cur_path.name}
            )

            all_documents.append(doc)

    return all_documents

def chunkDocuments(documents: list[Document]) -> list[Document]:
    if len(documents) == 0:
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 2000,
        chunk_overlap = 800,
        separators= ['/n/n', '.', '\n', ',']
    )

    chunks = []
    for single_document in documents:
        raw_chunked_texts = text_splitter.split_text(single_document.page_content)
        for text_chunk in raw_chunked_texts:
            chunk_doc = Document(
                page_content= text_chunk,
                metadata= {"source": single_document.metadata["source"]}
            )
            chunks.append(chunk_doc)

    return chunks

def runIngestionPipeline():
    # load documents
    docs = loadTextsFromDirectory('corpus')
    if len(docs) == 0:
        print("no documents found...")
        return
    # chunk documents
    chunks = chunkDocuments(docs)

    # create vector database in the persistent client
    vector_store_manager.addToVectorStore(langchain_documents= chunks)

    token_number = vector_store_manager.getTokensSpent()

    print(f"INGESTION COMPLETE!\n(spent a total of {token_number} tokens)")

if __name__ == "__main__":
    runIngestionPipeline()
