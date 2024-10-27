from langchain.text_splitter import CharacterTextSplitter
def chunks(txt):
    splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0
    )
    chunks = splitter.split_text(txt)
    return chunks