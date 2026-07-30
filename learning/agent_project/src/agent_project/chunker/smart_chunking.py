from langchain_text_splitters import RecursiveCharacterTextSplitter

def smart_chunk_text(text: str, chunk_size:int = 500, chunk_overlap: int = 50) -> list:
    """
    使用 langchain 的 RecursiveCharacterTextSplitter 将文本分割为多个块。
    
    :param text: 待分割的文本
    :param chunk_size: 每个块的最大字符数
    :param chunk_overlap: 块之间的重叠字符数
    :return: 分割后的文本块列表
    """
    Chinese_characters = ['。', '，', '！', '？', '；']
    English_characters = [',', '.', '!', '?', ';']

    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap 必须大于或等于 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap 必须小于 chunk_size")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", *Chinese_characters, *English_characters, " "],
    )
    
    chunks = text_splitter.split_text(text)
    
    return chunks