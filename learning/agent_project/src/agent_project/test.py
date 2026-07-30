import sys
import os
os.environ["HF_HOME"] = r"S:\huggingface_cache"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch, transformers, sentence_transformers
print('全部导入成功！')
print("Python模块搜索路径列表：", sys.path)

