"""
RAG系统配置文件
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]

@dataclass
class RAGConfig:
    """RAG系统配置类"""

    # 路径配置
    data_path: str = str(PROJECT_ROOT / "data" / "C8" / "cook")
    # 教学阶段复用索引实验已经生成的数据。
    index_save_path: str = str(PROJECT_ROOT / "learning" / "artifacts" / "c8_vector_index")

    # 模型配置
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_cache_folder: str = str(PROJECT_ROOT / "models" / "huggingface")
    embedding_local_files_only: bool = True
    llm_model: str = "deepseek-flash"

    # 检索配置
    top_k: int = 3
    use_chinese_bm25: bool = True

    # 生成配置
    temperature: float = 0.1
    max_tokens: int = 2048

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RAGConfig':
        """从字典创建配置对象"""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)

# 默认配置实例
DEFAULT_CONFIG = RAGConfig()
