"""
检索优化模块
"""

import logging
import hashlib
import re
from typing import List, Dict, Any, Callable, Optional

from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def chinese_bigram_tokenize(text: str) -> List[str]:
    """教学对照：中文按相邻二字切分，英文数字标识作为完整词项保留。

    二字片段不一定是语言学意义上的词；它用于演示预处理对匹配的影响。
    """
    spans = re.findall(r"[\u4e00-\u9fff]+|[a-z0-9]+(?:[._-][a-z0-9]+)*", text.lower())
    tokens = []
    for span in spans:
        if '\u4e00' <= span[0] <= '\u9fff' and len(span) > 1:
            tokens.extend(span[index:index + 2] for index in range(len(span) - 1))
        else:
            tokens.append(span)
    return tokens


class RetrievalOptimizationModule:
    """检索优化模块 - 负责混合检索和过滤"""
    
    def __init__(
        self,
        vectorstore: FAISS,
        chunks: List[Document],
        *,
        bm25_preprocess_func: Optional[Callable[[str], List[str]]] = None,
    ):
        """
        初始化检索优化模块
        
        Args:
            vectorstore: FAISS向量存储
            chunks: 文档块列表
            bm25_preprocess_func: 可选的词项切分方法，默认沿用原有空格切分
        """
        self.vectorstore = vectorstore
        self.chunks = chunks
        self.bm25_preprocess_func = bm25_preprocess_func
        self.setup_retrievers()

    def setup_retrievers(self):
        """设置向量检索器和BM25检索器"""
        logger.info("正在设置检索器...")

        # 向量检索器
        self.vector_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )

        # BM25检索器
        bm25_options = {"k": 5}
        if self.bm25_preprocess_func is not None:
            bm25_options["preprocess_func"] = self.bm25_preprocess_func
        self.bm25_retriever = BM25Retriever.from_documents(self.chunks, **bm25_options)

        logger.info("检索器设置完成")
    
    def hybrid_search(self, query: str, top_k: int = 3) -> List[Document]:
        """
        混合检索 - 结合向量检索和BM25检索，使用RRF重排

        Args:
            query: 查询文本
            top_k: 返回结果数量

        Returns:
            检索到的文档列表
        """
        # 分别获取向量检索和BM25检索结果
        vector_docs = self.vector_retriever.invoke(query)
        bm25_docs = self.bm25_retriever.invoke(query)

        # 使用RRF重排
        reranked_docs = self._rrf_rerank(vector_docs, bm25_docs)
        return reranked_docs[:top_k]
    
    def metadata_filtered_search(self, query: str, filters: Dict[str, Any], top_k: int = 5) -> List[Document]:
        """
        带元数据过滤的检索
        
        Args:
            query: 查询文本
            filters: 元数据过滤条件
            top_k: 返回结果数量
            
        Returns:
            过滤后的文档列表
        """
        if top_k <= 0:
            return []

        # 在截断前确定可用范围，避免目标文档不在全库前几个结果中而被漏掉。
        eligible = [
            index for index, chunk in enumerate(self.chunks)
            if self._metadata_matches(chunk.metadata, filters)
        ]
        logger.info("元数据条件 %s 对应 %d 个子块", filters, len(eligible))
        if not eligible:
            return []
        candidate_k = min(max(5, top_k), len(eligible))

        # 当前 FAISS 封装先找候选再过滤。小规模教程库取全部向量候选，
        # 确保目标范围不会被很小的 fetch_k 漏掉；复用原有向量，不重新建库。
        vector_docs = self.vectorstore.similarity_search(
            query,
            k=candidate_k,
            filter=lambda metadata: self._metadata_matches(metadata, filters),
            fetch_k=int(self.vectorstore.index.ntotal),
        )

        # 保留全库 BM25 词项统计，只在符合条件的范围里选择前几个结果。
        query_tokens = self.bm25_retriever.preprocess_func(query)
        scores = self.bm25_retriever.vectorizer.get_scores(query_tokens)
        ranked_indices = sorted(eligible, key=lambda index: float(scores[index]), reverse=True)
        bm25_docs = [self.chunks[index] for index in ranked_indices[:candidate_k]]

        return self._rrf_rerank(vector_docs, bm25_docs)[:top_k]

    @staticmethod
    def _metadata_matches(metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        for key, value in filters.items():
            if key not in metadata:
                return False
            if isinstance(value, list):
                if metadata[key] not in value:
                    return False
            elif metadata[key] != value:
                return False
        return True

    def _rrf_rerank(self, vector_docs: List[Document], bm25_docs: List[Document], k: int = 60) -> List[Document]:
        """
        使用RRF (Reciprocal Rank Fusion) 算法重排文档

        Args:
            vector_docs: 向量检索结果
            bm25_docs: BM25检索结果
            k: RRF参数，用于平滑排名

        Returns:
            重排后的文档列表
        """
        doc_scores = {}
        doc_objects = {}

        # 计算向量检索结果的RRF分数
        for rank, doc in enumerate(vector_docs):
            # 使用文档内容的确定性哈希作为唯一标识
            doc_id = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()
            doc_objects[doc_id] = doc

            # RRF公式: 1 / (k + rank)
            rrf_score = 1.0 / (k + rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score

            logger.debug(f"向量检索 - 文档{rank+1}: RRF分数 = {rrf_score:.4f}")

        # 计算BM25检索结果的RRF分数
        for rank, doc in enumerate(bm25_docs):
            doc_id = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()
            doc_objects[doc_id] = doc

            rrf_score = 1.0 / (k + rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score

            logger.debug(f"BM25检索 - 文档{rank+1}: RRF分数 = {rrf_score:.4f}")

        # 按最终RRF分数排序
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        # 构建最终结果
        reranked_docs = []
        for doc_id, final_score in sorted_docs:
            if doc_id in doc_objects:
                doc = doc_objects[doc_id]
                # 将RRF分数添加到文档元数据中
                doc.metadata['rrf_score'] = final_score
                reranked_docs.append(doc)
                logger.debug(f"最终排序 - 文档: {doc.page_content[:50]}... 最终RRF分数: {final_score:.4f}")

        logger.info(f"RRF重排完成: 向量检索{len(vector_docs)}个文档, BM25检索{len(bm25_docs)}个文档, 合并后{len(reranked_docs)}个文档")

        return reranked_docs
