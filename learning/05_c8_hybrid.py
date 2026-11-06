"""第八章第三步：对比向量、BM25 和 RRF，并观察中文预处理的影响。

先运行默认模式，再加 --tokenizer chinese。两次只改变 BM25 切词方式。
复用第四课保存的向量索引，在本地运行，不调用生成 API。
"""

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = REPO_ROOT / "learning" / "artifacts" / "c8_vector_index"
QUESTION = "西红柿炒鸡蛋怎么做？"
TOP_K = 3

sys.path.insert(0, str(REPO_ROOT / "code" / "C8"))
from rag_modules import DataPreparationModule, IndexConstructionModule, RetrievalOptimizationModule
from rag_modules.retrieval_optimization import chinese_bigram_tokenize


def show_results(label: str, documents) -> None:
    print(f"\n===== {label}：前 {len(documents)} 个结果 =====")
    for rank, document in enumerate(documents, start=1):
        print(
            f"{rank}. {document.metadata.get('dish_name', '未知')} | "
            f"{document.metadata.get('二级标题', '主标题或其他')}"
        )
        print(f"   {document.page_content.replace(chr(10), ' ')[:100]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="对比纯向量、BM25 与 RRF 融合检索")
    parser.add_argument(
        "--tokenizer", choices=["default", "chinese"], default="default",
        help="default：原有空格切分；chinese：中文相邻二字切分",
    )
    args = parser.parse_args()

    # 1. 固定原有资料、分块与向量索引，不重新编码文档。
    print("[1/3] 读取资料并加载已有向量索引……", flush=True)
    data_module = DataPreparationModule(str(REPO_ROOT / "data" / "C8" / "cook"))
    data_module.load_documents()
    chunks = data_module.chunk_documents()
    index_module = IndexConstructionModule(
        index_save_path=str(INDEX_DIR),
        cache_folder=str(REPO_ROOT / "models" / "huggingface"),
        local_files_only=True,
    )
    vectorstore = index_module.load_index()
    if vectorstore is None:
        raise SystemExit("没有加载到索引，请先运行 04_c8_index.py。")
    if vectorstore.index.ntotal != len(chunks):
        raise SystemExit("现有向量数量与本次分块数量不一致，请先检查索引与资料版本。")
    print(f"文本块：{len(chunks)}；已保存向量：{vectorstore.index.ntotal}")

    # 2. 两种模式只改变 BM25 的预处理函数；查询与文档使用相同函数。
    preprocess = chinese_bigram_tokenize if args.tokenizer == "chinese" else None
    print(f"[2/3] 创建检索器；切词模式：{args.tokenizer}", flush=True)
    retrieval = RetrievalOptimizationModule(
        vectorstore, chunks, bm25_preprocess_func=preprocess
    )
    query_tokens = retrieval.bm25_retriever.preprocess_func(QUESTION)
    scores = retrieval.bm25_retriever.vectorizer.get_scores(query_tokens)
    nonzero = int((scores != 0).sum())
    print(f"问题：{QUESTION}")
    print(f"BM25 问题词项：{query_tokens}")
    print(f"BM25 非零得分的文本块数量：{nonzero}")
    if nonzero == 0:
        print("本次 BM25 全部为零分：返回结果处于并列状态，不能据此判断相关性。")

    # 3. 两条检索路线各取五个候选，融合后展示前三个，与原 C8 设置一致。
    print("[3/3] 对比三种检索结果……", flush=True)
    vector_results = retrieval.vector_retriever.invoke(QUESTION)
    bm25_results = retrieval.bm25_retriever.invoke(QUESTION)
    hybrid_results = retrieval.hybrid_search(QUESTION, top_k=TOP_K)
    show_results("向量检索", vector_results[:TOP_K])
    show_results("BM25", bm25_results[:TOP_K])
    show_results("RRF 融合", hybrid_results)

    print("\n首位结果对照：")
    for label, documents in (
        ("向量", vector_results), ("BM25", bm25_results), ("融合", hybrid_results)
    ):
        print(f"  {label}：{documents[0].metadata.get('dish_name', '未知') if documents else '无结果'}")
    print("\n比较两次的词项、非零得分块数和首位菜谱。")
    print("菜名命中后还要核对正文是否有回答所需信息；一个问题不能代表整体检索效果。")


if __name__ == "__main__":
    main()
