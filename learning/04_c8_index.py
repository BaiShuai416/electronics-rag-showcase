"""第八章第二步：首次建立并保存 FAISS 索引，后续加载并检索。

直接调用 C8 的数据与索引模块，复用本地 Embedding 模型，不调用生成 API。
本课以当前固定版本的资料、分块方式和模型为基础，尚未实现自动检测资料变更。
"""

import sys
from pathlib import Path
from time import perf_counter


REPO_ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = REPO_ROOT / "learning" / "artifacts" / "c8_vector_index"
QUESTION = "西红柿炒鸡蛋怎么做？"

sys.path.insert(0, str(REPO_ROOT / "code" / "C8"))
from rag_modules import DataPreparationModule, IndexConstructionModule


def main() -> None:
    # 1. 读取父文档并按标题切分，供首次建库以及后续父文档回取使用。
    data_module = DataPreparationModule(str(REPO_ROOT / "data" / "C8" / "cook"))
    documents = data_module.load_documents()
    chunks = data_module.chunk_documents()
    print(f"[1/4] 父文档：{len(documents)}；子块：{len(chunks)}", flush=True)

    # 2. 加载编码模型。加载模型的时间与下面建立/加载索引的时间分别统计。
    print("[2/4] 从本地缓存加载 Embedding 模型……", flush=True)
    model_started = perf_counter()
    index_module = IndexConstructionModule(
        model_name="BAAI/bge-small-zh-v1.5",
        index_save_path=str(INDEX_DIR),
        cache_folder=str(REPO_ROOT / "models" / "huggingface"),
        local_files_only=True,
        show_progress=True,
    )
    print(f"模型加载用时：{perf_counter() - model_started:.2f} 秒", flush=True)

    # 3. 先尝试读取现有索引。没有可用索引时才编码全部子块并保存。
    index_started = perf_counter()
    vectorstore = index_module.load_index()
    if vectorstore is None:
        print("[3/4] 未加载到已有索引：开始为全部子块生成向量并保存……", flush=True)
        vectorstore = index_module.build_vector_index(chunks)
        index_module.save_index()
        mode = "新建并保存"
    else:
        print("[3/4] 已加载保存的索引，跳过全部文档块的向量化。", flush=True)
        mode = "加载已有索引"

    print(f"本次索引模式：{mode}")
    print(f"索引处理用时：{perf_counter() - index_started:.2f} 秒")
    print(f"索引中的向量数量：{vectorstore.index.ntotal}")
    print(f"每个向量的维度：{vectorstore.index.d}")
    print(f"保存位置：{INDEX_DIR}")
    for file_name in ("index.faiss", "index.pkl"):
        file_path = INDEX_DIR / file_name
        print(f"  {file_name}：{file_path.stat().st_size} 字节")

    # 4. 查询仍需为当前问题生成向量，再使用已有的文档向量搜索。
    print(f"\n[4/4] 检索问题：{QUESTION}", flush=True)
    results = index_module.similarity_search(QUESTION, k=3)
    for rank, chunk in enumerate(results, start=1):
        print(f"\n结果 {rank} | 菜谱：{chunk.metadata.get('dish_name', '未知')}")
        print(f"章节：{chunk.metadata.get('二级标题', '主标题或其他')}")
        print(f"parent_id：{chunk.metadata.get('parent_id')}")
        print(chunk.page_content[:180])

    parents = data_module.get_parent_documents(results)
    print(f"\n根据检索子块找回的完整原文数量：{len(parents)}")
    print(f"原文名称：{[doc.metadata.get('dish_name') for doc in parents]}")
    print("\n再次运行同一命令，对比索引模式与索引处理用时。")
    print("第二次仍需加载编码模型、编码新问题；已保存的文档向量可以直接复用。")


if __name__ == "__main__":
    main()
