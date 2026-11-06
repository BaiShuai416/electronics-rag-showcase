"""
RAG系统主程序
"""

import os
import sys
import logging
import re
from pathlib import Path

# 添加模块路径
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
from config import DEFAULT_CONFIG, PROJECT_ROOT, RAGConfig
from rag_modules import (
    DataPreparationModule,
    IndexConstructionModule,
    RetrievalOptimizationModule,
    GenerationIntegrationModule
)
from rag_modules.retrieval_optimization import chinese_bigram_tokenize
from openai import APIConnectionError, APIStatusError

# 加载环境变量
# 环境变量优先；其次读取 C8、仓库根目录与前面练习使用的本地配置。
for env_file in (
    Path(__file__).with_name(".env"),
    PROJECT_ROOT / ".env",
    PROJECT_ROOT / "learning" / ".env",
):
    load_dotenv(env_file, override=False)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def describe_api_failure(error: APIStatusError) -> str:
    """按状态码显示排查方向，避免输出原始认证错误正文。"""
    hints = {
        400: "请求格式有误，请核对生成接口参数。",
        401: "身份验证失败，请在本地检查 DeepSeek 密钥。",
        402: "账户额度不足，请到官方平台检查可用额度。",
        422: "请求参数不被接受，请核对模型接口要求。",
        429: "请求频率受限，请降低频率后再试。",
        503: "生成服务暂时过载，请稍后再试。",
    }
    hint = hints.get(error.status_code, "生成服务请求失败，请稍后重试或核对官方错误说明。")
    message = f"HTTP {error.status_code}：{hint}"
    request_id = getattr(error, "request_id", None)
    if request_id:
        message += f" 请求编号：{request_id}"
    return message


class RecipeRAGSystem:
    """食谱RAG系统主类"""

    def __init__(self, config: RAGConfig = None, *, require_api_key: bool = True):
        """
        初始化RAG系统

        Args:
            config: RAG系统配置，默认使用DEFAULT_CONFIG
        """
        self.config = config if config is not None else RAGConfig.from_dict(DEFAULT_CONFIG.to_dict())
        if config is None:
            self.config.llm_model = os.getenv("DEEPSEEK_MODEL", "").strip() or self.config.llm_model
        self.data_module = None
        self.index_module = None
        self.retrieval_module = None
        self.generation_module = None

        # 检查数据路径
        if not Path(self.config.data_path).exists():
            raise FileNotFoundError(f"数据路径不存在: {self.config.data_path}")

        # 检查API密钥
        if require_api_key and not os.getenv("DEEPSEEK_API_KEY", "").strip():
            raise ValueError("请在本地 learning/.env 或 code/C8/.env 中配置 DEEPSEEK_API_KEY")
    
    def initialize_system(self, *, with_generation: bool = True):
        """初始化所有模块"""
        print("🚀 正在初始化RAG系统...")

        # 1. 初始化数据准备模块
        print("初始化数据准备模块...")
        self.data_module = DataPreparationModule(self.config.data_path)

        # 2. 初始化索引构建模块
        print("初始化索引构建模块...")
        self.index_module = IndexConstructionModule(
            model_name=self.config.embedding_model,
            index_save_path=self.config.index_save_path,
            cache_folder=self.config.embedding_cache_folder,
            local_files_only=self.config.embedding_local_files_only,
            show_progress=True,
        )

        if with_generation:
            self.initialize_generation()
        print("✅ 系统初始化完成！")

    def initialize_generation(self):
        """按需创建回答生成模块，证据预览可以独立使用检索器。"""
        print("🤖 初始化生成集成模块...")
        self.generation_module = GenerationIntegrationModule(
            model_name=self.config.llm_model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )

    def build_knowledge_base(self):
        """构建知识库"""
        print("\n正在构建知识库...")

        # 1. 尝试加载已保存的索引
        vectorstore = self.index_module.load_index()

        if vectorstore is not None:
            print("✅ 成功加载已保存的向量索引！")
            # 仍需要加载文档和分块用于检索模块
            print("加载食谱文档...")
            self.data_module.load_documents()
            print("进行文本分块...")
            chunks = self.data_module.chunk_documents()
        else:
            print("未找到已保存的索引，开始构建新索引...")

            # 2. 加载文档
            print("加载食谱文档...")
            self.data_module.load_documents()

            # 3. 文本分块
            print("进行文本分块...")
            chunks = self.data_module.chunk_documents()

            # 4. 构建向量索引
            print("构建向量索引...")
            vectorstore = self.index_module.build_vector_index(chunks)

            # 5. 保存索引
            print("保存向量索引...")
            self.index_module.save_index()

        # 6. 初始化检索优化模块
        print("初始化检索优化...")
        preprocess = chinese_bigram_tokenize if self.config.use_chinese_bm25 else None
        self.retrieval_module = RetrievalOptimizationModule(
            vectorstore, chunks, bm25_preprocess_func=preprocess
        )
        print(f"BM25 预处理：{'中文相邻二字切分' if preprocess else '空格切分'}")

        # 7. 显示统计信息
        stats = self.data_module.get_statistics()
        print(f"\n📊 知识库统计:")
        print(f"   文档总数: {stats['total_documents']}")
        print(f"   文本块数: {stats['total_chunks']}")
        print(f"   菜品分类: {list(stats['categories'].keys())}")
        print(f"   难度分布: {stats['difficulties']}")

        print("✅ 知识库构建完成！")
    
    def retrieve_chunks(self, question: str, search_query: str):
        filters = self._extract_filters_from_query(question)
        if filters:
            print(f"应用过滤条件: {filters}")
            return self.retrieval_module.metadata_filtered_search(search_query, filters, top_k=self.config.top_k)
        return self.retrieval_module.hybrid_search(search_query, top_k=self.config.top_k)

    def ask_question(self, question: str, stream: bool = False, *, on_evidence=None):
        """
        回答用户问题

        Args:
            question: 用户问题
            stream: 是否使用流式输出

        Returns:
            生成的回答或生成器
        """
        if not all([self.retrieval_module, self.generation_module]):
            raise ValueError("请先构建知识库")
        
        print(f"\n❓ 用户问题: {question}")

        # 1. 查询路由
        route_type = self.generation_module.query_router(question)
        print(f"🎯 查询类型: {route_type}")

        # 2. 智能查询重写（根据路由类型）
        if route_type == 'list':
            # 列表查询保持原查询
            rewritten_query = question
            print(f"📝 列表查询保持原样: {question}")
        else:
            # 详细查询和一般查询使用智能重写
            print("🤖 智能分析查询...")
            rewritten_query = self.generation_module.query_rewrite(question)
        
        # 3. 检索相关子块（自动应用元数据过滤）
        print("🔍 检索相关文档...")
        relevant_chunks = self.retrieve_chunks(question, rewritten_query)

        # 显示检索到的子块信息
        if relevant_chunks:
            chunk_info = []
            for chunk in relevant_chunks:
                dish_name = chunk.metadata.get('dish_name', '未知菜品')
                # 尝试从内容中提取章节标题
                content_preview = chunk.page_content[:100].strip()
                if content_preview.startswith('#'):
                    # 如果是标题开头，提取标题（仅取第一行）
                    title_end = content_preview.find('\n') if '\n' in content_preview else len(content_preview)
                    section_title = content_preview[:title_end].replace('#', '').strip()
                    chunk_info.append(f"{dish_name}({section_title})")
                else:
                    chunk_info.append(f"{dish_name}(内容片段)")

            print(f"找到 {len(relevant_chunks)} 个相关文档块: {', '.join(chunk_info)}")
        else:
            print(f"找到 {len(relevant_chunks)} 个相关文档块")

        # 4. 检查是否找到相关内容
        if not relevant_chunks:
            return "抱歉，没有找到相关的食谱信息。请尝试其他菜品名称或关键词。"

        # 5. 复用同一份父文档构建回答和来源列表。
        relevant_docs = self.data_module.get_parent_documents(relevant_chunks)
        if on_evidence is not None:
            on_evidence(relevant_docs)

        if route_type == 'list':
            # 列表查询：直接返回菜品名称列表
            print("📋 生成菜品列表...")
            # 显示找到的文档名称
            doc_names = []
            for doc in relevant_docs:
                dish_name = doc.metadata.get('dish_name', '未知菜品')
                doc_names.append(dish_name)

            if doc_names:
                print(f"找到文档: {', '.join(doc_names)}")

            return self.generation_module.generate_list_answer(question, relevant_docs)
        else:
            # 详细查询：获取完整文档并生成详细回答
            print("获取完整文档...")
            # 显示找到的文档名称
            doc_names = []
            for doc in relevant_docs:
                dish_name = doc.metadata.get('dish_name', '未知菜品')
                doc_names.append(dish_name)

            if doc_names:
                print(f"找到文档: {', '.join(doc_names)}")
            else:
                print(f"对应 {len(relevant_docs)} 个完整文档")

            print("✍️ 生成分步骤回答..." if route_type == "detail" else "✍️ 生成一般回答...")

            # 根据路由类型自动选择回答模式
            if route_type == "detail":
                # 详细查询使用分步指导模式
                if stream:
                    return self.generation_module.generate_step_by_step_answer_stream(question, relevant_docs)
                else:
                    return self.generation_module.generate_step_by_step_answer(question, relevant_docs)
            else:
                # 一般查询使用基础回答模式
                if stream:
                    return self.generation_module.generate_basic_answer_stream(question, relevant_docs)
                else:
                    return self.generation_module.generate_basic_answer(question, relevant_docs)
    
    def _extract_dish_names_from_query(self, query: str) -> list:
        """匹配已加载资料的完整菜名，优先选择不重叠的长名称。"""
        if self.data_module is None:
            return []
        names = {
            doc.metadata.get('dish_name')
            for doc in self.data_module.documents
            if doc.metadata.get('dish_name')
        }
        candidates = []
        for name in names:
            for match in re.finditer(re.escape(name), query):
                candidates.append((match.start(), match.end(), name))

        selected = []
        for start, end, name in sorted(candidates, key=lambda item: (-(item[1] - item[0]), item[0], item[2])):
            if not any(start < other_end and other_start < end for other_start, other_end, _ in selected):
                selected.append((start, end, name))
        # 两个完整菜名可以同时保留；同一名称重复出现时只保留一次。
        return list(dict.fromkeys(name for _, _, name in sorted(selected)))

    def _extract_filters_from_query(self, query: str) -> dict:
        """
        从用户问题中提取元数据过滤条件
        """
        dish_names = self._extract_dish_names_from_query(query)
        similar_recommendation = (
            any(word in query for word in ('类似', '相似', '差不多'))
            and any(word in query for word in ('推荐', '还有', '其他', '其它'))
        ) or ('除了' in query and '推荐' in query)
        if dish_names and not similar_recommendation:
            # 具体菜谱问题优先固定对象；问题中的分类/难度可能是待确认的属性。
            return {'dish_name': dish_names[0] if len(dish_names) == 1 else dish_names}

        filters = {}
        # 分类关键词
        category_keywords = DataPreparationModule.get_supported_categories()
        for cat in category_keywords:
            if cat in query:
                filters['category'] = cat
                break

        # 难度关键词
        difficulty_keywords = DataPreparationModule.get_supported_difficulties()
        for diff in sorted(difficulty_keywords, key=len, reverse=True):
            if diff in query:
                filters['difficulty'] = diff
                break

        return filters
    
    def run_interactive(self):
        """运行交互式问答"""
        print("=" * 60)
        print("🍽️  尝尝咸淡RAG系统 - 交互式问答  🍽️")
        print("=" * 60)
        print("💡 解决您的选择困难症，告别'今天吃什么'的世纪难题！")
        print(f"生成服务：DeepSeek；模型：{self.config.llm_model}")
        
        # 初始化系统
        self.initialize_system()
        
        # 构建知识库
        self.build_knowledge_base()
        
        print("\n交互式问答 (输入'退出'结束):")
        
        while True:
            try:
                user_input = input("\n您的问题: ").strip()
                if user_input.lower() in ['退出', 'quit', 'exit', '']:
                    break
                
                # 询问是否使用流式输出
                stream_choice = input("是否使用流式输出? (y/n, 默认y): ").strip().lower()
                use_stream = stream_choice != 'n'

                print("\n回答:")
                if use_stream:
                    # 流式输出
                    for chunk in self.ask_question(user_input, stream=True):
                        print(chunk, end="", flush=True)
                    print("\n")
                else:
                    # 普通输出
                    answer = self.ask_question(user_input, stream=False)
                    print(f"{answer}\n")
                
            except KeyboardInterrupt:
                break
            except APIStatusError as e:
                print(f"生成请求失败：{describe_api_failure(e)}")
            except APIConnectionError:
                print("生成服务连接失败或等待超时，请检查网络或稍后重试。")
            except Exception as e:
                print(f"处理问题时出错: {e}")
        
        print("\n感谢使用尝尝咸淡RAG系统！")


def main():
    """主函数"""
    try:
        # 创建RAG系统
        rag_system = RecipeRAGSystem()
        
        # 运行交互式问答
        rag_system.run_interactive()
        
    except Exception as e:
        logger.error(f"系统运行出错: {e}")
        print(f"系统错误: {e}")
        raise SystemExit(1) from None

if __name__ == "__main__":
    main()
