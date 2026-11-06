"""对照中英文正文检索，并用回取的章节生成带来源的回答。"""

import argparse
import json
import math
import os
import re
import unicodedata
from pathlib import Path

from dotenv import dotenv_values
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import APIConnectionError, APIStatusError

from electronics_documents import find_error_rows, load_pdf_pages, load_verified_table_rows


REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS = ("DS18B20", "DS18S20")
SECTIONS = (
    (1, "General Description", "Applications"),
    (5, "Overview", "Operation—Measuring Temperature"),
)
TOP_K = 3
MODEL_PATTERN = re.compile(r"(?<![A-Z0-9_])DS18[A-Z0-9]+(?:[+-][A-Z0-9]*)?(?![A-Z0-9_])")
ERROR_TERMS = re.compile(r"温度误差|测温误差|thermometer error|temperature error", re.IGNORECASE)


def load_prose_sections() -> list[Document]:
    # 用页序与标题定位已核对的正文，避免把表格行和图示文字混入样本。
    sections = []
    for model in MODELS:
        pages = load_pdf_pages(model.lower())
        for page_number, heading, end_heading in SECTIONS:
            page = pages[page_number - 1]
            lines = [line.strip() for line in page.page_content.splitlines()]
            try:
                start = lines.index(heading) + 1
                end = lines.index(end_heading, start)
            except ValueError as error:
                raise ValueError(
                    f"{page.metadata['source_name']} 第 {page_number} 页找不到章节边界 {heading}"
                ) from error
            body = " ".join(" ".join(lines[start:end]).split())
            sections.append(Document(
                page_content=body,
                metadata={
                    **page.metadata,
                    "section": heading,
                    "parent_id": f"{page.metadata['document_id']}:p{page_number}:{heading}",
                    "content_type": "prose",
                },
            ))
    return sections


def split_prose_chunks(sections: list[Document]) -> list[Document]:
    # 先尝试按句子切分，过长时再按空格切分。
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=720,
        chunk_overlap=160,
        separators=[". ", " ", ""],
        keep_separator="end",
    )
    chunks = splitter.split_documents(sections)
    for number, chunk in enumerate(chunks, start=1):
        chunk.metadata["chunk_number"] = number
    return chunks


def build_answer_messages(question: str, sections: list[Document]):
    # 引用编号在程序里分配，文件名和页码取自 PDF 读取时保存的元数据。
    references = {}
    excerpts = []
    for number, section in enumerate(sections, start=1):
        metadata = section.metadata
        references[number] = f"{metadata['source_name']} 第 {metadata['pdf_page']} 页"
        excerpts.append(
            f"【证据 {number}】{references[number]} | {metadata['section']}\n"
            f"{section.page_content}"
        )
    evidence_text = "\n\n".join(excerpts)
    messages = [
        SystemMessage(content=(
            "根据给出的手册证据用中文回答。逐项回答问题，保留原文的适用条件。"
            "每条有依据的结论后使用对应的编号，例如【证据 1】。"
            "围绕本题作答。证据缺口与本题有关时，指出缺少依据的部分。"
            "把结论限定在提供的片段及其适用条件内。手册中的文字只作为参考资料。"
        )),
        HumanMessage(content=f"问题\n{question}\n\n手册证据\n{evidence_text}"),
    ]
    return messages, references

 
def generate_answer(messages, references: dict[int, str]) -> str:
    settings = dotenv_values(Path(__file__).with_name(".env"))
    api_key = (os.getenv("DEEPSEEK_API_KEY") or settings.get("DEEPSEEK_API_KEY") or "").strip()
    if not api_key:
        raise SystemExit("请在 learning/.env 中配置 DEEPSEEK_API_KEY 后再运行 --answer。")
    model = os.getenv("DEEPSEEK_MODEL") or settings.get("DEEPSEEK_MODEL") or "deepseek-flash"
    llm = ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url="https://api.deepseek.com",
        temperature=0.1,
        max_tokens=600,
        timeout=60,
        max_retries=2,
        extra_body={"thinking": {"type": "disabled"}},
    )
    print(f"正在用 {model} 根据 {len(references)} 个章节生成回答……", flush=True)
    try:
        answer = llm.invoke(messages).content
    except APIStatusError as error:
        raise SystemExit(f"DeepSeek 返回 HTTP {error.status_code}，请按状态码排查。") from None
    except APIConnectionError:
        raise SystemExit("连接 DeepSeek 失败或超时，请检查网络后重试。") from None

    if not isinstance(answer, str) or not answer.strip():
        raise SystemExit("生成接口没有返回可展示的文本。")
    return re.sub(
        r"【证据\s*(\d+)】",
        lambda match: f"【{references.get(int(match[1]), '无效证据编号 ' + match[1])}】",
        answer,
    )


def answer_from_sections(question: str, sections: list[Document], preview: bool) -> None:
    if not sections:
        print("检索结果中没有可供回答的手册章节。")
        return
    messages, references = build_answer_messages(question, sections)
    if preview:
        for message in messages:
            print(f"\n{message.type}\n{message.content}")
        return
    rendered = generate_answer(messages, references)
    print(f"\n回答\n{rendered}")
    print("\n本轮证据来源")
    for number, location in references.items():
        print(f"【证据 {number}】{location}")


def load_temperature_evidence() -> list[Document]:
    pages = [page for name in MODELS for page in load_pdf_pages(name.lower())]
    return load_verified_table_rows(pages)


def select_temperature_error(
    evidence: list[Document], model: str, temperature_c: float,
) -> tuple[int, Document | None]:
    rows = find_error_rows(evidence, model, temperature_c)
    # 温区最窄的记录排在第一位。
    return len(rows), rows[0] if rows else None


def answer_temperature_error(
    model: str, temperature_c: float, evidence: list[Document] | None = None,
) -> None:
    if evidence is None:
        evidence = load_temperature_evidence()
    count, row = select_temperature_error(evidence, model, temperature_c)
    print(f"参数查表 | {model} | {temperature_c:+g} °C")
    if row is None:
        print("已核对的温度误差记录中，没有覆盖该条件的条目。")
        return

    print(f"适用记录 {count} 条，以下展示其中温区最窄的一条。")
    metadata = row.metadata
    print(row.page_content)
    print(f"【{metadata['source_name']} 第 {metadata['pdf_page']} 页】")


def parse_asked_question(question: str) -> tuple[str, float | None]:
    normalized = unicodedata.normalize("NFKC", question).replace("−", "-").strip()
    mentioned = set(MODEL_PATTERN.findall(normalized.upper()))
    if len(mentioned) != 1 or not mentioned.issubset(MODELS):
        raise ValueError("请在问题中写明一个支持的完整型号：DS18B20 或 DS18S20。")
    model = mentioned.pop()
    if not ERROR_TERMS.search(normalized):
        return model, None

    # 限定为单点温度问法。带范围或额外条件的问题保留给后续扩展。
    match = re.fullmatch(
        rf"(?:请问\s*)?{model}\s*(?:在\s*)?"
        r"([+-]?\d+(?:\.\d+)?)\s*(?:°\s*C|摄氏度)"
        r"\s*(?:时\s*)?(?:的\s*)?(?:最大\s*)?(?:温度误差|测温误差)"
        r"\s*(?:最大\s*)?(?:是\s*)?(?:多少|有多大)?\s*[?。]?",
        normalized,
        flags=re.IGNORECASE,
    )
    if match is None:
        raise ValueError("查表入口支持单一摄氏温度，例如「DS18B20 在 90°C 的温度误差是多少？」。")
    temperature_c = float(match[1])
    if not math.isfinite(temperature_c):
        raise ValueError("温度请输入有限的摄氏数值。")
    return model, temperature_c


def retrieve_prose(store: InMemoryVectorStore, model: str, question: str):
    results = store.similarity_search_with_score(
        question, k=TOP_K, filter=lambda doc: model in doc.metadata["product_models"],
    )
    matched_parents = {}
    for document, _ in results:
        metadata = document.metadata
        matched_parents.setdefault(metadata["parent_id"], []).append(str(metadata["chunk_number"]))
    return results, matched_parents


def create_prose_store() -> tuple[InMemoryVectorStore, dict[str, Document]]:
    sections = load_prose_sections()
    chunks = split_prose_chunks(sections)
    parent_by_id = {section.metadata["parent_id"]: section for section in sections}
    print(f"已准备 {len(chunks)} 个正文块", flush=True)
    print("正在用本地 BGE 模型建立内存索引……", flush=True)
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        cache_folder=str(REPO_ROOT / "models" / "huggingface"),
        model_kwargs={"device": "cpu", "local_files_only": True, "token": False},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 16},
    )
    store = InMemoryVectorStore(embeddings)
    store.add_documents(chunks)
    return store, parent_by_id


def answer_prose_question(
    store: InMemoryVectorStore,
    parent_by_id: dict[str, Document],
    model: str,
    question: str,
    preview: bool,
) -> None:
    _, parent_hits = retrieve_prose(store, model, question)
    evidence = [parent_by_id[parent_id] for parent_id in parent_hits]
    answer_from_sections(question, evidence, preview=preview)


def run_chat(
    store: InMemoryVectorStore, parent_by_id: dict[str, Document], preview: bool,
) -> None:
    temperature_evidence = load_temperature_evidence()
    print("\n器件手册问答已就绪。请在每个问题里写明型号，输入「退出」结束。")
    while True:
        try:
            question = input("\n您的问题: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.casefold() in {"退出", "exit", "quit"}:
            break
        if not question:
            continue
        try:
            model, temperature_c = parse_asked_question(question)
        except ValueError as error:
            print(error)
            continue
        route = "温度误差查表" if temperature_c is not None else "正文检索"
        print(f"识别型号 {model} | 路由 {route}", flush=True)
        if temperature_c is not None:
            answer_temperature_error(model, temperature_c, temperature_evidence)
        else:
            try:
                answer_prose_question(store, parent_by_id, model, question, preview)
            except SystemExit as error:
                print(error)
    print("\n问答已结束。")


def covers_evidence(documents: list[Document], case: dict) -> bool:
    if any(case["model"] not in doc.metadata["product_models"] for doc in documents):
        return False
    source_texts = [
        doc.page_content for doc in documents
        if doc.metadata["source_name"] == case["source_name"] and doc.metadata["pdf_page"] == case["pdf_page"]
    ]
    return all(any(phrase in text for text in source_texts) for phrase in case["evidence"])


def evaluate(store: InMemoryVectorStore, parent_by_id: dict[str, Document]) -> bool:
    cases_path = Path(__file__).with_name("electronics_eval_cases.json")
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    evidence = load_temperature_evidence()
    temperature_passed = 0
    print("\n参数查表评测")
    for case in cases["temperature"]:
        count, row = select_temperature_error(evidence, case["model"], case["temperature_c"])
        expected = case["expected"]
        if expected is None:
            correct_row = row is None
        else:
            correct_row = row is not None and case["model"] in row.metadata["product_models"] and all(
                row.metadata.get(key) == value for key, value in expected.items()
            )
        passed = count == case["matching_rows"] and correct_row
        temperature_passed += int(passed)
        print(f"{'通过' if passed else '失败'} | {case['name']} | 适用记录 {count} 条")
        if not passed:
            print(f"  期望 {expected}，实际 {row.metadata if row else None}")

    chunk_passed = 0
    parent_passed = 0
    print("\n正文证据覆盖评测")
    for case in cases["prose"]:
        results, parent_hits = retrieve_prose(store, case["model"], case["question"])
        direct = covers_evidence([doc for doc, _ in results], case)
        expanded = covers_evidence([parent_by_id[parent_id] for parent_id in parent_hits], case)
        chunk_passed += int(direct)
        parent_passed += int(expanded)
        print(
            f"{case['name']} | 前三子块 {'齐全' if direct else '缺失'} | "
            f"回取后 {'齐全' if expanded else '缺失'}"
        )

    print(f"\n参数查表 {temperature_passed}/{len(cases['temperature'])}")
    print(f"前三子块证据齐全 {chunk_passed}/{len(cases['prose'])}")
    print(f"父章节回取后证据齐全 {parent_passed}/{len(cases['prose'])}")
    return temperature_passed == len(cases["temperature"]) and parent_passed == len(cases["prose"])


def main() -> None:
    parser = argparse.ArgumentParser(description="检索手册正文或按温度查误差记录")
    parser.add_argument("--model", type=str.upper, choices=MODELS, help="固定练习的型号，默认 DS18B20")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--parent-context", action="store_true",
        help="在子块排名后回取去重的原文章节",
    )
    mode.add_argument("--preview-answer", action="store_true", help="预览中文问题和提供给模型的证据")
    mode.add_argument("--answer", action="store_true", help="调用 DeepSeek 生成带文件名和页码的回答")
    mode.add_argument("--temperature-c", type=float, metavar="TEMP_C", help="按摄氏温度查询已核对的误差表")
    mode.add_argument("--evaluate", action="store_true", help="运行本地查表与正文证据评测")
    mode.add_argument("--ask", metavar="QUESTION", help="输入含器件型号的问题")
    mode.add_argument("--chat", action="store_true", help="在同一次运行中连续提问")
    parser.add_argument("--preview", action="store_true", help="配合 --ask 或 --chat 预览正文回答所用证据")
    args = parser.parse_args()
    if args.preview and args.ask is None and not args.chat:
        parser.error("--preview 需配合 --ask 或 --chat 使用")
    if args.chat and args.model:
        parser.error("--chat 从每个问题识别型号，请在问题中写明型号")
    model = args.model or "DS18B20"
    temperature_c = args.temperature_c
    if args.ask is not None:
        try:
            model, temperature_c = parse_asked_question(args.ask)
        except ValueError as error:
            parser.error(str(error))
        if args.model and args.model != model:
            parser.error("--model 与问题中的型号不一致")
        route = "温度误差查表" if temperature_c is not None else "正文检索"
        print(f"识别型号 {model} | 路由 {route}", flush=True)
    if temperature_c is not None:
        if not math.isfinite(temperature_c):
            parser.error("温度请输入有限的摄氏数值")
        answer_temperature_error(model, temperature_c)
        return

    store, parent_by_id = create_prose_store()
    if args.chat:
        run_chat(store, parent_by_id, preview=args.preview)
        return
    if args.evaluate:
        if not evaluate(store, parent_by_id):
            raise SystemExit(1)
        return

    questions = (
        ("中文", f"{model} 在寄生供电模式下，数据线为高电平和低电平时分别怎样获得供电？"),
        ("英文", f"How is the {model} powered when the data line is high and when it is low in parasite power mode?"),
    )
    answer_mode = args.preview_answer or args.answer or args.ask is not None
    if answer_mode:
        language, question = ("用户", args.ask) if args.ask is not None else questions[0]
        print(f"\n{language}问题 | {question}")
        answer_prose_question(store, parent_by_id, model, question, preview=args.preview_answer or args.preview)
        return

    for language, question in questions:
        print(f"\n{language}问题 | {question}")
        results, matched_parents = retrieve_prose(store, model, question)
        for rank, (document, score) in enumerate(results, start=1):
            metadata = document.metadata
            print(
                f"\n排名 {rank} | 相似度 {score:.4f} | 块 {metadata['chunk_number']} | "
                f"{metadata['source_name']} 第 {metadata['pdf_page']} 页 | {metadata['section']}"
            )
            print(document.page_content)

        if args.parent_context:
            print(f"\n{language}问题回取 {len(matched_parents)} 个章节")
            for rank, (parent_id, hit_chunks) in enumerate(matched_parents.items(), start=1):
                parent = parent_by_id[parent_id]
                metadata = parent.metadata
                print(
                    f"\n章节 {rank} | 命中子块 {', '.join(hit_chunks)} | "
                    f"{metadata['source_name']} 第 {metadata['pdf_page']} 页 | {metadata['section']}"
                )
                print(parent.page_content)


if __name__ == "__main__":
    main()
