"""电子器件练习共用的 PDF 来源与逐页读取函数。"""

import json
from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader


DATA_DIR = Path(__file__).resolve().parent / "迁移资料"
VERIFIED_ROWS_PATH = Path(__file__).with_name("electronics_verified_rows.json")
SOURCES = {
    "ds18b20": {
        "source_name": "DS18B20.pdf",
        "product_models": ["DS18B20"],
        "document_version": "Rev. 6",
        "source_url": "https://www.analog.com/en/products/ds18b20.html",
    },
    "ds18s20": {
        "source_name": "DS18S20.pdf",
        "product_models": ["DS18S20"],
        "document_version": "Rev. 3",
        "source_url": "https://www.analog.com/en/products/ds18s20.html",
    },
    "comparison": {
        "source_name": "Comparison of the DS18B20 and DS18S20 1-Wire® Digital Thermometers _ Analog Devices.pdf",
        "product_models": ["DS18B20", "DS18S20"],
        "document_version": None,
        "source_url": "https://www.analog.com/en/resources/design-notes/1wire-protocol-pdf-of-ds18s20-vs-ds18b20-digital-themormeters.html",
    },
}


def load_pdf_pages(source_id: str) -> list[Document]:
    """返回一页一个 Document，空文本页保留页码，便于发现提取问题。"""
    source = SOURCES[source_id]
    reader = PdfReader(DATA_DIR / source["source_name"])
    return [
        Document(
            page_content=(page.extract_text() or "").strip(),
            metadata={**source, "document_id": source_id, "pdf_page": number},
        )
        for number, page in enumerate(reader.pages, start=1)
    ]


def load_verified_table_rows(pages: list[Document]) -> list[Document]:
    """把人工核对的表格行转换为带来源的证据块。"""
    page_metadata = {
        (doc.metadata["document_id"], doc.metadata["pdf_page"]): doc.metadata
        for doc in pages
    }
    tables = json.loads(VERIFIED_ROWS_PATH.read_text(encoding="utf-8"))
    evidence = []
    for table in tables:
        metadata = page_metadata[(table["document_id"], table["pdf_page"])]
        for row in table["rows"]:
            text = (
                f"{', '.join(metadata['product_models'])} | {table['table']}\n"
                f"{table['parameter']} ({table['symbol']}) | "
                f"{row['temperature_min_c']:+g} to {row['temperature_max_c']:+g} °C | "
                f"{table['column']} ±{row['max_abs_error']:g} {table['unit']}\n"
                f"{table['table_conditions']} {table['note']}"
            )
            evidence.append(Document(
                page_content=text,
                metadata={**metadata, **row, "content_type": "verified_table_row"},
            ))
    return evidence


def find_error_rows(evidence: list[Document], model: str, temperature_c: float) -> list[Document]:
    """保留型号和温度都适用的记录，较窄区间先展示。"""
    matching = []
    for document in evidence:
        metadata = document.metadata
        if model in metadata["product_models"] and (
            metadata["temperature_min_c"] <= temperature_c <= metadata["temperature_max_c"]
        ):
            matching.append(document)
    return sorted(
        matching,
        key=lambda doc: doc.metadata["temperature_max_c"] - doc.metadata["temperature_min_c"],
    )
