"""菜名约束回归检查：使用真实本地索引，禁止发送远端 HTTP 请求。"""

import contextlib
import io
import logging
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code" / "C8"))
from main import RecipeRAGSystem


class RecipeConstraintTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        network = patch("httpx.Client.send", side_effect=AssertionError("Remote HTTP is disabled in regression checks"))
        cls.http_send = network.start()
        cls.addClassCleanup(network.stop)
        environment = patch.dict(os.environ, {"DEEPSEEK_API_KEY": "local-test-placeholder"})
        environment.start()
        cls.addClassCleanup(environment.stop)
        previous_logging = logging.root.manager.disable
        logging.disable(logging.CRITICAL)
        cls.addClassCleanup(logging.disable, previous_logging)
        with contextlib.redirect_stdout(io.StringIO()):
            cls.system = RecipeRAGSystem()
            cls.system.initialize_system()
            cls.system.build_knowledge_base()
        cls.system.index_module.embeddings.show_progress = False

    def search(self, question):
        filters = self.system._extract_filters_from_query(question)
        chunks = self.system.retrieval_module.metadata_filtered_search(question, filters, top_k=3)
        return filters, chunks, self.system.data_module.get_parent_documents(chunks)

    def test_interfering_ingredient_keeps_target_recipe(self):
        filters, chunks, parents = self.search("西红柿炒鸡蛋里一定要加入西葫芦吗？")
        self.assertEqual(filters, {"dish_name": "西红柿炒鸡蛋"})
        self.assertTrue(chunks)
        self.assertEqual({doc.metadata["dish_name"] for doc in chunks}, {"西红柿炒鸡蛋"})
        self.assertEqual([doc.metadata["dish_name"] for doc in parents], ["西红柿炒鸡蛋"])
        self.assertIn("## 必备原料和工具", parents[0].page_content)

    def test_missing_fact_question_keeps_target_recipe(self):
        _, chunks, parents = self.search("这份西红柿炒鸡蛋的精确热量是多少千卡？")
        self.assertTrue(chunks)
        self.assertEqual([doc.metadata["dish_name"] for doc in parents], ["西红柿炒鸡蛋"])
        self.assertNotIn("热量", parents[0].page_content)

    def test_regular_question_still_retrieves_target(self):
        _, chunks, parents = self.search("西红柿炒鸡蛋怎么做？")
        self.assertTrue(chunks)
        self.assertEqual([doc.metadata["dish_name"] for doc in parents], ["西红柿炒鸡蛋"])

    def test_named_attribute_question_does_not_filter_out_the_answer(self):
        filters = self.system._extract_filters_from_query("西红柿炒鸡蛋是不是很困难的荤菜？")
        self.assertEqual(filters, {"dish_name": "西红柿炒鸡蛋"})

    def test_explicit_similar_recommendation_does_not_lock_reference_recipe(self):
        filters = self.system._extract_filters_from_query("推荐类似西红柿炒鸡蛋的简单素菜")
        self.assertEqual(filters, {"category": "素菜", "difficulty": "简单"})

    def test_comparison_keeps_multiple_explicit_names(self):
        names = {"西红柿炒鸡蛋", "西葫芦炒鸡蛋"}
        filters, chunks, _ = self.search("西红柿炒鸡蛋和西葫芦炒鸡蛋有什么区别？")
        self.assertEqual(set(filters["dish_name"]), names)
        self.assertTrue(chunks)
        self.assertTrue({doc.metadata["dish_name"] for doc in chunks}.issubset(names))

    def test_longest_names_and_separate_occurrences(self):
        system = object.__new__(RecipeRAGSystem)
        system.data_module = SimpleNamespace(documents=[
            Document(page_content="", metadata={"dish_name": name})
            for name in ["炒鸡蛋", "西红柿炒鸡蛋", "西葫芦炒鸡蛋"]
        ])
        self.assertEqual(system._extract_dish_names_from_query("西红柿炒鸡蛋怎么做？"), ["西红柿炒鸡蛋"])
        self.assertEqual(system._extract_dish_names_from_query("西红柿炒鸡蛋和炒鸡蛋有什么区别？"), ["西红柿炒鸡蛋", "炒鸡蛋"])

    def test_target_survives_an_unrelated_global_ranking(self):
        chunks = self.system.retrieval_module.metadata_filtered_search(
            "蒸鸡蛋羹怎么做", {"dish_name": "西红柿炒鸡蛋"}, top_k=3
        )
        self.assertTrue(chunks)
        self.assertEqual({doc.metadata["dish_name"] for doc in chunks}, {"西红柿炒鸡蛋"})

    def test_category_filters_and_empty_scope(self):
        filters = self.system._extract_filters_from_query("推荐几个简单的素菜")
        self.assertEqual(filters, {"category": "素菜", "difficulty": "简单"})
        chunks = self.system.retrieval_module.metadata_filtered_search("推荐几个简单的素菜", filters, top_k=3)
        self.assertTrue(chunks)
        self.assertTrue(all(doc.metadata["category"] == "素菜" and doc.metadata["difficulty"] == "简单" for doc in chunks))
        self.assertEqual(self.system.retrieval_module.metadata_filtered_search("做法", {"dish_name": "不存在的菜谱"}), [])

    def test_main_passes_correct_evidence_to_generation_chain(self):
        question = "西红柿炒鸡蛋里一定要加入西葫芦吗？"
        self.system.generation_module.llm = FakeListChatModel(responses=["general", question, "LOCAL_PIPELINE_TEST_ONLY"])
        observed = []
        original = self.system.generation_module.generate_basic_answer

        def capture(query, documents):
            observed.extend(doc.metadata["dish_name"] for doc in documents)
            return original(query, documents)

        with patch.object(self.system.generation_module, "generate_basic_answer", side_effect=capture), contextlib.redirect_stdout(io.StringIO()):
            answer = self.system.ask_question(question, stream=False)
        self.assertEqual(answer, "LOCAL_PIPELINE_TEST_ONLY")
        self.assertEqual(observed, ["西红柿炒鸡蛋"])
        self.http_send.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
