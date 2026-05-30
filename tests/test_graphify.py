"""
test_graphify.py
Verification test script for the integrated Photocatalyst Graphify system.
"""

import os
import sys
import unittest

# Reconfigure encoding for Windows console support
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Add src to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from src.graphify import run_pipeline
from src.graphify.rag import GraphRAG
from src.graphify.explain import PredictExplainer
from src.graphify.discovery import DiscoveryEngine

class TestPhotocatalystGraphify(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Run pipeline once to create graph outputs
        print("Running Graphify pipeline for verification tests...")
        cls.G = run_pipeline(root_dir=".", force=True)
        cls.rag = GraphRAG()
        cls.explainer = PredictExplainer()
        cls.discovery = DiscoveryEngine()

    def test_graph_files_exist(self):
        """Verifies all export formats are generated."""
        self.assertTrue(os.path.exists("graphify-out/graph.json"))
        self.assertTrue(os.path.exists("graphify-out/graph.graphml"))
        self.assertTrue(os.path.exists("graphify-out/graph.html"))
        self.assertTrue(os.path.exists("graphify-out/dashboard.html"))

    def test_nodes_and_edges(self):
        """Checks if expected entity labels and relations are created."""
        labels = set(nx.get_node_attributes(self.G, "label").values())
        print(f"Active node labels in graph: {labels}")
        self.assertIn("Dataset", labels)
        self.assertIn("Catalyst", labels)
        self.assertIn("PerformanceMetric", labels)

    def test_graph_rag_qa(self):
        """Tests that GraphRAG answers common questions correctly."""
        ans = self.rag.answer_question("Which catalyst produced the highest HER?")
        print(f"RAG QA check: {ans}")
        self.assertTrue("highest" in ans.lower() or "µmol/g/h" in ans.lower())

    def test_recommendations(self):
        """Verifies that RAG recommendation engine returns list of dopants."""
        recs = self.rag.recommend("TiO2")
        print(f"Recommendations for TiO2: {recs}")
        self.assertTrue(len(recs) > 0)
        self.assertIn("dopant", recs[0])

    def test_discovery_engine(self):
        """Verifies that discovery engine generates novel candidates and gaps."""
        candidates = self.discovery.discover_novel_combinations(top_n=5)
        print(f"Discovery suggestions: {candidates}")
        self.assertTrue(len(candidates) > 0)
        self.assertIn("discovery_score", candidates[0])

    def test_explainer(self):
        """Verifies predict explainer generates confidence scores and justifications."""
        feats = {
            "host_material": "TiO2",
            "co_catalyst": "Pt",
            "co_catalyst_wt_pct": 1.0,
            "pH": 7.0,
            "temperature_C": 25.0
        }
        exp = self.explainer.explain_prediction(feats, 4500.0)
        self.assertIn("confidence_score", exp)
        self.assertTrue(exp["confidence_score"] > 0)
        self.explainer.print_ascii_explanation(exp)

if __name__ == "__main__":
    import networkx as nx
    unittest.main()
