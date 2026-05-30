"""
explain.py
Model Explainability Layer: Justifies machine learning predictions by finding similar
benchmark catalysts in the knowledge graph, mapping literature support, and calculating confidence scores.
"""

import json
import numpy as np
from .rag import GraphRAG

class PredictExplainer:
    def __init__(self, graph_path="graphify-out/graph.json"):
        self.rag = GraphRAG(graph_path)

    def explain_prediction(self, features, predicted_her):
        """
        Generates a structured explanation graph and confidence score for a prediction.
        
        features: dict containing keys like:
            host_material, co_catalyst, co_catalyst_wt_pct, temperature_C, pH, light_power_W
        predicted_her: float (original scale HER value in umol/g/h)
        """
        explanation = {
            "prediction": {
                "predicted_HER": float(predicted_her),
                "features": features
            },
            "literature_support": [],
            "intermediate_relationships": [],
            "confidence_score": 0.0,
            "confidence_reasons": []
        }

        if not self.rag.G or self.rag.G.number_of_nodes() == 0:
            explanation["confidence_reasons"].append("Knowledge graph not built yet.")
            return explanation

        host = str(features.get("host_material", "")).strip()
        co_cat = str(features.get("co_catalyst", "")).strip()
        ph = features.get("pH")
        temp = features.get("temperature_C")

        # 1. Look for matching Catalysts
        exact_matches = []
        similar_matches = []
        
        # Track counts of nodes for confidence score calculation
        host_exists = False
        cocat_exists = False
        paper_citations = 0

        # Scan catalysts in graph
        for node_id, attr in self.rag.G.nodes(data=True):
            if attr.get("label") == "Catalyst":
                c_name = attr.get("name", "")
                c_host = attr.get("host_material", "")
                
                # Check host match
                if host.lower() == c_host.lower():
                    host_exists = True
                    
                # Find dopant edges
                dopants = []
                for u, v in self.rag.G.out_edges(node_id):
                    target_attr = self.rag.G.nodes[v]
                    if target_attr.get("label") == "Dopant":
                        d_name = target_attr.get("name", "")
                        dopants.append(d_name)
                        if co_cat.lower() == d_name.lower():
                            cocat_exists = True

                # Check if matches current prediction formulation
                if host.lower() == c_host.lower() and co_cat.lower() in [d.lower() for d in dopants]:
                    exact_matches.append((node_id, c_name))
                elif host.lower() == c_host.lower():
                    similar_matches.append((node_id, c_name))

        # 2. Extract literature references and performance for matches
        supported_hers = []
        for cat_id, cat_name in (exact_matches if exact_matches else similar_matches[:5]):
            # Get papers citing this catalyst
            papers = []
            hers = []
            
            for u, v, edge_attr in self.rag.G.out_edges(cat_id, data=True):
                target_attr = self.rag.G.nodes[v]
                if target_attr.get("label") == "Publication":
                    papers.append(target_attr.get("title", v))
                    paper_citations += 1
                elif target_attr.get("label") == "PerformanceMetric":
                    val = target_attr.get("HER_std_umol_g_h", 0.0)
                    hers.append(val)
                    supported_hers.append(val)
                    
            explanation["literature_support"].append({
                "catalyst_name": cat_name,
                "relationship": "EXACT_FORMULATION" if exact_matches else "SAME_HOST_SEMICONDUCTOR",
                "reported_HERs": hers,
                "references": list(set(papers))
            })

        # 3. Intermediate Relationships (Synthesis & Conditions correlation)
        # Find if synthesis or conditions overlap
        if ph is not None:
            explanation["intermediate_relationships"].append({
                "entity": "pH_Level",
                "value": ph,
                "effect": "Direct input to photoreforming matrix"
            })
        if temp is not None:
            explanation["intermediate_relationships"].append({
                "entity": "Temperature_C",
                "value": temp,
                "effect": "Increases kinetic rates; thermal co-activation"
            })

        # 4. Calculate Confidence Score (0.0 to 1.0)
        score = 0.0
        
        # Base Host Material Support
        if host_exists:
            score += 0.35
            explanation["confidence_reasons"].append(f"Host semiconductor matrix '{host}' exists in training dataset.")
        else:
            explanation["confidence_reasons"].append(f"Host material '{host}' is novel/unseen in training records.")
            
        # Cocatalyst Support
        if co_cat and co_cat.lower() != "none" and co_cat.lower() != "unknown":
            if cocat_exists:
                score += 0.25
                explanation["confidence_reasons"].append(f"Co-catalyst/dopant '{co_cat}' is well-characterized in training datasets.")
            else:
                explanation["confidence_reasons"].append(f"Co-catalyst '{co_cat}' has no prior matching record in dataset.")
        else:
            score += 0.25
            explanation["confidence_reasons"].append("No co-catalyst requested (pure semiconductor prediction).")

        # Literature density
        if paper_citations > 0:
            cit_bonus = min(0.20, paper_citations * 0.05)
            score += cit_bonus
            explanation["confidence_reasons"].append(f"Supported by {paper_citations} direct literature paper citations in knowledge graph.")
        else:
            explanation["confidence_reasons"].append("No direct literature citation found matching this catalyst formulation.")

        # Performance variance checks
        if supported_hers:
            mean_ref = np.mean(supported_hers)
            # If predictions are within a reasonable bounds of literature values
            ratio = min(predicted_her, mean_ref) / max(predicted_her, mean_ref)
            pred_bonus = ratio * 0.20
            score += pred_bonus
            explanation["confidence_reasons"].append(f"Predicted HER matches historical experimental range (agreement score: {round(pred_bonus*100)}%).")
        else:
            explanation["confidence_reasons"].append("No experimental values found for range validation.")

        explanation["confidence_score"] = round(min(1.0, score), 2)
        return explanation

    def print_ascii_explanation(self, exp):
        """Generates a text graph representing the model reasoning chain."""
        import sys
        # Reconfigure console encoding to prevent crash on Windows standard outputs
        if sys.stdout.encoding != 'utf-8':
            try:
                sys.stdout.reconfigure(encoding='utf-8')
            except Exception:
                pass

        pred = exp["prediction"]
        feats = pred["features"]
        
        print("\n" + "="*60)
        print("         PREDICTION EXPLAINABILITY GRAPH (Graphify)")
        print("="*60)
        print(f" INPUT: Host={feats.get('host_material')} | Dopant={feats.get('co_catalyst')} | pH={feats.get('pH')}")
        print("   |")
        print("   +-- [Graph Traversal] ----> Mapped Literature Benchmarks:")
        for lit in exp["literature_support"][:2]:
            ref_str = lit['references'][0][:45] + "..." if lit['references'] else "No Reference Name"
            # Remove any special unicode subscripts from names during CLI prints
            cat_name_cleaned = lit['catalyst_name'].encode('ascii', errors='replace').decode('ascii')
            ref_str_cleaned = ref_str.encode('ascii', errors='replace').decode('ascii')
            print(f"   |      +-- Catalyst: {cat_name_cleaned} ({lit['relationship']})")
            print(f"   |      |     +-- Reported HER: {lit['reported_HERs']}")
            print(f"   |      |     +-- Source: {ref_str_cleaned}")
        print("   |")
        print(f"   +-- [Confidence Engine] -> Score: {int(exp['confidence_score'] * 100)}%")
        for reason in exp["confidence_reasons"][:3]:
            reason_cleaned = reason.encode('ascii', errors='replace').decode('ascii')
            print(f"   |      +-- [X] {reason_cleaned}")
        print("   |")
        print(f"   +-- [Model Output] ------> Predicted HER: {round(pred['predicted_HER'], 2)} umol/g/h")
        print("="*60 + "\n")

