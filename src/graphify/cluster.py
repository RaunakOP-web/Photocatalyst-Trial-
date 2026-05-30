"""
cluster.py
Applies Louvain community detection to identify functional clusters of catalysts,
experimental conditions, publications, and code components in the graph.
"""

import json
import networkx as nx
from networkx.algorithms.community import louvain_communities
from .build import GRAPH_FILE_JSON


def cluster_graph(G=None):
    """
    Applies Louvain community clustering to the graph and assigns 'community' attributes to nodes.
    Saves and returns the clustered graph.
    """
    if G is None:
        if not os.path.exists(GRAPH_FILE_JSON):
            from .build import build_graph
            G = build_graph()
        else:
            try:
                with open(GRAPH_FILE_JSON, "r") as f:
                    data = json.load(f)
                G = nx.readwrite.json_graph.node_link_graph(data)
            except Exception as e:
                print(f"Error loading graph: {e}")
                return None

    # Convert to undirected copy for community detection
    undirected_G = G.to_undirected()
    
    try:
        # Compute louvain communities
        communities = louvain_communities(undirected_G, seed=42)
        print(f"Detected {len(communities)} communities in the knowledge graph.")
        
        # Label nodes with community IDs
        for comm_id, node_set in enumerate(communities):
            for node_id in node_set:
                G.nodes[node_id]["community"] = comm_id
                
        # Save graph
        node_link_data = nx.readwrite.json_graph.node_link_data(G)
        with open(GRAPH_FILE_JSON, "w") as f:
            json.dump(node_link_data, f, indent=2)
            
        print("Community labels successfully saved to graph nodes.")
    except Exception as e:
        print(f"Error during community clustering: {e}")
        
    return G

if __name__ == "__main__":
    import os
    cluster_graph()
