"""
build.py
Aggregates file extractions into a NetworkX graph, performs entity resolution,
and creates inferred scientific relationships (e.g., OUTPERFORMS, SIMILAR_TO).
"""

import os
import json
import networkx as nx
from .detect import scan_workspace
from .extract import extract_file, clean_node_id
from .cache import GraphifyCache

GRAPH_FILE_JSON = "graphify-out/graph.json"

def build_graph(root_dir=".", force=False):
    """
    Scans the workspace, extracts features from files, updates the cache,
    builds the NetworkX graph, runs inference logic, and returns the graph.
    """
    os.makedirs("graphify-out", exist_ok=True)
    
    # 1. Initialize Cache and Scan Workspace
    cache = GraphifyCache()
    files_by_cat = scan_workspace(root_dir)
    
    # Flatten all files
    all_files = []
    for cat, paths in files_by_cat.items():
        all_files.extend(paths)
        
    # 2. Extract Data for changed files
    extractions = {}
    
    # If graph already exists, load it as base, else start fresh
    if not force and os.path.exists(GRAPH_FILE_JSON):
        print(f"Loading existing graph structure from {GRAPH_FILE_JSON}...")
        try:
            with open(GRAPH_FILE_JSON, "r") as f:
                data = json.load(f)
            G = nx.readwrite.json_graph.node_link_graph(data)
        except Exception as e:
            print(f"Error loading existing graph: {e}. Starting fresh.")
            G = nx.DiGraph()
    else:
        G = nx.DiGraph()

    # Process files
    processed_count = 0
    skipped_count = 0
    
    for filepath in all_files:
        # Check if changed
        if force or cache.is_changed(filepath) or filepath not in cache.cache:
            print(f"Extracting: {os.path.basename(filepath)}...")
            ext_data = extract_file(filepath)
            extractions[filepath] = ext_data
            processed_count += 1
        else:
            skipped_count += 1
            
    print(f"Extraction summary: {processed_count} files processed, {skipped_count} files cached.")
    
    # 3. Apply extractions to NetworkX Graph
    # If a file was updated, first remove its old nodes/edges (if they are owned solely by it)
    # For simplicity, we can rebuild or merge. If we are merging new extractions, we overwrite node properties.
    for filepath, ext_data in extractions.items():
        # Add Nodes
        for node in ext_data["nodes"]:
            node_id = node["id"]
            label = node["label"]
            properties = node.get("properties", {})
            properties["label"] = label # Ensure label is in properties for easy serialization
            
            if G.has_node(node_id):
                # Update properties
                G.nodes[node_id].update(properties)
            else:
                G.add_node(node_id, **properties)
                
        # Add Edges
        for edge in ext_data["edges"]:
            source = edge["source"]
            target = edge["target"]
            edge_type = edge["type"]
            properties = edge.get("properties", {})
            properties["type"] = edge_type
            
            # Add edge if nodes exist
            if G.has_node(source) and G.has_node(target):
                G.add_edge(source, target, **properties)

    # 4. Add ML Model Nodes explicitly if they exist on disk
    models_dir = os.path.join(root_dir, "models")
    if os.path.exists(models_dir):
        for fname in os.listdir(models_dir):
            if fname.endswith(".joblib") and fname != ".gitkeep":
                model_name = os.path.splitext(fname)[0]
                model_id = f"model_{clean_node_id(model_name)}"
                G.add_node(model_id, name=model_name, label="Model", type="saved_model")
                
                # Link model to its code scripts
                train_script_id = "file_train_py"
                if G.has_node(train_script_id):
                    G.add_edge(train_script_id, model_id, type="CREATED_MODEL", confidence="INFERRED")
                
                predict_script_id = "file_predict_py"
                if G.has_node(predict_script_id):
                    G.add_edge(model_id, predict_script_id, type="USED_BY", confidence="INFERRED")

    # 5. Build Scientific Inferences (OUTPERFORMS, SIMILAR_TO)
    print("Building scientific inferences...")
    
    # Get all catalyst and performance nodes
    catalysts = [n for n, attr in G.nodes(data=True) if attr.get("label") == "Catalyst"]
    performance_nodes = {n: attr for n, attr in G.nodes(data=True) if attr.get("label") == "PerformanceMetric"}
    
    # Inferred OUTPERFORMS: link catalysts by comparing standard HER
    cat_her = {}
    for p_id, p_attr in performance_nodes.items():
        her = p_attr.get("HER_std_umol_g_h", 0.0)
        # Find which catalyst is connected to this performance node
        in_edges = G.in_edges(p_id)
        for u, v in in_edges:
            if G.nodes[u].get("label") == "Catalyst":
                cat_her[u] = her
                
    # Create OUTPERFORMS edges between high performing pairs (limit to avoid dense graph)
    sorted_cats = sorted(cat_her.items(), key=lambda x: x[1], reverse=True)
    # Link top 10 catalysts sequentially to show hierarchy
    for i in range(len(sorted_cats) - 1):
        cat_high, her_high = sorted_cats[i]
        cat_low, her_low = sorted_cats[i+1]
        
        # Only add edge if difference is significant
        if her_high > her_low + 50:
            G.add_edge(
                cat_high,
                cat_low,
                type="OUTPERFORMS",
                diff=float(her_high - her_low),
                confidence="INFERRED"
            )
            
    # Inferred SIMILAR_TO: link catalysts with similar host_material or structure
    for i in range(len(catalysts)):
        for j in range(i + 1, len(catalysts)):
            c1 = catalysts[i]
            c2 = catalysts[j]
            attr1 = G.nodes[c1]
            attr2 = G.nodes[c2]
            
            host1 = attr1.get("host_material", "unknown")
            host2 = attr2.get("host_material", "unknown")
            
            if host1 != "unknown" and host1 == host2:
                G.add_edge(c1, c2, type="SIMILAR_TO", reason="same_host", confidence="INFERRED")
                G.add_edge(c2, c1, type="SIMILAR_TO", reason="same_host", confidence="INFERRED")

    # 6. Commit Cache and Save Graph
    cache.commit()
    
    # Save graph in NetworkX node-link format
    node_link_data = nx.readwrite.json_graph.node_link_data(G)
    with open(GRAPH_FILE_JSON, "w") as f:
        json.dump(node_link_data, f, indent=2)
        
    print(f"Graph successfully built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G

if __name__ == "__main__":
    build_graph()
