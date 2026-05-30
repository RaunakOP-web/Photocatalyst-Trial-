"""
src/graphify/__init__.py
Photocatalyst Graphify: Integrates literature datasets, ML model metadata,
and codebase structure into a unified, queryable knowledge graph.
"""

from .build import build_graph
from .cluster import cluster_graph
from .export import export_all
from .dashboard import compile_dashboard

def run_pipeline(root_dir=".", force=False):
    """
    Executes the entire Graphify pipeline in order:
    1. Scan workspace and extract file content/databases.
    2. Build NetworkX graph with scientific relations.
    3. Run Louvain community clustering.
    4. Export JSON and GraphML files.
    5. Compile interactive dashboard and D3 visualizer.
    """
    print("\n" + "="*50)
    print("      LAUNCHING PHOTOCATALYST GRAPHIFY PIPELINE")
    print("="*50)
    
    # 1. Build
    G = build_graph(root_dir, force=force)
    
    # 2. Cluster
    G = cluster_graph(G)
    
    # 3. Export
    export_all(G)
    
    # 4. Compile Dashboard
    compile_dashboard(root_dir)
    
    print("="*50)
    print("   GRAPHIFY PIPELINE COMPLETED SUCCESSFULLY!")
    print("   Open graphify-out/dashboard.html to explore.")
    print("="*50 + "\n")
    
    return G
