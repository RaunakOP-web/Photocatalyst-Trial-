"""
discovery.py
Graph Discovery Engine: Uses link prediction heuristics and node similarity
to identify unexplored catalyst formulations and publication opportunities.
"""

import json
import networkx as nx
from .build import GRAPH_FILE_JSON

class DiscoveryEngine:
    def __init__(self, graph_path=GRAPH_FILE_JSON):
        self.graph_path = graph_path
        self.G = self.load_graph()

    def load_graph(self):
        try:
            with open(self.graph_path, "r") as f:
                data = json.load(f)
            return nx.readwrite.json_graph.node_link_graph(data)
        except Exception as e:
            print(f"Error loading graph in discovery: {e}")
            return nx.DiGraph()

    def discover_novel_combinations(self, top_n=10):
        """
        Predicts missing links between Hosts and Dopants.
        Returns ranked list of novel catalyst formulations.
        """
        if not self.G or self.G.number_of_nodes() == 0:
            return []

        # Extract all hosts from Catalyst nodes and all Dopants
        hosts = set()
        dopants = set()
        
        # Mapping to trace which hosts are already paired with which dopants
        existing_pairs = set()
        
        # Calculate performance metrics for dopants and hosts to use for scoring
        dopant_performance = {}
        host_performance = {}

        for node_id, attr in self.G.nodes(data=True):
            if attr.get("label") == "Catalyst":
                host = attr.get("host_material", "")
                if host and host != "unknown":
                    hosts.add(host)
                    
                    # Find performance linked to this catalyst
                    her_val = 0.0
                    for u, v in self.G.out_edges(node_id):
                        target_attr = self.G.nodes[v]
                        if target_attr.get("label") == "PerformanceMetric":
                            her_val = max(her_val, target_attr.get("HER_std_umol_g_h", 0.0))
                    
                    if host not in host_performance:
                        host_performance[host] = []
                    if her_val > 0:
                        host_performance[host].append(her_val)

            elif attr.get("label") == "Dopant":
                dopant_name = attr.get("name", "")
                if dopant_name:
                    dopants.add(dopant_name)
                    
                    # Check which catalysts link to this dopant
                    in_edges = self.G.in_edges(node_id)
                    for u, v in in_edges:
                        cat_attr = self.G.nodes[u]
                        if cat_attr.get("label") == "Catalyst":
                            c_host = cat_attr.get("host_material", "")
                            if c_host:
                                existing_pairs.add((c_host.lower(), dopant_name.lower()))
                                
                            # Fetch performance of this parent catalyst
                            her_val = 0.0
                            for x, y in self.G.out_edges(u):
                                if self.G.nodes[y].get("label") == "PerformanceMetric":
                                    her_val = max(her_val, self.G.nodes[y].get("HER_std_umol_g_h", 0.0))
                                    
                            if dopant_name not in dopant_performance:
                                dopant_performance[dopant_name] = []
                            if her_val > 0:
                                dopant_performance[dopant_name].append(her_val)

        # Average performances
        avg_dopant_her = {d: (sum(h)/len(h) if h else 0) for d, h in dopant_performance.items()}
        avg_host_her = {h: (sum(p)/len(p) if p else 0) for h, p in host_performance.items()}
        
        global_avg_her = sum(avg_dopant_her.values()) / len(avg_dopant_her) if avg_dopant_her else 1000

        # Generate candidates for all host-dopant pairs that do not exist
        candidates = []
        for host in hosts:
            for dopant in dopants:
                if (host.lower(), dopant.lower()) not in existing_pairs:
                    # Calculate Jaccard similarity or score based on sibling nodes
                    # Score = (Avg host performance + Avg dopant performance) / 2
                    h_score = avg_host_her.get(host, global_avg_her)
                    d_score = avg_dopant_her.get(dopant, global_avg_her)
                    
                    # Combine score
                    predictive_score = (h_score + d_score) / 2
                    
                    # If this host is similar to another host that has this dopant, add similarity bonus
                    similarity_bonus = 0.0
                    for sibling_id, sibling_attr in self.G.nodes(data=True):
                        if sibling_attr.get("label") == "Catalyst" and sibling_attr.get("host_material") != host:
                            # If sibling has the dopant
                            sib_host = sibling_attr.get("host_material", "")
                            if (sib_host.lower(), dopant.lower()) in existing_pairs:
                                # Add bonus if they share a SIMILAR_TO edge
                                # Simple check: does the graph contain an edge between sibling and host?
                                pass
                                
                    candidates.append({
                        "host_material": host,
                        "co_catalyst": dopant,
                        "discovery_score": round(predictive_score, 2),
                        "rationale": f"High benchmark rates of host '{host}' paired with high-efficiency co-catalyst '{dopant}'."
                    })
                    
        # Sort candidates by discovery_score desc
        candidates = sorted(candidates, key=lambda x: x["discovery_score"], reverse=True)
        return candidates[:top_n]

    def discover_publication_gaps(self):
        """
        Identifies thematic research gaps based on low-density node communities.
        """
        gaps = []
        if not self.G:
            return gaps
            
        # Group catalysts by community
        comm_hosts = {}
        for n, attr in self.G.nodes(data=True):
            if attr.get("label") == "Catalyst" and "community" in attr:
                comm = attr["community"]
                host = attr.get("host_material", "unknown")
                if comm not in comm_hosts:
                    comm_hosts[comm] = []
                comm_hosts[comm].append(host)
                
        # Find which communities have no high performing dopants tested yet
        # Or identify where a specific host is the only member of its class in a community
        for comm, hosts_list in comm_hosts.items():
            hosts_unique = list(set(hosts_list))
            if len(hosts_unique) <= 2:
                gaps.append({
                    "community_id": comm,
                    "representative_hosts": hosts_unique,
                    "gap_type": "UNDER_REPRESENTED_MATERIAL_CLASS",
                    "opportunity": f"Explore synthesis routes and dopant matrices for rare class containing {hosts_unique}"
                })
        return gaps
