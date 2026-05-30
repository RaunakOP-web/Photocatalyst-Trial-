"""
export.py
Handles exporting the knowledge graph to JSON, GraphML, and a premium, interactive
D3.js-based HTML visualization page.
"""

import os
import json
import networkx as nx

GRAPH_JSON_PATH = "graphify-out/graph.json"
GRAPH_GRAPHML_PATH = "graphify-out/graph.graphml"
GRAPH_HTML_PATH = "graphify-out/graph.html"

def export_all(G):
    """
    Exports the NetworkX graph to JSON, GraphML, and builds the D3 HTML visualizer.
    """
    os.makedirs("graphify-out", exist_ok=True)
    
    # 1. Export JSON
    node_link = nx.readwrite.json_graph.node_link_data(G)
    with open(GRAPH_JSON_PATH, "w") as f:
        json.dump(node_link, f, indent=2)
    print(f"Exported graph to JSON: {GRAPH_JSON_PATH}")
    
    # 2. Export GraphML
    # NetworkX write_graphml requires all attribute values to be strings, ints, floats, or bools.
    # We must sanitize node and edge attributes first.
    G_clean = G.copy()
    for n, attr in G_clean.nodes(data=True):
        for k, v in list(attr.items()):
            if isinstance(v, (list, dict, set)):
                attr[k] = str(v)
                
    for u, v, attr in G_clean.edges(data=True):
        for k, v in list(attr.items()):
            if isinstance(v, (list, dict, set)):
                attr[k] = str(v)
                
    nx.write_graphml(G_clean, GRAPH_GRAPHML_PATH)
    print(f"Exported graph to GraphML: {GRAPH_GRAPHML_PATH}")
    
    # 3. Export D3 HTML Visualizer
    build_html_visualizer(node_link)

def build_html_visualizer(node_link_data):
    """Generates a premium, interactive force-directed graph in HTML."""
    
    html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Photocatalyst Knowledge Graph</title>
    <!-- Tailwind CSS (for modern UI styling) -->
    <script src="https://cdn.tailwindcss.com"></script>
    <!-- D3.js -->
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {
            background-color: #0f172a;
            color: #f8fafc;
            font-family: 'Inter', sans-serif;
            overflow: hidden;
        }
        .node:hover {
            cursor: pointer;
            filter: brightness(1.2);
        }
        .link {
            stroke-opacity: 0.4;
            stroke-width: 1.5px;
            transition: stroke-opacity 0.2s;
        }
        .tooltip {
            position: absolute;
            background: rgba(15, 23, 42, 0.95);
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 12px;
            pointer-events: none;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.5);
            max-width: 300px;
            z-index: 100;
        }
        /* Custom scrollbar for panel */
        ::-webkit-scrollbar {
            width: 6px;
        }
        ::-webkit-scrollbar-track {
            background: #1e293b;
        }
        ::-webkit-scrollbar-thumb {
            background: #475569;
            border-radius: 3px;
        }
    </style>
</head>
<body class="h-screen w-screen flex">

    <!-- Left Sidebar: Info Panel -->
    <div class="w-80 bg-slate-900/90 border-r border-slate-800 p-6 flex flex-col justify-between h-full z-10">
        <div>
            <h1 class="text-xl font-bold text-teal-400 mb-2 flex items-center gap-2">
                <span>🧬</span> Graphify-Science
            </h1>
            <p class="text-xs text-slate-400 mb-6">Photocatalyst Knowledge Graph Explorer</p>
            
            <!-- Filters -->
            <div class="space-y-4">
                <div>
                    <label class="block text-xs font-semibold text-slate-400 mb-2">Search Node</label>
                    <input type="text" id="search-box" placeholder="e.g. TiO2, Pt..." 
                           class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 text-sm focus:outline-none focus:border-teal-500 text-slate-200">
                </div>
                
                <div>
                    <label class="block text-xs font-semibold text-slate-400 mb-2">Filter Node Label</label>
                    <div class="space-y-2 text-xs">
                        <label class="flex items-center gap-2">
                            <input type="checkbox" checked class="filter-checkbox" value="Catalyst">
                            <span class="w-3 h-3 rounded-full bg-emerald-500"></span> Catalyst
                        </label>
                        <label class="flex items-center gap-2">
                            <input type="checkbox" checked class="filter-checkbox" value="Dopant">
                            <span class="w-3 h-3 rounded-full bg-amber-500"></span> Dopant/Co-catalyst
                        </label>
                        <label class="flex items-center gap-2">
                            <input type="checkbox" checked class="filter-checkbox" value="ExperimentalCondition">
                            <span class="w-3 h-3 rounded-full bg-sky-500"></span> Condition
                        </label>
                        <label class="flex items-center gap-2">
                            <input type="checkbox" checked class="filter-checkbox" value="PerformanceMetric">
                            <span class="w-3 h-3 rounded-full bg-rose-500"></span> Performance
                        </label>
                        <label class="flex items-center gap-2">
                            <input type="checkbox" checked class="filter-checkbox" value="Publication">
                            <span class="w-3 h-3 rounded-full bg-indigo-500"></span> Publication
                        </label>
                        <label class="flex items-center gap-2">
                            <input type="checkbox" checked class="filter-checkbox" value="Model">
                            <span class="w-3 h-3 rounded-full bg-fuchsia-500"></span> ML Model
                        </label>
                        <label class="flex items-center gap-2">
                            <input type="checkbox" checked class="filter-checkbox" value="CodeEntity">
                            <span class="w-3 h-3 rounded-full bg-slate-500"></span> Code Structure
                        </label>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Summary Stats -->
        <div class="border-t border-slate-800 pt-4 text-xs text-slate-400 space-y-1">
            <div>Nodes: <span id="nodes-count" class="text-slate-200">0</span></div>
            <div>Edges: <span id="edges-count" class="text-slate-200">0</span></div>
            <div class="text-[10px] text-slate-500 mt-2">D3.js Force Directed Layout</div>
        </div>
    </div>

    <!-- Main Canvas Area -->
    <div class="flex-1 h-full relative" id="canvas-container">
        <!-- SVG container for D3 -->
        <svg id="graph-svg" class="w-full h-full"></svg>
        
        <!-- Tooltip -->
        <div id="tooltip" class="tooltip hidden"></div>
        
        <!-- Graph controls -->
        <div class="absolute bottom-6 right-6 flex gap-2">
            <button onclick="resetZoom()" class="bg-slate-800 hover:bg-slate-700 border border-slate-700 px-3 py-1.5 rounded text-xs">Reset View</button>
        </div>
    </div>

    <!-- Node Detail Drawer -->
    <div id="detail-drawer" class="fixed right-0 top-0 h-full w-96 bg-slate-900/95 border-l border-slate-800 p-6 shadow-2xl transform translate-x-full transition-transform duration-300 z-50 overflow-y-auto">
        <div class="flex justify-between items-center mb-6">
            <h2 class="text-lg font-bold text-teal-400">Node Details</h2>
            <button onclick="closeDrawer()" class="text-slate-400 hover:text-slate-200">✕</button>
        </div>
        <div id="drawer-content" class="space-y-4 text-sm text-slate-300">
            <!-- Populated dynamically -->
        </div>
    </div>

    <!-- Inject Graph Data -->
    <script>
        const graphData = GRAPH_DATA_PLACEHOLDER;
        
        // Colors mapping by node label
        const colors = {
            "Catalyst": "#10b981", // Emerald
            "Dopant": "#f59e0b", // Amber
            "ExperimentalCondition": "#0ea5e9", // Sky
            "PerformanceMetric": "#f43f5e", // Rose
            "Publication": "#6366f1", // Indigo
            "Model": "#d946ef", // Fuchsia
            "CodeEntity": "#64748b", // Slate
            "Concept": "#84cc16", // Lime
            "Dataset": "#a855f7" // Purple
        };

        const width = document.getElementById('canvas-container').clientWidth;
        const height = document.getElementById('canvas-container').clientHeight;

        const svg = d3.select("#graph-svg");
        const g = svg.append("g");

        // Zoom behaviour
        const zoom = d3.zoom()
            .scaleExtent([0.1, 4])
            .on("zoom", (event) => {
                g.attr("transform", event.transform);
            });
        svg.call(zoom);

        // Update counts
        document.getElementById("nodes-count").innerText = graphData.nodes.length;
        document.getElementById("edges-count").innerText = graphData.links.length;

        // Force Simulation
        const simulation = d3.forceSimulation(graphData.nodes)
            .force("link", d3.forceLink(graphData.links).id(d => d.id).distance(60))
            .force("charge", d3.forceManyBody().strength(-120))
            .force("center", d3.forceCenter(width / 2, height / 2))
            .force("collision", d3.forceCollide().radius(25));

        // Create Links
        const link = g.append("g")
            .attr("class", "links")
            .selectAll("line")
            .data(graphData.links)
            .enter().append("line")
            .attr("class", "link")
            .attr("stroke", "#334155");

        // Create Nodes
        const node = g.append("g")
            .attr("class", "nodes")
            .selectAll("g")
            .data(graphData.nodes)
            .enter().append("g")
            .attr("class", "node")
            .call(d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended))
            .on("click", (event, d) => showDetails(d))
            .on("mouseover", showTooltip)
            .on("mouseout", hideTooltip);

        // Node circles
        node.append("circle")
            .attr("r", d => d.label === "Catalyst" ? 9 : 6)
            .attr("fill", d => colors[d.label] || "#94a3b8");

        // Node text labels
        node.append("text")
            .attr("dx", 12)
            .attr("dy", ".35em")
            .text(d => d.name || d.title || d.id)
            .attr("font-size", "10px")
            .attr("fill", "#94a3b8")
            .style("pointer-events", "none");

        // Ticks for force layout
        simulation.on("tick", () => {
            link
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);

            node
                .attr("transform", d => `translate(${d.x},${d.y})`);
        });

        // Search features
        d3.select("#search-box").on("input", function() {
            const query = this.value.toLowerCase();
            node.selectAll("circle")
                .attr("stroke", d => {
                    const name = (d.name || d.title || d.id).toLowerCase();
                    return name.includes(query) && query !== "" ? "#f59e0b" : "none";
                })
                .attr("stroke-width", d => {
                    const name = (d.name || d.title || d.id).toLowerCase();
                    return name.includes(query) && query !== "" ? 3 : 0;
                });
        });

        // Checkbox filtering
        d3.selectAll(".filter-checkbox").on("change", function() {
            const activeLabels = [];
            d3.selectAll(".filter-checkbox").each(function() {
                if (this.checked) activeLabels.push(this.value);
            });
            
            node.style("display", d => activeLabels.includes(d.label) ? "" : "none");
            link.style("display", d => {
                return activeLabels.includes(d.source.label) && activeLabels.includes(d.target.label) ? "" : "none";
            });
        });

        // Drag helpers
        function dragstarted(event, d) {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }

        function dragged(event, d) {
            d.fx = event.x;
            d.fy = event.y;
        }

        function dragended(event, d) {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }

        // Reset view
        function resetZoom() {
            svg.transition().duration(750).call(
                zoom.transform,
                d3.zoomIdentity
            );
        }

        // Tooltip handlers
        const tooltip = d3.select("#tooltip");
        function showTooltip(event, d) {
            tooltip.classed("hidden", false)
                .html(`
                    <div class="font-bold text-teal-400 text-xs mb-1">${d.label}</div>
                    <div class="font-semibold text-slate-200 text-sm">${d.name || d.title || d.id}</div>
                `)
                .style("left", (event.pageX + 15) + "px")
                .style("top", (event.pageY - 15) + "px");
        }

        function hideTooltip() {
            tooltip.classed("hidden", true);
        }

        // Details drawer
        function showDetails(d) {
            const drawer = document.getElementById("detail-drawer");
            const content = document.getElementById("drawer-content");
            
            drawer.classList.remove("translate-x-full");
            
            let propsHtml = '';
            for (const [key, val] of Object.entries(d)) {
                if (key !== 'x' && key !== 'y' && key !== 'vy' && key !== 'vx' && key !== 'index') {
                    propsHtml += `
                        <div class="border-b border-slate-800 pb-2">
                            <span class="text-xs font-semibold text-slate-500 block uppercase">${key}</span>
                            <span class="text-sm text-slate-200">${val}</span>
                        </div>
                    `;
                }
            }
            
            content.innerHTML = `
                <div class="flex items-center gap-2 mb-4">
                    <span class="w-4 h-4 rounded-full" style="background-color: ${colors[d.label] || '#64748b'}"></span>
                    <span class="text-xs uppercase font-bold text-slate-400">${d.label}</span>
                </div>
                <div class="text-lg font-semibold text-slate-100 mb-6">${d.name || d.title || d.id}</div>
                <div class="space-y-4">
                    ${propsHtml}
                </div>
            `;
        }

        function closeDrawer() {
            document.getElementById("detail-drawer").classList.add("translate-x-full");
        }
    </script>
</body>
</html>"""

    # Inject data (NetworkX returns list of links as source, target)
    html_content = html_template.replace("GRAPH_DATA_PLACEHOLDER", json.dumps(node_link_data))
    
    with open(GRAPH_HTML_PATH, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    print(f"Generated D3 HTML Graph Visualization: {GRAPH_HTML_PATH}")
