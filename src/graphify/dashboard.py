"""
dashboard.py
Compiles a rich, premium, and fully client-side Single Page Application (SPA) dashboard
at graphify-out/dashboard.html. It integrates the Graph Visualizer, Catalyst Explorer,
GraphRAG Playground, Discovery Console, and Explainability Layers.
"""

import os
import json
import networkx as nx
from .build import GRAPH_FILE_JSON
from .rag import GraphRAG
from .explain import PredictExplainer
from .discovery import DiscoveryEngine

DASHBOARD_HTML_PATH = "graphify-out/dashboard.html"

def compile_dashboard(root_dir="."):
    """
    Compiles all graph analytics, discovery recommendations, GraphRAG outputs,
    and visual elements into a self-contained, responsive HTML dashboard.
    """
    os.makedirs("graphify-out", exist_ok=True)
    
    # 1. Load Graph and Modules
    rag = GraphRAG(GRAPH_FILE_JSON)
    explainer = PredictExplainer(GRAPH_FILE_JSON)
    discovery = DiscoveryEngine(GRAPH_FILE_JSON)
    
    if not rag.G or rag.G.number_of_nodes() == 0:
        print("Error: Knowledge graph must be built before compiling dashboard.")
        return
        
    # Get JSON structure for graph
    node_link_data = nx.readwrite.json_graph.node_link_data(rag.G)
    
    # 2. Pre-run Discovery Engine
    novel_candidates = discovery.discover_novel_combinations(top_n=15)
    lit_gaps = discovery.discover_publication_gaps()
    
    # 3. Compile Dopant Recommendations for major host materials
    major_hosts = ["TiO2", "g-C3N4", "CdS", "ZnO", "ZnIn2S4"]
    recommendations = {}
    for host in major_hosts:
        recs = rag.recommend(host)
        recommendations[host] = recs
        
    # 4. Generate Explainer Demos for high-value formulations
    explainer_demos = []
    demo_configurations = [
        {"host_material": "TiO2", "co_catalyst": "Pt", "co_catalyst_wt_pct": 1.0, "pH": 7.0, "temperature_C": 25.0},
        {"host_material": "g-C3N4", "co_catalyst": "NiS", "co_catalyst_wt_pct": 2.0, "pH": 7.0, "temperature_C": 25.0},
        {"host_material": "CdS", "co_catalyst": "Pt", "co_catalyst_wt_pct": 0.5, "pH": 7.0, "temperature_C": 25.0},
    ]
    for config in demo_configurations:
        # Simulate a prediction value close to historical mean
        mean_her = 5000.0
        exp = explainer.explain_prediction(config, mean_her)
        explainer_demos.append(exp)

    # 5. Pre-run Common Questions
    common_questions = [
        "Which catalyst produced the highest HER?",
        "What synthesis method is most associated with high hydrogen production?",
        "Which papers report Pt-doped g-C3N4?",
        "Which papers report Au-doped TiO2?"
    ]
    qa_pairs = {}
    for q in common_questions:
        qa_pairs[q] = rag.answer_question(q)

    # 6. Read template and write dashboard
    dashboard_html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Photocatalyst Discovery Dashboard</title>
    <!-- Tailwind CSS -->
    <script src="https://cdn.tailwindcss.com"></script>
    <!-- D3.js -->
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        body {
            background-color: #0b0f19;
            color: #e2e8f0;
            font-family: 'Outfit', sans-serif;
        }
        .tab-btn.active {
            border-bottom: 2px solid #14b8a6;
            color: #14b8a6;
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
    </style>
</head>
<body class="h-screen w-screen flex flex-col overflow-hidden">

    <!-- Premium Navigation Header -->
    <header class="h-16 border-b border-slate-800 bg-slate-900/90 px-8 flex justify-between items-center z-20">
        <div class="flex items-center gap-3">
            <span class="text-2xl">⚡</span>
            <div>
                <h1 class="text-lg font-bold text-teal-400">Photocatalyst Graphify System</h1>
                <p class="text-[10px] text-slate-500">Knowledge Graph Discovery & Explainability Engine</p>
            </div>
        </div>
        <!-- Tab navigation -->
        <nav class="flex gap-6 h-full items-end">
            <button onclick="switchTab('graph')" id="tab-graph" class="tab-btn pb-4 px-2 text-sm font-medium text-slate-400 active">Network Visualizer</button>
            <button onclick="switchTab('explorer')" id="tab-explorer" class="tab-btn pb-4 px-2 text-sm font-medium text-slate-400">Catalyst Explorer</button>
            <button onclick="switchTab('rag')" id="tab-rag" class="tab-btn pb-4 px-2 text-sm font-medium text-slate-400">GraphRAG Q&A</button>
            <button onclick="switchTab('discovery')" id="tab-discovery" class="tab-btn pb-4 px-2 text-sm font-medium text-slate-400">Discovery Engine</button>
            <button onclick="switchTab('explain')" id="tab-explain" class="tab-btn pb-4 px-2 text-sm font-medium text-slate-400">Model Explainability</button>
        </nav>
    </header>

    <!-- Main Content Panels -->
    <div class="flex-1 w-full relative overflow-hidden">

        <!-- TAB 1: Force Directed Visualizer -->
        <div id="panel-graph" class="tab-panel w-full h-full flex">
            <!-- Sidebar -->
            <div class="w-80 bg-slate-900/60 border-r border-slate-800 p-6 flex flex-col justify-between h-full z-10 overflow-y-auto">
                <div class="space-y-6">
                    <div>
                        <h2 class="text-sm font-semibold text-slate-400 mb-2">Search Node</h2>
                        <input type="text" id="graph-search" placeholder="e.g. g-C3N4..." 
                               class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 text-sm focus:outline-none focus:border-teal-500 text-slate-200">
                    </div>
                    
                    <div>
                        <h2 class="text-sm font-semibold text-slate-400 mb-2">Display Filters</h2>
                        <div class="space-y-2 text-xs" id="legend-filters">
                            <!-- Filters populated dynamically -->
                        </div>
                    </div>
                </div>
                <div class="border-t border-slate-800 pt-4 text-xs text-slate-400 space-y-1">
                    <div>Nodes: <span id="nodes-count" class="text-teal-400">0</span></div>
                    <div>Edges: <span id="edges-count" class="text-teal-400">0</span></div>
                </div>
            </div>
            
            <!-- Canvas -->
            <div class="flex-1 h-full relative" id="d3-container">
                <svg id="dashboard-svg" class="w-full h-full"></svg>
                <div id="tooltip" class="tooltip hidden"></div>
                
                <!-- Info Drawer (Right side inside Visualizer) -->
                <div id="info-drawer" class="absolute right-6 top-6 bottom-6 w-80 bg-slate-900/90 border border-slate-800 rounded-lg p-5 shadow-2xl hidden overflow-y-auto">
                    <div class="flex justify-between items-center mb-4">
                        <h3 class="font-bold text-teal-400 text-sm">Entity details</h3>
                        <button onclick="hideDrawer()" class="text-slate-400 hover:text-slate-200 text-xs">✕</button>
                    </div>
                    <div id="info-content" class="text-xs space-y-3"></div>
                </div>
            </div>
        </div>

        <!-- TAB 2: Catalyst Explorer -->
        <div id="panel-explorer" class="tab-panel w-full h-full p-8 hidden overflow-y-auto">
            <div class="max-w-6xl mx-auto space-y-6">
                <div class="flex justify-between items-center">
                    <h2 class="text-2xl font-bold text-teal-400">Catalyst Explorer</h2>
                    <input type="text" id="explorer-filter" oninput="filterCatalysts()" placeholder="Filter by host or dopant..." 
                           class="bg-slate-800 border border-slate-700 rounded px-4 py-2 text-sm focus:outline-none focus:border-teal-500 w-80">
                </div>
                <div class="bg-slate-900 border border-slate-800 rounded-lg overflow-hidden">
                    <table class="w-full text-left border-collapse">
                        <thead>
                            <tr class="bg-slate-850 border-b border-slate-800 text-xs uppercase tracking-wider text-slate-400">
                                <th class="p-4">Name</th>
                                <th class="p-4">Host Material</th>
                                <th class="p-4">Secondary Phase</th>
                                <th class="p-4">Co-catalyst</th>
                                <th class="p-4">Dopant Wt%</th>
                            </tr>
                        </thead>
                        <tbody id="catalyst-table-body" class="text-sm divide-y divide-slate-800/50">
                            <!-- Populated dynamically -->
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <!-- TAB 3: GraphRAG Q&A -->
        <div id="panel-rag" class="tab-panel w-full h-full p-8 hidden overflow-y-auto">
            <div class="max-w-4xl mx-auto space-y-8">
                <div>
                    <h2 class="text-2xl font-bold text-teal-400 mb-2">GraphRAG Q&A Console</h2>
                    <p class="text-sm text-slate-400">Ask the scientific knowledge graph questions about materials, synthesis, and performance.</p>
                </div>

                <!-- QA Interface -->
                <div class="space-y-4">
                    <div class="bg-slate-900 border border-slate-800 rounded-lg p-6 space-y-4">
                        <label class="block text-sm font-semibold text-slate-300">Select a Common Literature Query:</label>
                        <div class="grid grid-cols-2 gap-3">
                            <button onclick="askQuestion(0)" class="text-left bg-slate-800 hover:bg-slate-700 border border-slate-700 p-3 rounded text-xs transition-colors">"Which catalyst produced the highest HER?"</button>
                            <button onclick="askQuestion(1)" class="text-left bg-slate-800 hover:bg-slate-700 border border-slate-700 p-3 rounded text-xs transition-colors">"What synthesis method is most associated with high hydrogen production?"</button>
                            <button onclick="askQuestion(2)" class="text-left bg-slate-800 hover:bg-slate-700 border border-slate-700 p-3 rounded text-xs transition-colors">"Which papers report Pt-doped g-C3N4?"</button>
                            <button onclick="askQuestion(3)" class="text-left bg-slate-800 hover:bg-slate-700 border border-slate-700 p-3 rounded text-xs transition-colors">"Which papers report Au-doped TiO2?"</button>
                        </div>
                    </div>

                    <!-- Output Console -->
                    <div class="bg-slate-900 border border-slate-800 rounded-lg overflow-hidden">
                        <div class="bg-slate-850 px-6 py-3 border-b border-slate-800 text-xs font-bold uppercase tracking-wider text-slate-400 flex justify-between">
                            <span>RAG Response Console</span>
                            <span class="text-teal-400">Offline traversal mode</span>
                        </div>
                        <div class="p-6 font-mono text-sm min-h-[150px] whitespace-pre-line leading-relaxed text-slate-200" id="rag-output">
                            Select a query above or enter a search node in the sidebar to begin.
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- TAB 4: Discovery Engine -->
        <div id="panel-discovery" class="tab-panel w-full h-full p-8 hidden overflow-y-auto">
            <div class="max-w-6xl mx-auto space-y-8">
                <div>
                    <h2 class="text-2xl font-bold text-teal-400 mb-2">Graph-Based Discovery Console</h2>
                    <p class="text-sm text-slate-400">Uses link prediction to suggest unexplored host-dopant combinations and detect gaps in the published literature.</p>
                </div>

                <div class="grid grid-cols-2 gap-8">
                    <!-- Column 1: Novel Combinations -->
                    <div class="space-y-4">
                        <h3 class="text-lg font-bold text-emerald-400">Suggested Catalyst Formulations</h3>
                        <div class="space-y-3" id="novel-list">
                            <!-- Populated dynamically -->
                        </div>
                    </div>

                    <!-- Column 2: Literature Gaps -->
                    <div class="space-y-4">
                        <h3 class="text-lg font-bold text-amber-400">Identified Publication Gaps</h3>
                        <div class="space-y-3" id="gap-list">
                            <!-- Populated dynamically -->
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- TAB 5: Model Explainability -->
        <div id="panel-explain" class="tab-panel w-full h-full p-8 hidden overflow-y-auto">
            <div class="max-w-4xl mx-auto space-y-8">
                <div>
                    <h2 class="text-2xl font-bold text-teal-400 mb-2">Prediction Explainability Console</h2>
                    <p class="text-sm text-slate-400">Links ML regression predictions back to the knowledge graph to show how known literature validates the output.</p>
                </div>

                <div class="space-y-6">
                    <div class="bg-slate-900 border border-slate-800 rounded-lg p-6 space-y-4">
                        <label class="block text-sm font-semibold text-slate-300">Select a Catalyst Formulation for Explanation Demo:</label>
                        <div class="flex gap-3">
                            <button onclick="showExplainerDemo(0)" class="bg-teal-600 hover:bg-teal-500 px-4 py-2 rounded text-xs font-semibold">Pt / TiO2 (1.0 wt%)</button>
                            <button onclick="showExplainerDemo(1)" class="bg-slate-800 hover:bg-slate-700 px-4 py-2 rounded text-xs font-semibold">NiS / g-C3N4 (2.0 wt%)</button>
                            <button onclick="showExplainerDemo(2)" class="bg-slate-800 hover:bg-slate-700 px-4 py-2 rounded text-xs font-semibold">Pt / CdS (0.5 wt%)</button>
                        </div>
                    </div>

                    <!-- Explanation Detail Block -->
                    <div class="bg-slate-900 border border-slate-800 rounded-lg p-6 space-y-6 hidden" id="explanation-container">
                        <!-- Head -->
                        <div class="flex justify-between items-center border-b border-slate-800 pb-4">
                            <div>
                                <h3 class="text-lg font-bold text-slate-100" id="exp-title">Pt / TiO2 (1.0 wt%)</h3>
                                <p class="text-xs text-slate-400 mt-1">Prediction Value: <span class="font-bold text-slate-200">5000 µmol/g/h</span></p>
                            </div>
                            <!-- Confidence Badge -->
                            <div class="text-right">
                                <span class="text-xs font-semibold text-slate-500 uppercase tracking-wider">Confidence Score</span>
                                <div class="text-2xl font-bold text-teal-400" id="exp-confidence">85%</div>
                            </div>
                        </div>

                        <!-- Content Layout -->
                        <div class="grid grid-cols-2 gap-8">
                            <!-- Left: Reasons -->
                            <div class="space-y-4">
                                <h4 class="text-xs font-bold uppercase tracking-wider text-slate-400">Confidence Justification</h4>
                                <ul class="space-y-2 text-xs" id="exp-reasons"></ul>
                            </div>
                            <!-- Right: Literature -->
                            <div class="space-y-4">
                                <h4 class="text-xs font-bold uppercase tracking-wider text-slate-400">Literature Anchors (Same Semiconductor)</h4>
                                <div class="space-y-3" id="exp-literature"></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

    </div>

    <script>
        // Graph Data injected by Python
        const graphData = GRAPH_DATA_PLACEHOLDER;
        const novelCandidates = NOVEL_CANDIDATES_PLACEHOLDER;
        const literatureGaps = LITERATURE_GAPS_PLACEHOLDER;
        const recommendations = RECOMMENDATIONS_PLACEHOLDER;
        const explainerDemos = EXPLAINER_DEMOS_PLACEHOLDER;
        const qaPairs = QA_PAIRS_PLACEHOLDER;

        // Node Color Mapping
        const colors = {
            "Catalyst": "#10b981", 
            "Dopant": "#f59e0b", 
            "ExperimentalCondition": "#0ea5e9", 
            "PerformanceMetric": "#f43f5e", 
            "Publication": "#6366f1", 
            "Model": "#d946ef", 
            "CodeEntity": "#64748b",
            "Concept": "#84cc16",
            "Dataset": "#a855f7"
        };

        // Switch active tab view
        function switchTab(tabId) {
            document.querySelectorAll(".tab-btn").forEach(btn => btn.classList.remove("active"));
            document.querySelectorAll(".tab-panel").forEach(panel => panel.classList.add("hidden"));
            
            document.getElementById("tab-" + tabId).classList.add("active");
            document.getElementById("panel-" + tabId).classList.remove("hidden");
        }

        // --- D3 Network Graph Logic ---
        const width = document.getElementById('d3-container').clientWidth;
        const height = document.getElementById('d3-container').clientHeight;
        const svg = d3.select("#dashboard-svg");
        const g = svg.append("g");

        const zoom = d3.zoom()
            .scaleExtent([0.1, 4])
            .on("zoom", (event) => g.attr("transform", event.transform));
        svg.call(zoom);

        document.getElementById("nodes-count").innerText = graphData.nodes.length;
        document.getElementById("edges-count").innerText = graphData.links.length;

        // Force Simulation
        const simulation = d3.forceSimulation(graphData.nodes)
            .force("link", d3.forceLink(graphData.links).id(d => d.id).distance(60))
            .force("charge", d3.forceManyBody().strength(-100))
            .force("center", d3.forceCenter(width / 2, height / 2))
            .force("collision", d3.forceCollide().radius(22));

        const link = g.append("g")
            .selectAll("line")
            .data(graphData.links)
            .enter().append("line")
            .attr("stroke", "#1e293b")
            .attr("stroke-opacity", 0.6)
            .attr("stroke-width", 1.5);

        const node = g.append("g")
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

        node.append("circle")
            .attr("r", d => d.label === "Catalyst" ? 8 : 5)
            .attr("fill", d => colors[d.label] || "#94a3b8");

        node.append("text")
            .attr("dx", 10)
            .attr("dy", ".35em")
            .text(d => d.name || d.title || d.id)
            .attr("font-size", "8px")
            .attr("fill", "#64748b")
            .style("pointer-events", "none");

        simulation.on("tick", () => {
            link
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);
            node.attr("transform", d => `translate(${d.x},${d.y})`);
        });

        // Legend filters
        const filtersDiv = document.getElementById("legend-filters");
        for (const [lbl, col] of Object.entries(colors)) {
            const hasNodes = graphData.nodes.some(n => n.label === lbl);
            if (hasNodes) {
                filtersDiv.innerHTML += `
                    <label class="flex items-center gap-2 mb-1.5 cursor-pointer">
                        <input type="checkbox" checked onchange="toggleFilter(this, '${lbl}')" class="rounded border-slate-700 bg-slate-800 text-teal-500 focus:ring-0">
                        <span class="w-2.5 h-2.5 rounded-full" style="background-color: ${col}"></span>
                        <span class="text-slate-300 font-medium">${lbl}</span>
                    </label>
                `;
            }
        }

        function toggleFilter(checkbox, label) {
            const activeLabels = [];
            document.querySelectorAll("#legend-filters input").forEach((input, idx) => {
                const labelName = Object.keys(colors)[idx];
                // Simple fetch matching order
            });
            // Apply simple Tailwind classes
            node.style("opacity", d => {
                // Fetch toggle state
                return 1;
            });
        }

        // Zoom helper
        function resetZoom() {
            svg.transition().duration(750).call(zoom.transform, d3.zoomIdentity);
        }

        function dragstarted(event, d) {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x; d.fy = d.y;
        }
        function dragged(event, d) {
            d.fx = event.x; d.fy = event.y;
        }
        function dragended(event, d) {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null; d.fy = null;
        }

        // Details drawer logic
        function showDetails(d) {
            const drawer = document.getElementById("info-drawer");
            const content = document.getElementById("info-content");
            drawer.classList.remove("hidden");
            
            let html = `<div class="mb-4"><span class="px-2 py-0.5 rounded text-[10px] uppercase font-bold" style="background-color: ${colors[d.label]}33; color: ${colors[d.label]}">${d.label}</span></div>`;
            html += `<h4 class="text-sm font-bold text-slate-100 mb-4">${d.name || d.title || d.id}</h4>`;
            
            for (const [k, v] of Object.entries(d)) {
                if (!['x','y','vx','vy','index','id','label'].includes(k)) {
                    html += `
                        <div class="border-b border-slate-800/80 pb-2">
                            <span class="text-[9px] uppercase tracking-wider text-slate-500 block">${k}</span>
                            <span class="text-xs text-slate-300">${v}</span>
                        </div>
                    `;
                }
            }
            content.innerHTML = html;
        }
        function hideDrawer() {
            document.getElementById("info-drawer").classList.add("hidden");
        }

        // Tooltip handlers
        const tooltip = d3.select("#tooltip");
        function showTooltip(event, d) {
            tooltip.classed("hidden", false)
                .html(`
                    <div class="font-bold text-teal-400 text-xs mb-0.5">${d.label}</div>
                    <div class="text-slate-200 text-xs font-medium">${d.name || d.title || d.id}</div>
                `)
                .style("left", (event.pageX + 12) + "px")
                .style("top", (event.pageY - 12) + "px");
        }
        function hideTooltip() {
            tooltip.classed("hidden", true);
        }

        // Search in canvas
        d3.select("#graph-search").on("input", function() {
            const q = this.value.toLowerCase();
            node.selectAll("circle")
                .attr("stroke", d => (d.name || d.title || d.id).toLowerCase().includes(q) && q !== "" ? "#f59e0b" : "none")
                .attr("stroke-width", d => (d.name || d.title || d.id).toLowerCase().includes(q) && q !== "" ? 3 : 0);
        });

        // --- TAB 2: Catalyst Table Explorer ---
        const tableBody = document.getElementById("catalyst-table-body");
        const catalysts = graphData.nodes.filter(n => n.label === "Catalyst");
        
        function populateCatalystTable(catsList) {
            tableBody.innerHTML = "";
            catsList.forEach(c => {
                tableBody.innerHTML += `
                    <tr class="hover:bg-slate-800/30 transition-colors">
                        <td class="p-4 font-semibold text-teal-400">${c.name || c.id}</td>
                        <td class="p-4 text-slate-300">${c.host_material || 'N/A'}</td>
                        <td class="p-4 text-slate-400">${c.semiconductor_2 || 'none'}</td>
                        <td class="p-4 text-slate-300">${c.co_catalyst || 'none'}</td>
                        <td class="p-4 text-slate-400">${c.co_catalyst_wt_pct || '0.0'}</td>
                    </tr>
                `;
            });
        }
        populateCatalystTable(catalysts);

        function filterCatalysts() {
            const query = document.getElementById("explorer-filter").value.toLowerCase();
            const filtered = catalysts.filter(c => 
                (c.name || "").toLowerCase().includes(query) || 
                (c.host_material || "").toLowerCase().includes(query) ||
                (c.co_catalyst || "").toLowerCase().includes(query)
            );
            populateCatalystTable(filtered);
        }

        // --- TAB 3: GraphRAG Q&A Console ---
        const questionsList = [
            "Which catalyst produced the highest HER?",
            "What synthesis method is most associated with high hydrogen production?",
            "Which papers report Pt-doped g-C3N4?",
            "Which papers report Au-doped TiO2?"
        ];
        function askQuestion(index) {
            const output = document.getElementById("rag-output");
            const q = questionsList[index];
            output.innerText = `>>> Query: "${q}"\n\n` + (qaPairs[q] || "No answer found.");
        }

        // --- TAB 4: Discovery Lists ---
        const novelList = document.getElementById("novel-list");
        novelCandidates.forEach(cand => {
            novelList.innerHTML += `
                <div class="bg-slate-900 border border-slate-800 rounded-lg p-4 flex justify-between items-center hover:border-emerald-500/30 transition-colors">
                    <div>
                        <div class="font-bold text-slate-200">${cand.host_material} <span class="text-emerald-400">+ ${cand.co_catalyst}</span></div>
                        <div class="text-[10px] text-slate-400 mt-1">${cand.rationale}</div>
                    </div>
                    <div class="text-right">
                        <span class="text-[9px] uppercase font-bold text-slate-500 block">Discovery Score</span>
                        <span class="text-sm font-bold text-teal-400">${cand.discovery_score}</span>
                    </div>
                </div>
            `;
        });

        const gapList = document.getElementById("gap-list");
        if (literatureGaps.length === 0) {
            gapList.innerHTML = `<div class="text-slate-500 text-xs">No significant thematic gaps identified in active communities.</div>`;
        } else {
            literatureGaps.forEach(gap => {
                gapList.innerHTML += `
                    <div class="bg-slate-900 border border-slate-800 rounded-lg p-4 hover:border-amber-500/30 transition-colors">
                        <div class="flex justify-between items-center mb-1">
                            <span class="px-2 py-0.5 rounded text-[9px] font-bold bg-amber-500/10 text-amber-400">Community ${gap.community_id}</span>
                            <span class="text-[10px] text-slate-500">${gap.gap_type}</span>
                        </div>
                        <div class="text-xs text-slate-300 font-semibold mt-2">${gap.opportunity}</div>
                    </div>
                `;
            });
        }

        // --- TAB 5: Explainability Demo ---
        function showExplainerDemo(index) {
            const exp = explainerDemos[index];
            const container = document.getElementById("explanation-container");
            container.classList.remove("hidden");
            
            const feats = exp.prediction.features;
            document.getElementById("exp-title").innerText = `${feats.host_material} + ${feats.co_catalyst} (${feats.co_catalyst_wt_pct} wt%)`;
            document.getElementById("exp-confidence").innerText = `${Math.round(exp.confidence_score * 100)}%`;
            
            // Reasons list
            const reasonsUl = document.getElementById("exp-reasons");
            reasonsUl.innerHTML = "";
            exp.confidence_reasons.forEach(r => {
                reasonsUl.innerHTML += `<li class="flex items-start gap-2 text-slate-300"><span class="text-teal-500 font-bold">✓</span> ${r}</li>`;
            });
            
            // Literature anchors
            const litDiv = document.getElementById("exp-literature");
            litDiv.innerHTML = "";
            if (exp.literature_support.length === 0) {
                litDiv.innerHTML = `<div class="text-slate-500 text-xs">No direct literature support anchor found.</div>`;
            } else {
                exp.literature_support.forEach(l => {
                    const paper = l.references.length > 0 ? l.references[0] : "Reference Source";
                    litDiv.innerHTML += `
                        <div class="bg-slate-850 p-3 rounded border border-slate-800/80">
                            <div class="font-bold text-slate-200 text-xs">${l.catalyst_name}</div>
                            <div class="text-[10px] text-slate-400 mt-1">${paper}</div>
                            <div class="text-[10px] text-teal-400 mt-1.5 font-semibold">Reported HER: ${l.reported_HERs.join(', ')} µmol/g/h</div>
                        </div>
                    `;
                });
            }
        }
        // Initialize with first demo
        showExplainerDemo(0);
    </script>
</body>
</html>"""

    # Replace placeholders
    html_content = dashboard_html_template
    html_content = html_content.replace("GRAPH_DATA_PLACEHOLDER", json.dumps(node_link_data))
    html_content = html_content.replace("NOVEL_CANDIDATES_PLACEHOLDER", json.dumps(novel_candidates))
    html_content = html_content.replace("LITERATURE_GAPS_PLACEHOLDER", json.dumps(lit_gaps))
    html_content = html_content.replace("RECOMMENDATIONS_PLACEHOLDER", json.dumps(recommendations))
    html_content = html_content.replace("EXPLAINER_DEMOS_PLACEHOLDER", json.dumps(explainer_demos))
    html_content = html_content.replace("QA_PAIRS_PLACEHOLDER", json.dumps(qa_pairs))

    with open(DASHBOARD_HTML_PATH, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    print(f"Generated Interactive Discovery Dashboard: {DASHBOARD_HTML_PATH}")

if __name__ == "__main__":
    compile_dashboard()
