"""
serve.py
Model Context Protocol (MCP) Server for Photocatalyst Graphify.
Exposes tools for GraphRAG query answering, catalyst recommendations, predictive explanations,
and gap discoveries.
"""

import sys
import json
import argparse
from .rag import GraphRAG
from .explain import PredictExplainer
from .discovery import DiscoveryEngine

def run_cli_command(args):
    """Fallback CLI mode for quick queries and testing."""
    rag = GraphRAG()
    explainer = PredictExplainer()
    discovery = DiscoveryEngine()

    if args.action == "query":
        ans = rag.answer_question(args.query)
        print(ans)
    elif args.action == "recommend":
        recs = rag.recommend(args.host)
        print(json.dumps(recs, indent=2))
    elif args.action == "explain":
        features = {
            "host_material": args.host,
            "co_catalyst": args.dopant,
            "co_catalyst_wt_pct": args.wt_pct,
            "pH": args.ph,
            "temperature_C": args.temp
        }
        exp = explainer.explain_prediction(features, args.her)
        explainer.print_ascii_explanation(exp)
    elif args.action == "discover":
        candidates = discovery.discover_novel_combinations()
        print(json.dumps(candidates, indent=2))
    else:
        print("Invalid action. Use --help for usage details.")

def run_mcp_server():
    """
    Standard Model Context Protocol (MCP) stdio server implementation.
    Allows AI coding assistants to call graphify tools directly.
    """
    rag = GraphRAG()
    explainer = PredictExplainer()
    discovery = DiscoveryEngine()

    print("Photocatalyst Graphify MCP server started via stdio...", file=sys.stderr)
    
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
                
            request = json.loads(line)
            req_id = request.get("id")
            method = request.get("method")
            params = request.get("params", {})
            
            # Simple JSON-RPC execution
            result = None
            error = None
            
            if method == "tools/list":
                result = {
                    "tools": [
                        {
                            "name": "graphify_rag_query",
                            "description": "Ask the knowledge graph questions about catalysts, papers, or synthesis methods.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {"query": {"type": "string"}},
                                "required": ["query"]
                            }
                        },
                        {
                            "name": "graphify_recommend",
                            "description": "Get dopant/co-catalyst recommendations for a host semiconductor.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {"host": {"type": "string"}},
                                "required": ["host"]
                            }
                        },
                        {
                            "name": "graphify_explain",
                            "description": "Retrieve model predictions and explain them using literature supports from the graph.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "host": {"type": "string"},
                                    "dopant": {"type": "string"},
                                    "her": {"type": "number"},
                                    "wt_pct": {"type": "number", "default": 1.0},
                                    "ph": {"type": "number", "default": 7.0},
                                    "temp": {"type": "number", "default": 25.0}
                                },
                                "required": ["host", "dopant", "her"]
                            }
                        },
                        {
                            "name": "graphify_discover",
                            "description": "Discover novel catalyst combinations and literature gaps.",
                            "inputSchema": {"type": "object", "properties": {}}
                        }
                    ]
                }
            elif method == "tools/call":
                name = params.get("name")
                arguments = params.get("arguments", {})
                
                if name == "graphify_rag_query":
                    result = {"content": [{"type": "text", "text": rag.answer_question(arguments.get("query"))}]}
                elif name == "graphify_recommend":
                    recs = rag.recommend(arguments.get("host"))
                    result = {"content": [{"type": "text", "text": json.dumps(recs, indent=2)}]}
                elif name == "graphify_explain":
                    features = {
                        "host_material": arguments.get("host"),
                        "co_catalyst": arguments.get("dopant"),
                        "co_catalyst_wt_pct": arguments.get("wt_pct", 1.0),
                        "pH": arguments.get("ph", 7.0),
                        "temperature_C": arguments.get("temp", 25.0)
                    }
                    exp = explainer.explain_prediction(features, arguments.get("her"))
                    # Render ASCII layout to text
                    import io
                    old_stdout = sys.stdout
                    sys.stdout = buffer = io.StringIO()
                    explainer.print_ascii_explanation(exp)
                    sys.stdout = old_stdout
                    result = {"content": [{"type": "text", "text": buffer.getvalue()}]}
                elif name == "graphify_discover":
                    cands = discovery.discover_novel_combinations()
                    result = {"content": [{"type": "text", "text": json.dumps(cands, indent=2)}]}
                else:
                    error = {"code": -32601, "message": f"Tool not found: {name}"}
            else:
                error = {"code": -32601, "message": f"Method not found: {method}"}
                
            # Send response
            response = {"jsonrpc": "2.0", "id": req_id}
            if error:
                response["error"] = error
            else:
                response["result"] = result
                
            print(json.dumps(response))
            sys.stdout.flush()
            
        except Exception as e:
            print(f"Error in MCP: {e}", file=sys.stderr)
            break

def main():
    parser = argparse.ArgumentParser(description="Photocatalyst Graphify MCP & CLI Server")
    parser.add_argument("--mcp", action="store_true", help="Start in MCP mode (stdio communication)")
    
    # CLI parameters (fallback)
    subparsers = parser.add_subparsers(dest="action", help="CLI actions")
    
    # Query parser
    qp = subparsers.add_parser("query", help="Ask a question")
    qp.add_argument("query", type=str, help="Question text")
    
    # Recommend parser
    rp = subparsers.add_parser("recommend", help="Get dopant recommendations")
    rp.add_argument("host", type=str, help="Host semiconductor")
    
    # Explain parser
    ep = subparsers.add_parser("explain", help="Explain an ML prediction")
    ep.add_argument("--host", required=True, type=str)
    ep.add_argument("--dopant", required=True, type=str)
    ep.add_argument("--her", required=True, type=float, help="Predicted HER rate")
    ep.add_argument("--wt-pct", type=float, default=1.0)
    ep.add_argument("--ph", type=float, default=7.0)
    ep.add_argument("--temp", type=float, default=25.0)
    
    # Discover parser
    subparsers.add_parser("discover", help="Discover novel combinations")

    args = parser.parse_args()
    
    if args.mcp or len(sys.argv) == 1:
        run_mcp_server()
    else:
        run_cli_command(args)

if __name__ == "__main__":
    main()
