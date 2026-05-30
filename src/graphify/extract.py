"""
extract.py
Extracts nodes and edges from heterogeneous files (Python scripts, Jupyter notebooks,
Markdown notes, PDF research papers, and CSV/JSON datasets).
"""

import os
import ast
import json
import re
import pandas as pd

# Try importing pdfplumber for PDF extraction. Degrade gracefully if unavailable.
try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

# Catalyst related keywords for entity matching in unstructured text
CATALYST_KEYWORDS = {
    "host_material": ["TiO2", "ZnO", "g-C3N4", "C3N4", "CdS", "MoS2", "NiO", "CuO", "BiOCl", "ZnIn2S4", "ZnS", "CeO2", "MIL-101", "UiO-66"],
    "dopant": ["Pt", "Au", "Ag", "Ni", "Co", "Fe", "Cu", "Ni2P", "CoSx", "NiS", "MoS2", "Fe2O3", "Cu2O"],
    "synthesis": ["hydrothermal", "sol-gel", "microwave", "calcination", "photodeposition", "solvothermal", "impregnation", "thermal decomposition"],
    "conditions": ["pH", "temperature", "light source", "glycerol", "loading", "reaction volume", "xe lamp", "led", "solar simulator"],
    "performance": ["HER", "AQY", "hydrogen evolution", "stability", "yield"]
}

def clean_node_id(name):
    """Sanitizes text to form a clean identifier."""
    return re.sub(r'[^a-zA-Z0-9_.-]', '_', str(name).strip())

def extract_python(filepath):
    """Parses a Python file using AST to extract imports, classes, functions, and docstrings."""
    nodes = []
    edges = []
    
    file_id = f"file_{clean_node_id(os.path.basename(filepath))}"
    nodes.append({
        "id": file_id,
        "label": "CodeEntity",
        "properties": {
            "name": os.path.basename(filepath),
            "type": "script",
            "path": filepath
        }
    })
    
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            code = f.read()
        
        tree = ast.parse(code)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    imp_id = f"import_{clean_node_id(name.name)}"
                    nodes.append({
                        "id": imp_id,
                        "label": "CodeEntity",
                        "properties": {"name": name.name, "type": "import"}
                    })
                    edges.append({
                        "source": file_id,
                        "target": imp_id,
                        "type": "IMPORTS",
                        "properties": {"confidence": "EXTRACTED"}
                    })
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                imp_id = f"import_{clean_node_id(module)}"
                nodes.append({
                    "id": imp_id,
                    "label": "CodeEntity",
                    "properties": {"name": module, "type": "import"}
                })
                edges.append({
                    "source": file_id,
                    "target": imp_id,
                    "type": "IMPORTS",
                    "properties": {"confidence": "EXTRACTED"}
                })
            elif isinstance(node, ast.ClassDef):
                class_id = f"class_{clean_node_id(node.name)}"
                nodes.append({
                    "id": class_id,
                    "label": "CodeEntity",
                    "properties": {
                        "name": node.name,
                        "type": "class",
                        "docstring": ast.get_docstring(node) or ""
                    }
                })
                edges.append({
                    "source": file_id,
                    "target": class_id,
                    "type": "DEFINES_CLASS",
                    "properties": {"confidence": "EXTRACTED"}
                })
            elif isinstance(node, ast.FunctionDef):
                func_id = f"func_{clean_node_id(node.name)}"
                nodes.append({
                    "id": func_id,
                    "label": "CodeEntity",
                    "properties": {
                        "name": node.name,
                        "type": "function",
                        "docstring": ast.get_docstring(node) or ""
                    }
                })
                edges.append({
                    "source": file_id,
                    "target": func_id,
                    "type": "DEFINES_FUNCTION",
                    "properties": {"confidence": "EXTRACTED"}
                })
    except Exception as e:
        print(f"Error parsing AST for {filepath}: {e}")
        
    return {"nodes": nodes, "edges": edges}

def extract_notebook(filepath):
    """Extracts cells, markdown texts, and code from Jupyter Notebooks."""
    nodes = []
    edges = []
    
    nb_id = f"notebook_{clean_node_id(os.path.basename(filepath))}"
    nodes.append({
        "id": nb_id,
        "label": "CodeEntity",
        "properties": {
            "name": os.path.basename(filepath),
            "type": "notebook",
            "path": filepath
        }
    })
    
    try:
        with open(filepath, "r", encoding="utf-8-sig") as f:
            nb = json.load(f)
            
        cells = nb.get("cells", [])
        for idx, cell in enumerate(cells):
            cell_type = cell.get("cell_type", "unknown")
            source = "".join(cell.get("source", []))
            
            if not source.strip():
                continue
                
            cell_id = f"{nb_id}_cell_{idx}"
            nodes.append({
                "id": cell_id,
                "label": "CodeEntity",
                "properties": {
                    "name": f"Cell {idx} ({cell_type})",
                    "type": f"notebook_cell_{cell_type}",
                    "content": source[:200] + ("..." if len(source) > 200 else "")
                }
            })
            edges.append({
                "source": nb_id,
                "target": cell_id,
                "type": "CONTAINS_CELL",
                "properties": {"index": idx, "confidence": "EXTRACTED"}
            })
            
            # Simple keyword matching for catalysts inside notebook comments or code
            for cat, keywords in CATALYST_KEYWORDS.items():
                for kw in keywords:
                    if re.search(r'\b' + re.escape(kw) + r'\b', source, re.IGNORECASE):
                        kw_id = f"keyword_{clean_node_id(kw)}"
                        nodes.append({
                            "id": kw_id,
                            "label": "Concept",
                            "properties": {"name": kw, "category": cat}
                        })
                        edges.append({
                            "source": cell_id,
                            "target": kw_id,
                            "type": "MENTIONS",
                            "properties": {"confidence": "INFERRED", "keyword": kw}
                        })
    except Exception as e:
        print(f"Error parsing notebook {filepath}: {e}")
        
    return {"nodes": nodes, "edges": edges}

def extract_markdown(filepath):
    """Parses headers and lists from Markdown logs or reports."""
    nodes = []
    edges = []
    
    doc_id = f"doc_{clean_node_id(os.path.basename(filepath))}"
    nodes.append({
        "id": doc_id,
        "label": "Publication",
        "properties": {
            "title": os.path.basename(filepath),
            "type": "markdown",
            "path": filepath
        }
    })
    
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        current_header_id = doc_id
        for idx, line in enumerate(lines):
            header_match = re.match(r'^(#{1,6})\s+(.*)$', line)
            if header_match:
                level = len(header_match.group(1))
                title = header_match.group(2).strip()
                header_id = f"{doc_id}_h_{idx}"
                nodes.append({
                    "id": header_id,
                    "label": "Publication",
                    "properties": {"title": title, "type": f"section_l{level}"}
                })
                edges.append({
                    "source": current_header_id,
                    "target": header_id,
                    "type": "HAS_SECTION",
                    "properties": {"confidence": "EXTRACTED"}
                })
                current_header_id = header_id
            else:
                # Text scan for catalyst keywords
                for cat, keywords in CATALYST_KEYWORDS.items():
                    for kw in keywords:
                        if re.search(r'\b' + re.escape(kw) + r'\b', line, re.IGNORECASE):
                            kw_id = f"concept_{clean_node_id(kw)}"
                            nodes.append({
                                "id": kw_id,
                                "label": "Concept",
                                "properties": {"name": kw, "category": cat}
                            })
                            edges.append({
                                "source": current_header_id,
                                "target": kw_id,
                                "type": "MENTIONS",
                                "properties": {"confidence": "INFERRED", "keyword": kw}
                            })
    except Exception as e:
        print(f"Error parsing markdown {filepath}: {e}")
        
    return {"nodes": nodes, "edges": edges}

def extract_pdf(filepath):
    """Parses a scientific paper PDF using pdfplumber to extract text and reference citations."""
    nodes = []
    edges = []
    
    pdf_id = f"doc_{clean_node_id(os.path.basename(filepath))}"
    nodes.append({
        "id": pdf_id,
        "label": "Publication",
        "properties": {
            "title": os.path.basename(filepath),
            "type": "pdf",
            "path": filepath
        }
    })
    
    if not HAS_PDFPLUMBER:
        print(f"Warning: pdfplumber not installed. Skipping PDF text parsing for {filepath}")
        return {"nodes": nodes, "edges": edges}
        
    try:
        full_text = ""
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"
                    
        # Extract title if possible (fallback to filename)
        first_line = full_text.split("\n")[0] if full_text else ""
        if first_line.strip() and len(first_line) < 150:
            nodes[0]["properties"]["title"] = first_line.strip()
            
        # Detect catalyst keywords in full text
        found_keywords = {}
        for cat, keywords in CATALYST_KEYWORDS.items():
            for kw in keywords:
                matches = list(re.finditer(r'\b' + re.escape(kw) + r'\b', full_text, re.IGNORECASE))
                if matches:
                    found_keywords[kw] = {
                        "category": cat,
                        "count": len(matches)
                    }
                    
        # Build keyword mention nodes and edges
        for kw, info in found_keywords.items():
            kw_id = f"concept_{clean_node_id(kw)}"
            nodes.append({
                "id": kw_id,
                "label": "Concept",
                "properties": {
                    "name": kw,
                    "category": info["category"]
                }
            })
            edges.append({
                "source": pdf_id,
                "target": kw_id,
                "type": "MENTIONS",
                "properties": {
                    "count": info["count"],
                    "confidence": "INFERRED"
                }
            })
            
        # Extract bibliography links (very rough regex for [1], [2], or author names)
        citations = re.findall(r'\[\s*\d+\s*\]', full_text)
        if citations:
            num_citations = len(set(citations))
            nodes[0]["properties"]["citation_count_in_text"] = num_citations

    except Exception as e:
        print(f"Error parsing PDF {filepath}: {e}")
        
    return {"nodes": nodes, "edges": edges}

def clean_dataset_value(val):
    if pd.isna(val) or val == "—" or not str(val).strip():
        return None
    cleaned = str(val).replace("~", "").replace(",", "").strip()
    cleaned = re.sub(r'\s*\(.*?\)', '', cleaned)
    cleaned = cleaned.replace("µmol h⁻¹", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        match = re.search(r'([-+]?[\d.]+)', cleaned)
        if match:
            return float(match.group(1))
        return None

def extract_dataset(filepath):
    """
    Parses a tabular scientific dataset (JSON/CSV) to extract Catalyst,
    Dopant, Synthesis, Conditions, and Performance nodes/edges.
    """
    nodes = []
    edges = []
    
    dataset_id = f"dataset_{clean_node_id(os.path.basename(filepath))}"
    nodes.append({
        "id": dataset_id,
        "label": "Dataset",
        "properties": {
            "name": os.path.basename(filepath),
            "path": filepath
        }
    })
    
    try:
        # Load file
        if filepath.endswith(".json"):
            with open(filepath, "r", encoding="utf-8-sig") as f:
                data = json.load(f)
            # If list of lists (master dataset format)
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
                headers = data[0]
                rows = data[1:]
                df = pd.DataFrame(rows, columns=headers)
            else:
                df = pd.DataFrame(data)
        elif filepath.endswith(".csv"):
            df = pd.read_csv(filepath)
        elif filepath.endswith((".xlsx", ".xlsm")):
            df = pd.read_excel(filepath, engine="openpyxl")
        else:
            return {"nodes": nodes, "edges": edges}
            
        # Standardize column mappings
        col_mapping = {
            "Catalyst name": "catalyst_name",
            "Catalyst": "catalyst_name",
            "Composition_Raw": "composition",
            "HER_Raw (µmol/g/h)": "HER_std_umol_g_h",
            "HER_Raw": "HER_std_umol_g_h",
            "Co_Catalyst": "co_catalyst",
            "Family": "family",
            "Reference": "Reference"
        }
        df = df.rename(columns=col_mapping)
        
        # Map rows into entities
        for idx, row in df.iterrows():
            row_id = f"{dataset_id}_row_{idx}"
            
            # Map host material from Family or Composition
            family = row.get("family", "")
            comp = row.get("composition", "")
            
            host = "unknown"
            if not pd.isna(family) and str(family).strip():
                host = str(family).strip()
            elif not pd.isna(comp) and str(comp).strip():
                # Extract basic semiconductor keyword
                for k in ["TiO2", "ZnO", "C3N4", "CdS", "MoS2", "ZnS", "MIL-101", "UiO-66"]:
                    if k.lower() in str(comp).lower():
                        host = k
                        break
            
            cat_name = row.get("catalyst_name", "")
            if pd.isna(cat_name) or not str(cat_name).strip():
                cat_name = f"{host}_catalyst"
                
            cat_id = f"cat_{clean_node_id(cat_name)}"
            
            # Define Catalyst Node
            nodes.append({
                "id": cat_id,
                "label": "Catalyst",
                "properties": {
                    "name": str(cat_name),
                    "host_material": host,
                    "composition": str(comp) if not pd.isna(comp) else ""
                }
            })
            
            # Connect Catalyst to Dataset
            edges.append({
                "source": dataset_id,
                "target": cat_id,
                "type": "CONTAINS_CATALYST",
                "properties": {"row_index": idx, "confidence": "EXTRACTED"}
            })
            
            # 2. Co-catalysts/Dopants
            co_cat = row.get("co_catalyst", "none")
            if not pd.isna(co_cat) and str(co_cat).lower() != "none" and str(co_cat).lower() != "unknown" and str(co_cat).strip() != "—":
                dop_id = f"dop_{clean_node_id(co_cat)}"
                
                # Parse weight percent from composition if possible
                wt_pct = 0.0
                comp_str = str(comp)
                if not pd.isna(comp):
                    match = re.search(r'([\d.]+)\s*wt\s*%\s*' + re.escape(str(co_cat)), comp_str, re.IGNORECASE)
                    if match:
                        wt_pct = float(match.group(1))
                    else:
                        match = re.search(r'([\d.]+)\s*mol\s*%\s*' + re.escape(str(co_cat)), comp_str, re.IGNORECASE)
                        if match:
                            wt_pct = float(match.group(1))
                            
                nodes.append({
                    "id": dop_id,
                    "label": "Dopant",
                    "properties": {
                        "name": str(co_cat),
                        "wt_percent": wt_pct
                    }
                })
                edges.append({
                    "source": cat_id,
                    "target": dop_id,
                    "type": "DOPED_WITH",
                    "properties": {
                        "wt_pct": wt_pct,
                        "confidence": "EXTRACTED"
                    }
                })
                
            # 3. Experimental conditions
            cond_id = f"cond_{row_id}"
            cond_props = {}
            
            # Look for explicit condition columns
            for col in ["pH", "temperature_C", "glycerol_concentration_std", "catalyst_loading_mg", "reaction_volume_mL", "light_power_W", "wavelength_cutoff_nm"]:
                if col in row and not pd.isna(row[col]):
                    val = clean_dataset_value(row[col])
                    if val is not None:
                        cond_props[col] = val
            
            # Map Light Source raw string
            light_src = row.get("Light_Source", "")
            if not pd.isna(light_src) and str(light_src).strip() and str(light_src).strip() != "—":
                cond_props["light_source"] = str(light_src).strip()
            
            if cond_props:
                nodes.append({
                    "id": cond_id,
                    "label": "ExperimentalCondition",
                    "properties": cond_props
                })
                edges.append({
                    "source": cat_id,
                    "target": cond_id,
                    "type": "TESTED_UNDER",
                    "properties": {"confidence": "EXTRACTED"}
                })
                
            # 4. Performance Nodes
            her_raw = row.get("HER_std_umol_g_h", 0.0)
            her_val = clean_dataset_value(her_raw)
            
            if her_val is not None and her_val > 0:
                perf_id = f"perf_{row_id}"
                
                aqy_raw = row.get("AQY_Raw (%)", row.get("AQY_Raw", 0.0))
                aqy_val = clean_dataset_value(aqy_raw)
                
                sth_raw = row.get("STH_Raw (%)", row.get("STH_Raw", 0.0))
                sth_val = clean_dataset_value(sth_raw)
                
                nodes.append({
                    "id": perf_id,
                    "label": "PerformanceMetric",
                    "properties": {
                        "HER_std_umol_g_h": float(her_val),
                        "AQY_pct": float(aqy_val) if aqy_val is not None else 0.0,
                        "STH_pct": float(sth_val) if sth_val is not None else 0.0
                    }
                })
                edges.append({
                    "source": cat_id,
                    "target": perf_id,
                    "type": "PRODUCED_HER",
                    "properties": {
                        "value": float(her_val),
                        "confidence": "EXTRACTED"
                    }
                })
                if cond_props:
                    edges.append({
                        "source": cond_id,
                        "target": perf_id,
                        "type": "RESULTED_IN",
                        "properties": {"confidence": "EXTRACTED"}
                    })
                    
            # 5. References/Publications
            ref_paper = row.get("Reference", "")
            if not pd.isna(ref_paper) and str(ref_paper).strip() and str(ref_paper).strip() != "—":
                ref_id = f"pub_{clean_node_id(ref_paper)[:50]}"
                nodes.append({
                    "id": ref_id,
                    "label": "Publication",
                    "properties": {
                        "title": str(ref_paper),
                        "citation": str(ref_paper),
                        "type": "citation"
                    }
                })
                edges.append({
                    "source": cat_id,
                    "target": ref_id,
                    "type": "CATALYST_USED_IN_PAPER",
                    "properties": {"confidence": "EXTRACTED"}
                })
                if her_val is not None and her_val > 0:
                    edges.append({
                        "source": perf_id,
                        "target": ref_id,
                        "type": "REPORTED_IN",
                        "properties": {"confidence": "EXTRACTED"}
                    })
    except Exception as e:
        print(f"Error parsing dataset {filepath}: {e}")
        
    return {"nodes": nodes, "edges": edges}


def extract_file(filepath):
    """Delegates extraction based on file extension."""
    ext = os.path.splitext(filepath)[1].lower()
    
    if ext == ".py":
        return extract_python(filepath)
    elif ext == ".ipynb":
        return extract_notebook(filepath)
    elif ext == ".md":
        return extract_markdown(filepath)
    elif ext == ".pdf":
        return extract_pdf(filepath)
    elif ext in {".json", ".csv", ".xlsx", ".xlsm"}:
        return extract_dataset(filepath)
    else:
        return {"nodes": [], "edges": []}
