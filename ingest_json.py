import json
import re
import os

def ingest_json_to_html(
    html_path="photocatalytic_H2_glycerol_catalyst_table.html"
):
    # Determine correct JSON path
    json_path_candidates = ["1779702377357_combined_catalyst_dataset_16col.json", "combined_catalyst_dataset_16col.json"]
    json_path = None
    for path in json_path_candidates:
        if os.path.exists(path):
            json_path = path
            break
            
    if not json_path:
        print(f"Error: Could not find combined catalyst JSON file.")
        exit(1)
        
    print(f"Using JSON file: {json_path}")

    # 1. Read JSON, skip header row (index 0)
    with open(json_path, "r", encoding="utf-8") as f:
        all_rows = json.load(f)
    data_rows = all_rows[1:]   # skip header at index 0
    print(f"Loaded {len(data_rows)} data entries from JSON.")

    # 2. Read current HTML and extract existing catalyst names
    with open(html_path, "r", encoding="utf-8") as f:
        content = f.read()
    # Extract all existing first-element values from raw array
    existing_names = set(re.findall(r'\["([^"]+)",\s*"[^"]*",', content))
    print(f"Existing entries in HTML: {len(existing_names)} unique names found.")

    # 3. Filter to only genuinely new entries
    # Also apply outlier exclusions and fix duplicate name
    EXCLUDE_NAMES = {"0.5wt%Pt/TiO2", "Bi2WO6", "Cu/TiO2 (Thermal, 70C)"}
    seen_names = set()
    new_rows = []
    for row in data_rows:
        name = row[0]
        if name in EXCLUDE_NAMES:
            print(f"  EXCLUDED (outlier): {name}")
            continue
        if name in existing_names:
            continue
        # Handle duplicate SG1 name
        if name == "SG1(10%)TiO2/MC-H" and name in seen_names:
            name = "SG1(10%)TiO2/MC-H (v2)"
            row = [name] + row[1:]
        seen_names.add(name)
        new_rows.append(row)

    print(f"New entries to append: {len(new_rows)}")

    # 4. Append to HTML raw array
    idx_raw = content.find("const raw=[")
    if idx_raw == -1:
        # try without spaces or other formatting
        idx_raw = content.find("const raw = [")
    if idx_raw == -1:
        print("Error: Could not find 'const raw=[' array in HTML.")
        exit(1)
        
    idx_close = content.find("];", idx_raw)
    if idx_close == -1:
        print("Error: Could not find closing '];' of raw array.")
        exit(1)

    new_js_rows = []
    for row in new_rows:
        # Clean each field: escape quotes, strip unicode special chars
        cleaned = []
        for field in row:
            s = str(field).replace('"', '\\"')
            cleaned.append(s)
        js_row = '["' + '", "'.join(cleaned) + '"]'
        new_js_rows.append(js_row)

    new_content = content[:idx_close].rstrip()
    for js_row in new_js_rows:
        # If the last character is not a comma or bracket, add a comma
        if not new_content.endswith(",") and not new_content.endswith("["):
            new_content += ","
        new_content += f"\n  {js_row}"
    new_content += "\n];"
    new_content += content[idx_close + 2:]

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(new_content)

    print(f"Successfully appended {len(new_rows)} new entries.")
    return len(new_rows)

if __name__ == "__main__":
    added = ingest_json_to_html()
    print(f"\nTotal new entries added: {added}")
    print("Run data_clean.py next.")
