import patches
import argparse
import sys
import subprocess
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Add new experimental catalyst data and retrain model.")
    parser.add_argument("--catalyst", type=str, help="Catalyst name (e.g. 'Pt/TiO2-New')")
    parser.add_argument("--composition", type=str, help="Composition description (e.g. 'TiO2 + 1 wt% Pt')")
    parser.add_argument("--bandgap", type=str, help="Bandgap in eV (e.g. '3.2')")
    parser.add_argument("--vb", type=str, help="VB edge vs NHE in V (e.g. '+2.7')")
    parser.add_argument("--cb", type=str, help="CB edge vs NHE in V (e.g. '-0.5')")
    parser.add_argument("--hj", type=str, help="Heterojunction type (e.g. 'None', 'Z-scheme', 'Type-II')")
    parser.add_argument("--cocat", type=str, help="Co-catalyst (e.g. 'Pt', 'Cu', 'None')")
    parser.add_argument("--light", type=str, help="Light source details (e.g. 'Solar simulator AM 1.5G')")
    parser.add_argument("--her", type=str, help="HER in umol g-1 h-1 (e.g. '9500')")
    parser.add_argument("--aqy", type=str, default="—", help="AQY in % (e.g. '5.2', or '—')")
    parser.add_argument("--sth", type=str, default="—", help="STH in % (e.g. '0.1', or '—')")
    parser.add_argument("--bet", type=str, default="—", help="BET surface area in m2/g (e.g. '50', or '—')")
    parser.add_argument("--photocurrent", type=str, default="—", help="Photocurrent in uA cm-2 (e.g. '120', or '—')")
    parser.add_argument("--lifetime", type=str, default="—", help="Carrier lifetime description (e.g. '1.5 ns', or '—')")
    parser.add_argument("--family", type=str, choices=["TiO2", "CdS", "g-C3N4", "ZnS", "MOF", "Other"], help="Material family")
    parser.add_argument("--ref", type=str, default="Lab experiment, 2026", help="Reference citation")
    return parser.parse_known_args()

def prompt_interactive():
    print("\n--- Enter New Experimental Laboratory Results ---")
    data = {}
    data["catalyst"] = input("Catalyst name (e.g. Pt/TiO2-New): ").strip()
    data["composition"] = input("Composition description (e.g. TiO2 + 1 wt% Pt): ").strip()
    data["bandgap"] = input("Bandgap in eV (e.g. 3.2): ").strip()
    data["vb"] = input("VB edge vs NHE in V (e.g. +2.7): ").strip()
    data["cb"] = input("CB edge vs NHE in V (e.g. -0.5): ").strip()
    data["hj"] = input("Heterojunction type (None/Z-scheme/Type-II/Schottky): ").strip() or "None"
    data["cocat"] = input("Co-catalyst (Pt/Cu/NiS/None): ").strip() or "None"
    data["light"] = input("Light source details: ").strip() or "Solar simulator"
    data["her"] = input("HER in umol g-1 h-1 (e.g. 9500): ").strip()
    data["aqy"] = input("AQY in % (or press Enter for —): ").strip() or "—"
    data["sth"] = input("STH in % (or press Enter for —): ").strip() or "—"
    data["bet"] = input("BET surface area in m2/g (or press Enter for —): ").strip() or "—"
    data["photocurrent"] = input("Photocurrent in uA cm-2 (or press Enter for —): ").strip() or "—"
    data["lifetime"] = input("Carrier lifetime (or press Enter for —): ").strip() or "—"
    
    while True:
        fam = input("Material family (TiO2/CdS/g-C3N4/ZnS/MOF/Other): ").strip()
        if fam in ["TiO2", "CdS", "g-C3N4", "ZnS", "MOF", "Other"]:
            data["family"] = fam
            break
        print("Invalid family. Please choose from: TiO2, CdS, g-C3N4, ZnS, MOF, Other.")
        
    data["ref"] = input("Reference (default: Lab experiment, 2026): ").strip() or "Lab experiment, 2026"
    return data

def main():
    args, unknown = parse_args()
    
    # If key fields are missing from arguments, fall back to interactive prompting
    if not (args.catalyst and args.composition and args.bandgap and args.vb and args.cb and args.her and args.family):
        data = prompt_interactive()
    else:
        data = vars(args)
        
    # Format as Javascript row:
    # Columns: [name, composition, Eg, VB, CB, HJ, cocatalyst, lightsource, HER, AQY, STH, BET, photocurrent, lifetime, family, reference]
    new_row_js = f'["{data["catalyst"]}", "{data["composition"]}", "{data["bandgap"]}", "{data["vb"]}", "{data["cb"]}", "{data["hj"]}", "{data["cocat"]}", "{data["light"]}", "{data["her"]}", "{data["aqy"]}", "{data["sth"]}", "{data["bet"]}", "{data["photocurrent"]}", "{data["lifetime"]}", "{data["family"]}", "{data["ref"]}"]'
    
    html_path = "photocatalytic_H2_glycerol_catalyst_table.html"
    if not os.path.exists(html_path):
        print(f"Error: {html_path} not found.")
        return
        
    print(f"\nAppending new experiment to {html_path}...")
    
    # Read HTML content
    with open(html_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    # Locate the closing bracket of raw array
    idx_raw = content.find("const raw=[")
    if idx_raw == -1:
        print("Error: Could not find 'const raw=[' array in HTML.")
        return
        
    idx_close = content.find("];", idx_raw)
    if idx_close == -1:
        print("Error: Could not find closing '];' of raw array in HTML.")
        return
        
    # Insert new row before the closing bracket
    new_content = content[:idx_close].rstrip()
    if new_content.endswith(","):
        new_content += f"\n  {new_row_js}\n];"
    else:
        new_content += f",\n  {new_row_js}\n];"
    new_content += content[idx_close+2:]
    
    # Write back modified HTML
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(new_content)
        
    print("Successfully updated database in HTML.")
    
    # Trigger active learning loop: sanitization and retraining
    print("\n--- Starting Closed-Loop Active Learning Pipeline ---")
    
    print("\nStep 1: Running Data Sanitization & Cleaning...")
    import data_clean
    data_clean.main()
    
    print("\nStep 2: Triggering AutoML Model Retraining...")
    # Run pipeline_train.py in a subprocess
    python_exe = sys.executable
    try:
        subprocess.run([python_exe, "pipeline_train.py"], check=True)
        print("\n=== Closed-Loop Retraining Successfully Completed! ===")
    except subprocess.CalledProcessError as e:
        print(f"\nError: Model retraining failed with exit code {e.returncode}")
        sys.exit(e.returncode)

if __name__ == "__main__":
    main()
