"""
run_all.py — Execute the full pipeline end-to-end.
Usage:
    python run_all.py                          # full run
    python run_all.py --skip-preprocess        # skip preprocessing
    python run_all.py --skip-train             # skip training
"""
import subprocess, argparse, time

def run(cmd):
    print(f"\n{'='*60}\nRunning: {cmd}\n{'='*60}")
    t0 = time.time()
    subprocess.run(cmd, shell=True, check=True)
    print(f"Done in {time.time() - t0:.1f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--skip-train",      action="store_true")
    args = parser.parse_args()

    if not args.skip_preprocess:
        run("python src/preprocess.py")
    if not args.skip_train:
        run("python src/train.py")
    run("python src/evaluate.py")
    print("\nPipeline complete. All results are in data/results/")
