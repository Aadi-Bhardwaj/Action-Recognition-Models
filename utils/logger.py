import csv
import os

def log_results_to_csv(results, filename="results/conv3d_results.csv"):
    if not results:
        print("[ERROR] No results to write. The 'results' list is empty.")
        return

    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"[INFO] Results successfully written to {filename}")
    except Exception as e:
        print(f"[ERROR] Failed to write CSV: {e}")
