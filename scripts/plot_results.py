import csv
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "benchmark_results.csv"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

rows = []

with open(CSV_PATH, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

def f(row, key):
    return float(row[key])

# 1. Recall vs latency
plt.figure()
for row in rows:
    if f(row, "recall_at_k") > 0:
        plt.scatter(f(row, "avg_ms"), f(row, "recall_at_k"))
        plt.text(f(row, "avg_ms"), f(row, "recall_at_k"), row["index_name"], fontsize=8)

plt.xlabel("Average Latency (ms)")
plt.ylabel("Recall@10")
plt.title("Recall vs Latency")
plt.grid(True)
plt.savefig(RESULTS_DIR / "recall_vs_latency.png", dpi=200, bbox_inches="tight")
plt.close()

# 2. Memory usage, deduplicated by index/config family
memory_rows = []

seen = set()
for row in rows:
    name = row["index_name"]

    if name == "IVF-Flat":
        label = "IVF-Flat"
    else:
        label = name

    if label in seen:
        continue

    seen.add(label)
    memory_rows.append((label, f(row, "memory_bytes") / (1024 * 1024)))

names = [x[0] for x in memory_rows]
memory_mb = [x[1] for x in memory_rows]

plt.figure(figsize=(9, 5))
plt.bar(names, memory_mb)
plt.xticks(rotation=35, ha="right")
plt.ylabel("Memory (MB)")
plt.title("Index Memory Usage")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "memory_usage.png", dpi=200)
plt.close()

# 3. IVF-Flat recall-latency curve
ivf_rows = [row for row in rows if row["index_name"] == "IVF-Flat"]
ivf_rows.sort(key=lambda r: int(r["nprobe"]))

plt.figure()
plt.plot([f(r, "avg_ms") for r in ivf_rows], [f(r, "recall_at_k") for r in ivf_rows], marker="o")

for r in ivf_rows:
    plt.text(f(r, "avg_ms"), f(r, "recall_at_k"), f"nprobe={r['nprobe']}", fontsize=8)

plt.xlabel("Average Latency (ms)")
plt.ylabel("Recall@10")
plt.title("IVF-Flat Recall-Latency Tradeoff")
plt.grid(True)
plt.savefig(RESULTS_DIR / "ivf_recall_latency_curve.png", dpi=200, bbox_inches="tight")
plt.close()

# 4. Thread scaling / QPS plot
THROUGHPUT_CSV = ROOT / "throughput_results.csv"

if THROUGHPUT_CSV.exists():
    throughput_rows = []

    with open(THROUGHPUT_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            throughput_rows.append(row)

    throughput_rows.sort(key=lambda r: int(r["threads"]))

    threads = [int(r["threads"]) for r in throughput_rows]
    qps_values = [float(r["qps"]) for r in throughput_rows]

    plt.figure(figsize=(7, 5))
    plt.plot(threads, qps_values, marker="o")
    plt.xlabel("Threads")
    plt.ylabel("Queries per second")
    plt.title("IVF-PQ Parallel Throughput Scaling")
    plt.grid(True)
    plt.savefig(RESULTS_DIR / "qps_scaling.png", dpi=200, bbox_inches="tight")
    plt.close()

print("Plots written to results/")