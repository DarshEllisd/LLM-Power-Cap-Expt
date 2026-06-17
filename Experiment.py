import subprocess
import time
import csv
import os
import threading
import sys

# ==========================
# CONFIG
# ==========================

MODELS = [
    "llama3:8b",
]

POWER_CAPS = [100, 110, 120, 130, 140, 150, 160, 170,
              180, 190, 200, 210, 220, 230, 240, 250]

PROMPTS = [
    "Explain gradient descent from first principles and derive its update rule mathematically.",
    "Compare batch, stochastic, and mini-batch gradient descent with convergence analysis.",
    "Implement gradient descent in Python and visualize its convergence on a convex function.",
    "Derive the backpropagation algorithm step-by-step for a multilayer perceptron.",
    "Explain vanishing and exploding gradients and propose mitigation strategies.",
    "Implement quicksort and analyze its average and worst-case time complexity.",
    "Compare quicksort, mergesort, and heapsort in terms of space and stability.",
    "Prove why quicksort has expected O(n log n) complexity.",
    "Implement a neural network from scratch without using deep learning libraries.",
    "Explain the bias-variance tradeoff with mathematical intuition.",
    "Derive the closed-form solution for linear regression using normal equations.",
    "Implement logistic regression using gradient descent from scratch.",
    "Compare L1 and L2 regularization and show their geometric interpretations.",
    "Explain the mathematical intuition behind support vector machines.",
    "Explain scaling laws in large language models.",
    "Propose a novel improvement to transformer efficiency and justify it technically."
]

SAMPLE_INTERVAL = 0.1
RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

# ==========================
# GPU Selection (argument)
# ==========================

if len(sys.argv) < 2:
    print("Usage: sudo python3 script.py <GPU_ID>")
    sys.exit(1)

GPU_ID = int(sys.argv[1])

print(f"\nUsing GPU {GPU_ID}")

# ==========================
# Utility Functions
# ==========================

def set_power_cap(power):
    subprocess.run(
        ["sudo", "nvidia-smi", "-i", str(GPU_ID), "-pl", str(power)],
        check=True
    )


def enable_persistence():
    subprocess.run(
        ["sudo", "nvidia-smi", "-i", str(GPU_ID), "-pm", "1"],
        check=True
    )


def get_power():
    result = subprocess.run(
        [
            "nvidia-smi",
            "-i", str(GPU_ID),
            "--query-gpu=power.draw",
            "--format=csv,noheader,nounits",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    return float(result.stdout.strip())


def extract_eval_count(output_text):
    lines = output_text.splitlines()
    for line in lines:
        if line.strip().startswith("eval count:"):
            parts = line.strip().split()
            return int(parts[2])
    return 0


def run_prompt(model, prompt):
    total_energy = 0
    stop_sampling = False

    def power_sampler():
        nonlocal total_energy
        while not stop_sampling:
            power = get_power()
            total_energy += power * SAMPLE_INTERVAL
            time.sleep(SAMPLE_INTERVAL)

    sampler_thread = threading.Thread(target=power_sampler)
    sampler_thread.start()

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

    start_time = time.time()

    result = subprocess.run(
        ["ollama", "run", "--verbose", model, prompt],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env
    )

    latency = time.time() - start_time

    stop_sampling = True
    sampler_thread.join()

    tokens_generated = extract_eval_count(result.stdout)

    return total_energy, tokens_generated, latency


def sanitize_model_name(model):
    return model.replace(":", "_").replace(".", "_")


def run_model(model):
    print(f"\n=== Running Model: {model} on GPU {GPU_ID} ===")

    csv_name = f"{sanitize_model_name(model)}_gpu{GPU_ID}.csv"
    result_file = os.path.join(RESULT_DIR, csv_name)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

    print("Warming up model...")
    subprocess.run(
        ["ollama", "run", model, "hello"],
        stdout=subprocess.PIPE,
        env=env
    )

    with open(result_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "PowerCap(W)",
            "TotalLatency(s)",
            "TotalEnergy(J)",
            "TotalTokens",
            "EnergyPerToken"
        ])

        for power in POWER_CAPS:
            print(f"\nRunning power cap: {power}W")
            set_power_cap(power)
            time.sleep(3)

            total_energy = 0
            total_tokens = 0
            total_latency = 0

            for prompt in PROMPTS:
                energy, tokens, latency = run_prompt(model, prompt)

                total_energy += energy
                total_tokens += tokens
                total_latency += latency

                print(f"Prompt done | Tokens: {tokens} | Energy: {energy:.2f}J")

            energy_per_token = (
                total_energy / total_tokens if total_tokens > 0 else 0
            )

            writer.writerow([
                power,
                total_latency,
                total_energy,
                total_tokens,
                energy_per_token
            ])

            print(f"Total Energy: {total_energy:.2f}J")
            print(f"Total Tokens: {total_tokens}")
            print(f"Energy/Token: {energy_per_token:.4f}")

    print(f"\nFinished model {model}. Results saved to {result_file}")


def main():
    enable_persistence()

    for model in MODELS:
        run_model(model)

    print("\nAll models completed.")


if __name__ == "__main__":
    main()