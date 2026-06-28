import subprocess
import time
import csv
import os
import threading
import sys

# ==========================
# CONFIGURATION
# ==========================
# Define models, power caps (in Watts), prompts, and directory paths.

MODELS = [
    "qwen2.5:3b", "phi3-8k:latest", "granite4:3b", "gemma3:4b"
]

POWER_CAPS = [30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130]


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
    "Derive the kernel trick and provide practical examples.",
    "Implement a decision tree using information gain and entropy.",
    "Compare entropy and Gini impurity for classification tasks.",
    "Build a random forest and explain how bagging reduces variance.",
    "Explain boosting and derive the AdaBoost algorithm.",
    "Compare gradient boosting with XGBoost at the algorithmic level.",
    "Derive the Bellman equation in reinforcement learning.",
    "Implement Q-learning from scratch for a grid-world problem.",
    "Compare policy gradient methods with value-based methods.",
    "Explain actor-critic architecture with mathematical formulation.",
    "Implement Deep Q-Network and explain experience replay.",
    "Explain the exploration-exploitation tradeoff and epsilon-greedy strategy.",
    "Compare model-based and model-free reinforcement learning.",
    "Derive the update rules for Proximal Policy Optimization.",
    "Explain temporal difference learning with equations.",
    "Analyze convergence guarantees in reinforcement learning.",
    "Explain convolution operations in CNNs with mathematical formulation.",
    "Implement a convolutional layer from scratch in NumPy.",
    "Compare CNNs and Vision Transformers for image classification.",
    "Derive self-attention from first principles.",
    "Explain multi-head attention and its computational complexity.",
    "Derive the transformer architecture mathematically.",
    "Compare RNNs, LSTMs, and GRUs at the architectural level.",
    "Explain why LSTMs mitigate vanishing gradients.",
    "Implement backpropagation through time.",
    "Analyze the computational complexity of transformer models.",
    "Explain how dropout acts as regularization mathematically.",
    "Compare batch normalization and layer normalization.",
    "Derive the softmax function and its gradient.",
    "Explain cross-entropy loss and its relation to KL divergence.",
    "Implement a custom loss function and compute gradients manually.",
    "Explain how automatic differentiation works internally.",
    "Compare forward-mode and reverse-mode autodiff.",
    "Design a distributed training architecture for large-scale models.",
    "Explain data parallelism vs model parallelism.",
    "Analyze bottlenecks in GPU-accelerated training."
]

SAMPLE_INTERVAL = 0.1
RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

# ==========================
# GPU Selection
# ==========================
# The script requires a target GPU ID as a command line argument (e.g., 0).
# Changing GPU power caps requires root privileges, so this script must be run with sudo.
if len(sys.argv) < 2:
    print("Usage: sudo python3 script.py <GPU_ID>")
    sys.exit(1)

GPU_ID = int(sys.argv[1])

print(f"\nUsing GPU {GPU_ID}")

# ==========================
# Utility Functions
# ==========================

def set_power_cap(power):
    """
    Sets the GPU power limit (cap) in Watts for the specified GPU using nvidia-smi.
    Requires root (sudo) privileges to run.
    """
    subprocess.run(
        ["sudo", "nvidia-smi", "-i", str(GPU_ID), "-pl", str(power)],
        check=True
    )


def enable_persistence():
    """
    Enables GPU Persistence Mode.
    This ensures that the driver remains loaded even when no applications are using
    the GPU, speeding up nvidia-smi commands during power limit transitions.
    """
    subprocess.run(
        ["sudo", "nvidia-smi", "-i", str(GPU_ID), "-pm", "1"],
        check=True
    )


def get_power():
    """
    Queries the current power draw of the GPU in Watts.
    Uses nvidia-smi to query 'power.draw' and returns it as a float.
    """
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
    """
    Parses Ollama's verbose output to find the 'eval count' line, which
    indicates the total number of tokens generated during inference.
    """
    lines = output_text.splitlines()
    for line in lines:
        if line.strip().startswith("eval count:"):
            parts = line.strip().split()
            return int(parts[2])
    return 0


def run_prompt(model, prompt):
    """
    Executes a single prompt for a given model, and measures:
    1. Total Energy (Joules) consumed using a background power-sampling thread.
    2. Total Tokens generated.
    3. Latency (seconds).
    """
    total_energy = 0
    stop_sampling = False

    # Define the background sampler that queries power draw at high frequency (SAMPLE_INTERVAL)
    def power_sampler():
        nonlocal total_energy
        while not stop_sampling:
            power = get_power()
            total_energy += power * SAMPLE_INTERVAL  # Energy = Power * Time
            time.sleep(SAMPLE_INTERVAL)

    # Start the power-sampling thread
    sampler_thread = threading.Thread(target=power_sampler)
    sampler_thread.start()

    # Configure CUDA device visibility for Ollama
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

    start_time = time.time()

    # Run the prompt with verbose output using the Ollama CLI
    result = subprocess.run(
        ["ollama", "run", "--verbose", model, prompt],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env
    )

    latency = time.time() - start_time

    # Stop and clean up the power-sampling thread
    stop_sampling = True
    sampler_thread.join()

    # Extract number of tokens generated from verbose output
    tokens_generated = extract_eval_count(result.stdout)

    return total_energy, tokens_generated, latency


def sanitize_model_name(model):
    """
    Helper function to sanitize model names (e.g. llama3:8b -> llama3_8b)
    to make them safe for file system paths.
    """
    return model.replace(":", "_").replace(".", "_")


def is_model_running(model):
    """
    Checks whether the given model is currently loaded in Ollama by
    parsing the output of 'ollama ps'.
    Returns True if the model appears in the running process list.
    """
    try:
        result = subprocess.run(
            ["ollama", "ps"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        # Each line after the header lists a running model name in the first column
        for line in result.stdout.strip().splitlines()[1:]:
            if line.strip().startswith(model):
                return True
    except Exception:
        pass
    return False


def stop_model(model, max_retries=8):
    """
    Attempts to stop/unload the model from GPU VRAM.
    Retries up to max_retries times, verifying via 'ollama ps' after each attempt.
    Returns True if successfully stopped, False if all retries exhausted.
    """
    for attempt in range(1, max_retries + 1):
        subprocess.run(
            ["ollama", "stop", model],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        time.sleep(2)  # Give Ollama a moment to release resources

        if not is_model_running(model):
            print(f"Model {model} stopped successfully (attempt {attempt}).")
            return True

        print(f"Model {model} still running after attempt {attempt}/{max_retries}, retrying...")

    print(f"ERROR: Failed to stop model {model} after {max_retries} attempts.")
    return False


def run_model(model):
    """
    Benchmarks a single model across all configured GPU power limits:
    1. Performs a model warm-up so it is fully loaded in VRAM.
    2. Iterates over the specified power limits.
    3. Runs the entire benchmark prompt suite under each power limit.
    4. Saves per-prompt data to perprompt_{power}W.csv in the model directory.
    5. Saves aggregated summary to summary.csv in the model directory.
    6. Unloads/stops the model from GPU VRAM to ensure clean isolation.

    Output structure:
        results/<model_name>/
            perprompt_30W.csv    # Per-prompt data for 30W cap
            perprompt_40W.csv    # Per-prompt data for 40W cap
            ...
            summary.csv          # Aggregated summary across all power caps
    """
    print(f"\n=== Running Model: {model} on GPU {GPU_ID} ===")

    # Create a model-specific output directory
    model_dir_name = f"{sanitize_model_name(model)}_gpu{GPU_ID}"
    model_dir = os.path.join(RESULT_DIR, model_dir_name)
    os.makedirs(model_dir, exist_ok=True)

    summary_file = os.path.join(model_dir, "summary.csv")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

    # 1. Warm-up phase: guarantees the model is already in GPU VRAM
    # to avoid caching/first-run latency distorting the results.
    print("Warming up model...")
    subprocess.run(
        ["ollama", "run", model, "hello"],
        stdout=subprocess.PIPE,
        env=env
    )

    # Open the summary CSV file to record aggregated results
    with open(summary_file, "w", newline="") as summary_f:
        summary_writer = csv.writer(summary_f)
        summary_writer.writerow([
            "PowerCap(W)",
            "TotalLatency(s)",
            "TotalEnergy(J)",
            "TotalTokens",
            "EnergyPerToken"
        ])

        # 2. Iterate through each power limit level
        for power in POWER_CAPS:
            print(f"\nRunning power cap: {power}W")
            set_power_cap(power)
            time.sleep(3)  # Wait for the power cap transition to settle

            total_energy = 0
            total_tokens = 0
            total_latency = 0

            # Collect per-prompt results for this power cap
            per_prompt_results = []

            # 3. Run each test prompt under the current power cap
            for i, prompt in enumerate(PROMPTS, start=1):
                energy, tokens, latency = run_prompt(model, prompt)

                total_energy += energy
                total_tokens += tokens
                total_latency += latency

                # Calculate per-prompt energy efficiency
                prompt_energy_per_token = (
                    energy / tokens if tokens > 0 else 0
                )

                per_prompt_results.append({
                    "prompt_index": i,
                    "prompt": prompt,
                    "latency": latency,
                    "energy": energy,
                    "tokens": tokens,
                    "energy_per_token": prompt_energy_per_token,
                })

                print(f"Prompt {i}/{len(PROMPTS)} done | Tokens: {tokens} | Energy: {energy:.2f}J")

            # 4. Write per-prompt CSV for this power cap
            perprompt_file = os.path.join(model_dir, f"perprompt_{power}W.csv")
            with open(perprompt_file, "w", newline="") as pp_f:
                pp_writer = csv.writer(pp_f)
                pp_writer.writerow([
                    "PromptIndex",
                    "Latency(s)",
                    "Energy(J)",
                    "Tokens",
                    "EnergyPerToken"
                ])
                for row in per_prompt_results:
                    pp_writer.writerow([
                        row["prompt_index"],
                        row["latency"],
                        row["energy"],
                        row["tokens"],
                        row["energy_per_token"],
                    ])

            print(f"  Per-prompt data saved to {perprompt_file}")

            # Calculate average Energy per Token (J/Token) for summary
            energy_per_token = (
                total_energy / total_tokens if total_tokens > 0 else 0
            )

            # 5. Log aggregated power cap metrics to summary CSV
            summary_writer.writerow([
                power,
                total_latency,
                total_energy,
                total_tokens,
                energy_per_token
            ])

            print(f"Total Energy: {total_energy:.2f}J")
            print(f"Total Tokens: {total_tokens}")
            print(f"Energy/Token: {energy_per_token:.4f}")

    print(f"\nFinished model {model}. Results saved to {model_dir}/")

    # Write a reference file listing all prompts used in the experiment
    prompts_file = os.path.join(model_dir, "prompts.csv")
    with open(prompts_file, "w", newline="") as pf:
        prompts_writer = csv.writer(pf)
        prompts_writer.writerow(["PromptIndex", "Prompt"])
        for i, prompt in enumerate(PROMPTS, start=1):
            prompts_writer.writerow([i, prompt])
    print(f"Prompts list saved to {prompts_file}")

    # 6. Cleanup phase: stop/unload the model from GPU VRAM to ensure it is not kept
    # loaded in memory during subsequent model benchmarks.
    print(f"Stopping model {model} to free VRAM...")
    if not stop_model(model):
        print(f"ABORTING: Could not stop model {model}. All data collected so far has been saved to {model_dir}/")
        sys.exit(1)


def main():
    """
    Main script execution flow.
    Enables Persistence Mode on the selected GPU, then runs benchmarks
    for all configured models.
    """
    enable_persistence()

    for model in MODELS:
        run_model(model)

    print("\nAll models completed.")


if __name__ == "__main__":
    main()