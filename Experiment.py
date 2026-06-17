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
    "llama3:8b"
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
    "Analyze bottlenecks in GPU-accelerated training.",
    "Implement a memory allocator and explain fragmentation issues.",
    "Explain how a compiler converts high-level code into machine code.",
    "Compare interpreted and compiled languages at runtime level.",
    "Implement a simple virtual machine.",
    "Explain garbage collection algorithms like mark-sweep and generational GC.",
    "Analyze deadlocks and implement a deadlock detection algorithm.",
    "Explain scheduling algorithms in operating systems.",
    "Implement a thread-safe queue.",
    "Compare mutexes, semaphores, and monitors.",
    "Explain how context switching works internally.",
    "Implement a TCP-like reliable protocol over UDP.",
    "Explain how consensus algorithms like Raft work.",
    "Compare Paxos and Raft.",
    "Design a distributed key-value store.",
    "Explain CAP theorem with practical examples.",
    "Implement consistent hashing.",
    "Explain how load balancing algorithms work.",
    "Compare SQL and NoSQL from a systems perspective.",
    "Design a scalable microservices architecture.",
    "Explain eventual consistency in distributed systems.",
    "Implement RSA encryption from scratch.",
    "Explain elliptic curve cryptography mathematically.",
    "Compare symmetric and asymmetric cryptography.",
    "Explain how hashing functions like SHA-256 work.",
    "Implement a Merkle tree.",
    "Analyze blockchain consensus mechanisms.",
    "Explain zero-knowledge proofs at a conceptual level.",
    "Compare proof-of-work and proof-of-stake.",
    "Implement a simple blockchain prototype.",
    "Analyze vulnerabilities in smart contracts.",
    "Implement depth-first search and breadth-first search.",
    "Compare Dijkstra and A* algorithms.",
    "Implement a Trie and analyze its complexity.",
    "Derive the time complexity of dynamic programming solutions.",
    "Explain memoization vs tabulation.",
    "Implement the KMP string matching algorithm.",
    "Analyze suffix arrays and suffix trees.",
    "Implement a red-black tree.",
    "Compare B-trees and AVL trees.",
    "Analyze amortized complexity with examples.",
    "Implement gradient clipping and explain why it stabilizes training.",
    "Compare Adam, RMSProp, and SGD with momentum.",
    "Derive the Adam optimizer update rule mathematically.",
    "Explain learning rate scheduling strategies.",
    "Implement a custom optimizer from scratch.",
    "Analyze overfitting and implement early stopping.",
    "Compare cross-validation techniques.",
    "Design an experiment to evaluate a neural network rigorously.",
    "Explain scaling laws in large language models.",
    "Propose a novel improvement to transformer efficiency and justify it technically."
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


def run_model(model):
    """
    Benchmarks a single model across all configured GPU power limits:
    1. Performs a model warm-up so it is fully loaded in VRAM.
    2. Iterates over the specified power limits.
    3. Runs the entire benchmark prompt suite under each power limit.
    4. Computes latency, energy consumption, and energy efficiency (Joules/token).
    5. Saves all statistics to a CSV file in the results directory.
    """
    print(f"\n=== Running Model: {model} on GPU {GPU_ID} ===")

    csv_name = f"{sanitize_model_name(model)}_gpu{GPU_ID}.csv"
    result_file = os.path.join(RESULT_DIR, csv_name)

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

    # Open target CSV file to record results
    with open(result_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
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

            # 3. Run each test prompt under the current power cap
            for prompt in PROMPTS:
                energy, tokens, latency = run_prompt(model, prompt)

                total_energy += energy
                total_tokens += tokens
                total_latency += latency

                print(f"Prompt done | Tokens: {tokens} | Energy: {energy:.2f}J")

            # Calculate average Energy per Token (J/Token)
            energy_per_token = (
                total_energy / total_tokens if total_tokens > 0 else 0
            )

            # Log current power cap metrics
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