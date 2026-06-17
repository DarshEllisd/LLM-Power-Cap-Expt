# LLM Power Cap Experiment

This repository contains a benchmarking script to measure the energy efficiency, latency, and throughput of running Large Language Models (LLMs) via **Ollama** under various GPU power limits (power caps).

The script automates the process of setting GPU power limits, running a list of benchmark prompts, sampling power consumption in real-time, and calculating the energy efficiency metric: **Energy per Token (Joules/token)**.

---

## Prerequisites

To run this script, your system must meet the following requirements:

1. **System & OS**: Linux-based system with root access (required to modify GPU power caps via `nvidia-smi`).
2. **NVIDIA GPU & Drivers**: NVIDIA GPU with proprietary drivers installed.
3. **NVIDIA System Management Interface (`nvidia-smi`)**: Must be available in the system path.
4. **Ollama**: Ollama must be installed, running, and the models you plan to test must be pre-pulled (e.g., `ollama pull llama3:8b`).
5. **Python 3**: Python 3.x with no external dependencies required (uses built-in standard libraries: `subprocess`, `time`, `csv`, `os`, `threading`, `sys`).

---

## How It Works

The script executes the benchmark using the following steps:

1. **Persistence Mode**: Enables persistence mode (`nvidia-smi -pm 1`) on the selected GPU to speed up power management state changes.
2. **Model Warm-up**: Runs a single quick prompt (`"hello"`) to load the target model into the GPU memory before measurement starts.
3. **Power Cap Iteration**: Iterates through a specified list of power limits (in Watts). For each power limit:
   - Sets the GPU power cap using `sudo nvidia-smi -i <GPU_ID> -pl <Power_Cap>`.
   - Pauses for 3 seconds to allow the power state to settle.
   - Runs a series of pre-configured benchmarking prompts sequentially.
4. **Real-time Power Sampling**: During each prompt execution, a background thread queries the GPU's power draw (`power.draw`) every `0.1` seconds.
5. **Metric Calculation & Logging**:
   - Computes total energy consumed (integral of power draw over prompt duration).
   - Extracts the exact number of tokens generated from Ollama's verbose evaluation statistics (`eval count`).
   - Calculates the average **Energy per Token** (Total Joules / Total Tokens).
6. **Data Output**: Stores the aggregated results for each power cap into a CSV file.

---

## Configuration

You can customize the benchmark directly in the `Experiment.py` script by modifying the variables in the **CONFIG** section:

```python
# Models to benchmark (Make sure these are pre-pulled in Ollama)
MODELS = [
    "llama3:8b",
]

# Power caps to test (in Watts)
POWER_CAPS = [100, 110, 120, 130, 140, 150, 160, 170,
              180, 190, 200, 210, 220, 230, 240, 250]

# Prompts used for benchmarking
PROMPTS = [
    "Explain gradient descent from first principles and derive its update rule mathematically.",
    ...
]

# Sampling interval for power draw (in seconds)
SAMPLE_INTERVAL = 0.1

# Directory where CSV result files will be saved
RESULT_DIR = "results"
```

---

## Usage

Since setting GPU power limits requires root privileges, the script must be run with `sudo`. You also need to pass the target **GPU ID** as a command-line argument.

```bash
sudo python3 Experiment.py <GPU_ID>
```

### Example
To run the benchmark on GPU `0`:
```bash
sudo python3 Experiment.py 0
```

---

## Output CSV Structure

Results are saved inside the `results/` directory with the naming convention `{model_name}_gpu{GPU_ID}.csv` (e.g., `results/llama3_8b_gpu0.csv`).

The CSV contains the following columns for each tested power limit:

| Column | Description |
| :--- | :--- |
| **PowerCap(W)** | The GPU power limit set for the run (Watts). |
| **TotalLatency(s)** | The combined time taken to complete all test prompts (seconds). |
| **TotalEnergy(J)** | The total energy consumed by the GPU during the run (Joules). |
| **TotalTokens** | The sum of all tokens generated across all test prompts. |
| **EnergyPerToken** | Average energy required to generate a single token (Joules/token). |
