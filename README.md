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

## How It Works & Methodology

### 1. Determining Power Limits
Before running the benchmark, the minimum and maximum power limits of the GPU were identified using `nvidia-smi` (typically by running `nvidia-smi -q -d POWER` to check the supported power limits). Based on these bounds, the `POWER_CAPS` array in the script was populated to test the GPU at **10W intervals** from the minimum to the maximum supported power cap.

### 2. Model VRAM Lifecycle Management
To ensure clean and isolated benchmarks, **only a single model should be tested at a time**. Ollama keeps loaded models in GPU VRAM after inference completes. After testing each model, the script automatically unloads it from VRAM using `ollama stop`. The script verifies the model has actually stopped by checking `ollama ps`, retrying up to **8 times** before aborting. If the model cannot be stopped, the script exits gracefully, preserving all data collected so far.

### 3. Execution Pipeline
The script executes the benchmark using the following steps:

1. **Persistence Mode**: Enables persistence mode (`nvidia-smi -pm 1`) on the selected GPU to speed up power management state changes.
2. **Model Warm-up**: Runs a single quick prompt (`"hello"`) to load the target model into the GPU memory before measurement starts.
3. **Power Cap Iteration**: Iterates through the specified list of power limits. For each power limit:
   - Sets the GPU power cap using `sudo nvidia-smi -i <GPU_ID> -pl <Power_Cap>`.
   - Pauses for 3 seconds to allow the power state to settle.
   - Runs a series of pre-configured benchmarking prompts sequentially.
4. **Real-time Power Sampling**: During each prompt execution, a background thread queries the GPU's power draw (`power.draw`) every `0.1` seconds.
5. **Metric Calculation & Logging**:
   - Computes total energy consumed (integral of power draw over prompt duration).
   - Extracts the exact number of tokens generated from Ollama's verbose evaluation statistics (`eval count`).
   - Calculates the average **Energy per Token** (Total Joules / Total Tokens).
6. **Data Output**: Stores per-prompt results and aggregated summary into CSV files within a model-specific directory.
7. **Model Cleanup**: Verifies and stops the model from GPU VRAM with retry logic (up to 8 attempts) before proceeding to the next model.

---

## Configuration

You can customize the benchmark directly in the `Experiment.py` script by modifying the variables in the **CONFIG** section:

```python
# Models to benchmark (Make sure these are pre-pulled in Ollama)
MODELS = [
    "qwen2.5:3b", "phi3-8k:latest", "granite4:3b", "gemma3:4b"
]

# Power caps to test (in Watts)
POWER_CAPS = [30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130]

# Prompts used for benchmarking (50 prompts covering ML, systems, crypto, algorithms)
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

## Output Structure

Results are saved inside a **model-specific subdirectory** under `results/`:

```
results/
└── <model_name>_gpu<GPU_ID>/
    ├── perprompt_30W.csv        # Per-prompt data at 30W power cap
    ├── perprompt_40W.csv        # Per-prompt data at 40W power cap
    ├── ...
    ├── perprompt_<max>W.csv     # Per-prompt data at max power cap
    ├── summary.csv              # Aggregated summary across all power caps
    └── prompts.csv              # Reference list of all prompts used
```

### Per-Prompt CSV (`perprompt_{power}W.csv`)

Contains one row per prompt for a given power cap:

| Column | Description |
| :--- | :--- |
| **PromptIndex** | The 1-based index of the prompt. |
| **Latency(s)** | Time taken to complete this prompt (seconds). |
| **Energy(J)** | Energy consumed by the GPU during this prompt (Joules). |
| **Tokens** | Number of tokens generated for this prompt. |
| **EnergyPerToken** | Energy efficiency for this prompt (Joules/token). |

### Summary CSV (`summary.csv`)

Contains one row per power cap, aggregated across all prompts:

| Column | Description |
| :--- | :--- |
| **PowerCap(W)** | The GPU power limit set for the run (Watts). |
| **TotalLatency(s)** | The combined time taken to complete all test prompts (seconds). |
| **TotalEnergy(J)** | The total energy consumed by the GPU during the run (Joules). |
| **TotalTokens** | The sum of all tokens generated across all test prompts. |
| **EnergyPerToken** | Average energy required to generate a single token (Joules/token). |

### Prompts CSV (`prompts.csv`)

A reference file listing all prompts used in the experiment:

| Column | Description |
| :--- | :--- |
| **PromptIndex** | The 1-based index of the prompt. |
| **Prompt** | The full prompt text. |

---

## Repository Directory Structure

Collected experiment data is organized by GPU and batch:

```
├── Experiment.py                    # Benchmarking script
├── README.md
├── rtx 4000/                        # NVIDIA RTX 4000 results
│   ├── batch_1/                     # First batch of per-prompt data
│   │   ├── gemma3_4b_gpu0/
│   │   ├── granite4_3b_gpu0/
│   │   ├── phi3-8k_latest_gpu0/
│   │   └── qwen2_5_3b_gpu0/
│   ├── batch_2/                     # Second batch of per-prompt data
│   │   └── ...
│   └── *.csv                        # Legacy summary-only CSVs
├── rtx 5000/                        # NVIDIA RTX 5000 results
│   ├── batch_1/
│   ├── batch_2/
│   └── *.csv
└── rtx 6000/                        # NVIDIA RTX 6000 results
    ├── batch_1/
    ├── batch_2/
    └── *.csv
```

Each `batch_*/` directory contains model subdirectories (e.g., `qwen2_5_3b_gpu0/`) with the full per-prompt CSV files, summary CSV, and prompts reference file.
