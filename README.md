# Embedding Compression Benchmark

A benchmarking tool for analyzing the impact of embedding compression techniques (quantization and PCA) on performance and efficiency.

## Features

- Multiple quantization types support:
  - FLOAT32 (baseline)
  - FLOAT16
  - BFLOAT16
  - FLOAT8 (E4M3 and E5M2)
  - FLOAT4 (E2M1)
  - INT8
  - Binary
- PCA dimensionality reduction with configurable components
- Caching system using Qdrant vector database
- Comprehensive evaluation using [MTEB](https://github.com/embeddings-benchmark/mteb) benchmarks
- Automated experiment configuration through YAML files

## Requirements

- Python 3.12
- Docker and Docker Compose
- CUDA-capable GPU (recommended)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/embedding-compression-bench.git
cd embedding-compression-bench
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start Qdrant vector database:
```bash
docker-compose up -d
```

## Usage

1. Configure your experiments in a YAML file (see `configs/experiment.yml` for an example):
   - Select model
   - Choose benchmarks
   - Define compression experiments

2. Run experiments:
```bash
HF_DATASETS_TRUST_REMOTE_CODE=1 python run_experiments.py --config configs/experiment.yml
```

3. Visualize results:
   - Open and run `plot/plot_results.ipynb` in Jupyter Notebook
   - Results will be plotted showing relative performance differences against the FLOAT32 baseline

## Configuration

This tool uses YAML files for experiment configuration. Two example configurations are provided:
- `configs/experiment.yml`: Basic set of benchmarks and compression techniques
- `configs/experiment_complete.yml`: Comprehensive evaluation with more benchmarks

Example configuration structure:
```yaml
model_name: "BAAI/bge-small-en-v1.5"

tasks:
  - AppsRetrieval
  - ARCChallenge
  # Add more tasks...

experiments:
  - name: float32
    quantization_type: FLOAT32
    pca_config: null

  - name: float16_50PCA
    quantization_type: FLOAT16
    pca_config: 0.5
  # Add more experiments...
```

See `core/configs/pca_config.py` and `core/configs/quantization_type.py` to see the avaiable options.

## Results

Results are stored in the `results/` directory, organized by model name. Each experiment generates a JSON file containing NDCG@10 scores for **each** benchmark. More metrics soon to be added.


