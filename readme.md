## 📁 Repository Structure

* **`experiment.py`**: Python wrapper class that interfaces with the MOA `.jar` to run the data stream evaluations.
* **`run_experiments.py`**: Multiprocessing orchestrator that executes multiple MOA experiments in parallel across different configurations (e.g., Lags, Models).
* **`forecast_horizon_analysis.ipynb`**: Evaluates the global model performance (Accuracy, Kappa, Precision, Recall) across multiple window sizes (lags).
* **`asset_performance_distribution.ipynb`**: Generates descriptive statistics and boxplots showing the distribution of model performance across all valid assets.
* **`high_liquidity_asset_analysis.ipynb`**: Performs a stress test on a top-10 diversified portfolio of high-liquidity assets, calculating micro-averaged metrics.
* **`results/`**: Directory where the raw CSV outputs from MOA are saved.
* **`figures/`**: Directory where the generated IEEE-standard PDFs are exported.

---

## 🚀 Usage Guide

### Step 1: Run the MOA Experiments
To generate the raw performance data, use the orchestration script. It utilizes all available CPU cores (minus one) to run the experiments in parallel.

1. Ensure your MOA `.jar` file and datasets are properly referenced in `experiment.py`.
2. Execute the runner:

```bash
python run_experiments.py