```
project_name/
│
├── requirements.txt         # Python dependencies
├── data/
│   ├── raw/                 # Original datasets
│   ├── processed/           # Cleaned / preprocessed datasets
│   └── external/            # External sources if needed
│
├── src/
│   ├── __init__.py
│   ├── config.py            # Hyperparameters, paths, constants
│   ├── data_loader.py       # Dataset class, data preprocessing, batching
│   ├── model.py             # Model architecture class(es)
│   ├── train.py             # Training loop, optimizer, scheduler
│   ├── evaluate.py          # Metrics calculation, evaluation logic
│   └── utils.py             # Helper functions (logging, visualization)
│
├── notebooks/
│   └── exploration.ipynb    # EDA, preliminary experiments, visualizations
│
├── experiments/
│   └── experiment_1/        # Saved models, logs, results
│
└── outputs/
    ├── figures/             # Plots, graphs for paper
    └── results/             # CSVs, predictions, metrics

```