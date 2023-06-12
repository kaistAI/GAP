# GAP
[ACL 2023] Gradient Ascent Post-training Enhances Language Model Generalization

Dongkeun Yoon*, Joel Jang*, Sungdong Kim, Minjoon Seo (* Equal Contribution)

Updating language models towards divergence can lead to better performance.

### 1. Run GAP
The below example command will run 100 GAP runs with Github (Pile) on OPT-350M. You can modify the script or the config to change the training data, model etc. 
```bash
bash scripts/run_git.sh
```

### 2. Test some of our top performing GAP runs
The below runs resulted in substantial performance gains in dialogue generation tasks. Note that performance gains in dialogue tasks doesn't necessarily correlate with performance gains in classification tasks, and the following models may result in performance drops in classification tasks.

- OPT-350M
    ```bash
    python run.py --config configs/final/extraction-350.json --index 79 --num_train_epochs 8 --check_val_every_n_epoch 8
    ```
- OPT-1.3B
    ```bash
    python run.py --config configs/final/github-1.3.json --index 56 --num_train_epochs 7 --check_val_every_n_epoch 7
    ```
- OPT-2.7B
    ```bash
    python run.py --config configs/final/cc-2.7.json --index 39 --num_train_epochs 7 --check_val_every_n_epoch 7
    ```

### 3. Conduct your own analysis
We provide the full evaluation results in `full_results/`. These results can be aligned with the training dataset files in `data/train_data/` using the `corpora` and `index` column. For example if `corpora == Git.` and `index == 0`, the evaluation result comes from a model trained on the 0th entry of `data/train_data/github.csv`.
