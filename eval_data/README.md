# eval_data

Drop Databento `.dbn.zst` files here to score the saved model on data it has never seen.

The archive this project was built on ends **2025-10-08**. Any file dated after that is a
genuine out-of-sample test — the checkpoint in `models/` cannot have been fitted to it.

```bash
python src/evaluate.py
```

Each day produces `reports/eod_YYYYMMDD.json` and `.txt` with balance, ROI and accuracy,
and updates the running account in `reports/portfolio_state.json`.

To backtest on archive days instead (the 12 most recent were held out of training):

```bash
python src/evaluate.py --from-zip last:12
```

Files here are gitignored — market data does not belong in the repo.
