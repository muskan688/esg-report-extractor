# Evaluation results

`eval_scores.csv` holds the per-field scores printed by `python -m esg_extractor.cli eval`
on 2026-09-26: Gemini (`gemini-3.6-flash`), the 12-report synthetic corpus, 2% tolerance,
0 failed LLM calls out of 12. `predictions/` holds the 12 per-report extraction JSONs from that run (value, unit as printed,
page, source quote, confidence, and call counters). To re-score without calling any API, copy
them into `data/processed/` and run `python -m esg_extractor.cli eval`; this reproduces
`eval_scores.csv` exactly.

The ground truth (`data/ground_truth/ground_truth.csv`) is produced by
`python scripts/generate_sample_reports.py` with a fixed seed (42), so it is reproducible.

Overall (micro and macro averages are identical): precision 0.99, recall 0.99, F1 0.99;
99 of 100 disclosed values correct, 1 wrong value, 0 hallucinated values (20 undisclosed cells).
