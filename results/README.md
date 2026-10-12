# Evaluation results

`eval_scores.csv` holds the per-field scores printed by `python -m esg_extractor.cli eval`
on 2026-09-26: Gemini (`gemini-3.6-flash`), the 12-report synthetic corpus, 2% tolerance,
0 failed LLM calls out of 12. The scores were copied from that command's console output.
The per-report extraction JSONs from that run were not saved, so the scores can only be
re-derived by re-running the pipeline.

The ground truth (`data/ground_truth/ground_truth.csv`) is produced by
`python scripts/generate_sample_reports.py` with a fixed seed (42), so it is reproducible.

Overall (micro and macro averages are identical): precision 0.99, recall 0.99, F1 0.99;
99 of 100 disclosed values correct, 1 wrong value, 0 hallucinated values (20 undisclosed cells).
