# readmit-transfer

Does a 30-day readmission model trained at hospital A transfer to hospital B?

* `data/site_a.csv` - hospital A, used for training (75%) and in-site validation (25%, random patients).
* `data/site_b.csv` - hospital B, used only for evaluation. **Site B is a different hospital with a different lab
  analyser (shifted and rescaled lab_1, lab_2, lab_3) and we expect, and report, a drop relative to site A.**
  No site B data is used for fitting or for standardisation.

## Run

    python analysis_nb.py        # reads config.json, writes out/nb_run/metrics.json

## Result

In-site validation (site A): acc **TBD**, auc **TBD**.
Site B (shifted): acc **TBD**, auc **TBD**.
