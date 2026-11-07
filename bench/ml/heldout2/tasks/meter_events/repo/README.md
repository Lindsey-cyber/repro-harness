# meterlab

Predicts grid-meter anomaly events from quantised site telemetry.

## Run

    python -m meterlab.cli run exp.json

Summary is written to `runs/<name>/summary.json` (`name` comes from `exp.json`).

## Notes on the data

`data/readings.csv` is exported by the logger, which quantises temperature (3 degC steps),
load (0.5 kW steps) and hour (4 h blocks). Identical feature rows therefore recur many
times across the file, in both splits; this is expected and rows are deliberately not
de-duplicated. `split` is the logger's own train/test assignment.

## Result

Test F1 (positive class = event): **0.726**.
