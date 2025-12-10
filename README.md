# Brazil Zonal Curtailment Simulation

This repo builds demand/generation forecasts, runs a zonal DC-OPF with ENS/dump handling, and summarizes curtailment across multiple scenarios.

## Data prep (run once)
1) **CSV preprocessing** (`csv_preprocessing/`): clean/align ONS generation, weather, and demand inputs. Key outputs:
   - `data/merged_generation_weather_v2.csv` (plant gen + weather, daily)
   - `data/demand_data/demand_projection_clean.csv` (state/subsystem monthly MWh)
2) **Network build** (already committed): relaxed 5-bus net at `models/pandapower_snapshots/brazil_network_zonal_5bus_relaxed.json`.

## Core workflow
```bash
# Demand forecast (scenarios: inferior/referencia/superior)
python forecast_demand_from_table.py --scenario referencia --output data/demand_data/demand_projection_2025_2028.csv

# Generation forecast (auto-eval both models; uses HGB by default)
python forecast_generation_ml2.py --val-leakage --output results/generation_forecast_2025_2028.csv

# Curtailment sim (monthly, zonal DC-OPF)
python run_forecast_curtailment_sim.py \
  --net-json models/pandapower_snapshots/brazil_network_zonal_5bus_relaxed.json \
  --demand data/demand_data/demand_projection_2025_2028.csv \
  --gen results/generation_forecast_2025_2028.csv \
  --n-trials 200 \
  --line-loading-percent 100 \
  --out results/curtailment_simulations.csv

# Summaries + plots
python summarize_curtailment_sim.py --input results/curtailment_simulations.csv
```

## Optional helpers
- **Batch scenarios**: `bash run_scenario_batch.sh` (runs inferior/referencia/superior end-to-end).
- **Sensitivity sweep**: `python run_sensitivity.py` (line loading × slack × scenarios) then `python summarize_sensitivity.py`.
- **Quarterly plot**: after a historic baseline + future runs, `python plot_quarterly_curtailment.py --historic <historic_monthly_detailed.csv>`.

## Key assumptions
- Demand forecasts are average MW per month derived from growth targets + seasonal factors (2020–2024 history).
- Generation forecasts are monthly avg MW per subsystem via AR+seasonal models (ridge/HGB) with weather features and optional noise; `--val-leakage` uses actual lags for validation.
- OPF is monthly DC, zonal 5-bus; ENS/dump modeled as high-cost injections; slack optional. No time coupling/storage.
- Historical gen with daily timestamps is treated as MWh/day and divided by 24 to get MW.

## Repro tips
- Use `python3` in your venv with required deps (pandas, numpy, sklearn, pandapower, matplotlib).
- Seeds are exposed (`--seed`); set forecast noise stds to 0 for deterministic runs.
- Results are gitignored (`results/`).
