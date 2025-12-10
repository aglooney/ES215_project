#!/usr/bin/env bash
set -euo pipefail

# Run curtailment simulations and summaries for all demand scenarios in
# data/demand_data/demand_projection_2025_2028.csv. Requires python deps installed.

SCENARIOS=("inferior" "referencia" "superior")
NET_JSON="models/pandapower_snapshots/brazil_network_zonal_5bus_relaxed.json"
GEN_FORECAST="results/generation_forecast_2025_2028.csv"
GEN_SCRIPT="forecast_generation_ml2_residual.py"

# Adjust these knobs as desired
N_TRIALS=200
LINE_LOADING=100.0
ALLOW_SLACK=0   # set to 1 to allow slack imports
HIST_START=2020
HIST_END=2024
START_YEAR=2025
END_YEAR=2028

run_sim_and_summary() {
  local scenario="$1"
  local demand_out="data/demand_data/demand_projection_${scenario}_2025_2028.csv"
  local sim_out="results/curtailment_simulations_${scenario}.csv"
  local sim_fail="results/curtailment_simulations_${scenario}_opf_failures.csv"
  local summary_dir="results/${scenario}_summary"

  echo ">> Scenario: ${scenario}"
  # Build generation forecast (residual noise script)
  python3 "${GEN_SCRIPT}" --output "${GEN_FORECAST}"

  # Build scenario-specific demand projection
  python3 forecast_demand_from_table.py \
    --historical-path "data/demand_data/demand_projection_clean.csv" \
    --hist-start-year "${HIST_START}" \
    --hist-end-year "${HIST_END}" \
    --start-year "${START_YEAR}" \
    --end-year "${END_YEAR}" \
    --scenario "${scenario}" \
    --output "${demand_out}"

  # Run simulation
  slack_flag=$([ "$ALLOW_SLACK" -eq 1 ] && echo "--allow-slack-imports" || echo "")
  python3 run_forecast_curtailment_sim.py \
    --net-json "${NET_JSON}" \
    --demand "${demand_out}" \
    --gen "${GEN_FORECAST}" \
    --n-trials "${N_TRIALS}" \
    --line-loading-percent "${LINE_LOADING}" \
    --demand-scenario "${scenario}" \
    --out "${sim_out}" \
    --opf-failures-out "${sim_fail}" \
    ${slack_flag}

  # Summarize + plots
  python3 summarize_curtailment_sim.py \
    --input "${sim_out}" \
    --out-dir "${summary_dir}" \
    --fig-dir "${summary_dir}/figures"

  echo "  summary + plots at ${summary_dir}"
}

for s in "${SCENARIOS[@]}"; do
  run_sim_and_summary "$s"
done

echo "Done."
