"""Monte Carlo curtailment simulation using forecasted demand and generation."""
import argparse
import itertools
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd
import pandapower as pp

NETWORK_PATH = "models/brazil_5bus_network.json"
SUBSYSTEMS = ["NORTE", "NORDESTE", "SUDESTE", "SUL", "PARAGUAI"]

parser = argparse.ArgumentParser(description="Run stochastic curtailment simulations through 2028.")
parser.add_argument("--demand-forecast", default="data/demand_data/demand_projection_2025_2028.csv")
parser.add_argument("--generation-forecast", default="results/generation_forecast_2024_2028.csv")
parser.add_argument("--scenario", default="referencia", choices=["inferior", "referencia", "superior"])
parser.add_argument("--start-year", type=int, default=2025)
parser.add_argument("--end-year", type=int, default=2028)
parser.add_argument("--trials", type=int, default=200)
parser.add_argument("--demand-noise-std", type=float, default=0.0)
parser.add_argument("--gen-noise-std", type=float, default=0.05)
parser.add_argument("--demand-history", default="data/demand_data/demand_projection_clean.csv")
parser.add_argument("--generation-history", default="data/merged_generation_weather_v2.csv")
parser.add_argument("--hist-start-year", type=int, default=2020)
parser.add_argument("--hist-end-year", type=int, default=2024)
parser.add_argument("--use-correlated-noise", action="store_true", help="Use historical covariance for noise sampling.")
parser.add_argument("--demand-noise-scale", type=float, default=1.0, help="Scale factor applied to correlated demand deviations.")
parser.add_argument("--gen-noise-scale", type=float, default=1.0, help="Scale factor applied to correlated generation deviations.")
parser.add_argument("--allow-slack-imports", action="store_true")
parser.add_argument("--output", default="results/curtailment_simulations.csv")
parser.add_argument("--seed", type=int, default=123)
parser.add_argument(
    "--storage-source",
    default=None,
    help="Subsystem where pseudo-storage absorbs surplus (generation reduced).",
)
parser.add_argument(
    "--storage-target",
    default=None,
    help="Subsystem where pseudo-storage re-injects energy (generation increased).",
)
parser.add_argument(
    "--storage-transfer-mw",
    type=float,
    default=0.0,
    help="MW shifted from source to target each month to emulate storage.",
)
args = parser.parse_args()

rng = np.random.default_rng(args.seed)
STORAGE_SOURCE = args.storage_source.upper().strip() if args.storage_source else None
STORAGE_TARGET = args.storage_target.upper().strip() if args.storage_target else None

def load_network():
    net = pp.from_json(NETWORK_PATH)
    for table in (net.load, net.gen, net.sgen, net.ext_grid, net.poly_cost):
        if not table.empty:
            table.drop(table.index, inplace=True)
    bus_lookup = {}
    for name in SUBSYSTEMS:
        match = net.bus[net.bus["name"] == f"BUS_{name}"]
        if match.empty:
            raise ValueError(f"Missing bus BUS_{name}")
        bus_lookup[name] = match.index[0]
    slack_idx = pp.create_ext_grid(net, bus=bus_lookup["SUDESTE"], vm_pu=1.0, name="SLACK")
    return net, bus_lookup, slack_idx


def ensure_scaling(df):
    if "scaling" not in df.columns:
        df["scaling"] = 1.0


def month_hours(year: int, month: int) -> int:
    import calendar
    return calendar.monthrange(int(year), int(month))[1] * 24


def aggregate_demand_history(path, start_year, end_year):
    df = pd.read_csv(path)
    df["subsystem"] = df["subsystem"].str.upper().str.replace("SUDESTE/CENTRO-OESTE", "SUDESTE")
    df = df[(df["year"] >= start_year) & (df["year"] <= end_year)]
    if df.empty:
        raise ValueError("Historical demand slice empty for covariance computation.")
    agg = df.groupby(["subsystem", "year", "month"])["MWh"].sum().reset_index()
    agg["hours"] = agg.apply(lambda r: month_hours(r["year"], r["month"]), axis=1)
    agg["avg_mw"] = agg["MWh"] / agg["hours"]
    agg["date"] = pd.to_datetime(
        agg["year"].astype(int).astype(str) + "-" + agg["month"].astype(int).astype(str) + "-01"
    )
    pivot = agg.pivot_table(index="date", columns="subsystem", values="avg_mw").sort_index()
    return pivot


def aggregate_generation_history(path, start_year, end_year):
    raw = pd.read_csv(path, parse_dates=["date"])
    raw["subsystem"] = raw["subsys_name"].astype(str).str.upper().str.replace("SUDESTE/CENTRO-OESTE", "SUDESTE")
    raw["year"] = raw["date"].dt.year
    raw["month"] = raw["date"].dt.month
    df = raw[(raw["year"] >= start_year) & (raw["year"] <= end_year)]
    if df.empty:
        raise ValueError("Historical generation slice empty for covariance computation.")
    agg = df.groupby(["subsystem", "year", "month"])["gen_val(MW)"].sum().reset_index()
    agg["hours"] = agg.apply(lambda r: month_hours(r["year"], r["month"]), axis=1)
    agg["avg_mw"] = agg["gen_val(MW)"] / agg["hours"]
    agg["date"] = pd.to_datetime(
        agg["year"].astype(int).astype(str) + "-" + agg["month"].astype(int).astype(str) + "-01"
    )
    pivot = agg.pivot_table(index="date", columns="subsystem", values="avg_mw").sort_index()
    return pivot


def compute_covariance_matrix(series_df):
    returns = np.log(series_df / series_df.shift(1))
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if returns.empty:
        return None
    cov = returns.cov()
    return cov


demand_cov = None
generation_cov = None
if args.use_correlated_noise:
    demand_series = aggregate_demand_history(
        args.demand_history, args.hist_start_year, args.hist_end_year
    )
    generation_series = aggregate_generation_history(
        args.generation_history, args.hist_start_year, args.hist_end_year
    )
    demand_cov = compute_covariance_matrix(demand_series)
    generation_cov = compute_covariance_matrix(generation_series)


def run_single_month(subsys_demand, subsys_gen, allow_slack):
    net, bus_lookup, slack_idx = load_network()

    gen_indices = {}
    for subsys, value in subsys_gen.items():
        if subsys not in bus_lookup:
            continue
        idx = pp.create_gen(
            net,
            bus=bus_lookup[subsys],
            p_mw=value,
            vm_pu=1.0,
            min_p_mw=0.0,
            max_p_mw=value,
            name=f"GEN_{subsys}",
            controllable=True,
        )
        gen_indices[subsys] = idx

    for subsys, value in subsys_demand.items():
        if subsys == "PARAGUAI" or subsys not in bus_lookup:
            continue
        pp.create_load(net, bus=bus_lookup[subsys], p_mw=value, q_mvar=0.0, name=f"LOAD_{subsys}")

    total_demand = sum(subsys_demand.values())
    total_generation = sum(subsys_gen.values())
    if not allow_slack and total_generation + 1e-6 < total_demand:
        raise ValueError(
            f"Insufficient generation ({total_generation:.1f} MW) for demand ({total_demand:.1f} MW) in perfect-balance mode."
        )

    if allow_slack:
        net.ext_grid.loc[slack_idx, ["controllable", "min_p_mw", "max_p_mw"]] = [True, 0.0, 1e6]
    else:
        net.ext_grid.loc[slack_idx, ["controllable", "min_p_mw", "max_p_mw"]] = [True, 0.0, 0.0]

    if not net.poly_cost.empty:
        net.poly_cost.drop(net.poly_cost.index, inplace=True)

    # cost setup: cheap subsystem gen, expensive slack
    SLACK_COST = 1000.0
    GEN_COST = 1.0
    pp.create_poly_cost(net, element=slack_idx, et="ext_grid", cp1_eur_per_mw=SLACK_COST, cp0_eur=0.0)
    for idx in net.gen.index:
        pp.create_poly_cost(net, element=idx, et="gen", cp1_eur_per_mw=GEN_COST, cp0_eur=0.0)

    ensure_scaling(net.load)
    ensure_scaling(net.gen)
    if "controllable" not in net.load.columns:
        net.load["controllable"] = False
    if "controllable" not in net.gen.columns:
        net.gen["controllable"] = True
    else:
        net.gen["controllable"] = True

    pp.rundcopp(net, verbose=False)

    rows = []
    for idx, row in net.gen.iterrows():
        name = str(row["name"])
        subsys = name.replace("GEN_", "")
        pmax = row["max_p_mw"]
        dispatch = net.res_gen.at[idx, "p_mw"]
        curtailed = max(pmax - dispatch, 0.0)
        rows.append({
            "subsystem": subsys,
            "pmax_mw": pmax,
            "dispatch_mw": dispatch,
            "curtailed_mw": curtailed,
        })
    return pd.DataFrame(rows)


def build_month_range(start_year, end_year):
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            yield year, month


def load_forecasts():
    demand_df = pd.read_csv(args.demand_forecast)
    demand_df["scenario"] = demand_df["scenario"].str.lower()
    demand_df = demand_df[demand_df["scenario"] == args.scenario]
    gen_df = pd.read_csv(args.generation_forecast)
    gen_df["subsystem"] = gen_df["subsystem"].str.upper().str.replace("SUDESTE/CENTRO-OESTE", "SUDESTE")
    return demand_df, gen_df


def sample_values(demand_df, gen_df, year, month):
    dvals = (
        demand_df[(demand_df["year"] == year) & (demand_df["month"] == month)]
        .set_index("subsystem")["demand_mw"]
        .to_dict()
    )
    gvals = (
        gen_df[(gen_df["year"] == year) & (gen_df["month"] == month)]
        .set_index("subsystem")["predicted_avg_mw"]
        .to_dict()
    )
    return dvals, gvals


def apply_noise(values, std, cov=None, scale=1.0):
    noisy = {}
    if cov is not None:
        keys = [k for k in values.keys() if k in cov.index]
        if keys:
            cov_matrix = cov.loc[keys, keys].values * (scale ** 2)
            try:
                shock = rng.multivariate_normal(np.zeros(len(keys)), cov_matrix)
            except np.linalg.LinAlgError:
                shock = rng.normal(0.0, std, size=len(keys))
            for key, eps in zip(keys, shock):
                noisy[key] = max(values[key] * np.exp(eps), 0.0)
        remaining = set(values.keys()) - set(keys)
        for key in remaining:
            mult = 1 + rng.normal(0.0, std)
            noisy[key] = max(values[key] * mult, 0.0)
        return noisy
    for key, val in values.items():
        mult = 1 + rng.normal(0.0, std)
        noisy[key] = max(val * mult, 0.0)
    return noisy


def apply_virtual_storage(gen_vals, demand_vals, source, target, amount):
    if not source or not target or amount <= 0:
        return gen_vals, 0.0
    if source == target:
        return gen_vals, 0.0
    if source not in gen_vals:
        return gen_vals, 0.0
    # limit charging to local surplus to avoid infeasible deficits
    local_gen = gen_vals.get(source, 0.0)
    local_demand = demand_vals.get(source, 0.0)
    surplus = max(local_gen - local_demand, 0.0)
    if surplus <= 0:
        return gen_vals, 0.0
    transfer = min(amount, surplus)
    if transfer <= 0:
        return gen_vals, 0.0
    gen_vals[source] = local_gen - transfer
    gen_vals[target] = gen_vals.get(target, 0.0) + transfer
    return gen_vals, transfer


def main():
    demand_df, gen_df = load_forecasts()
    records = []
    month_list = list(build_month_range(args.start_year, args.end_year))
    storage_records = []
    for trial in tqdm(range(1, args.trials + 1)):
        for year, month in month_list:
            dvals, gvals = sample_values(demand_df, gen_df, year, month)
            if not dvals or not gvals:
                continue
            dvals = apply_noise(
                dvals,
                args.demand_noise_std,
                cov=demand_cov,
                scale=args.demand_noise_scale,
            )
            gvals = apply_noise(
                gvals,
                args.gen_noise_std,
                cov=generation_cov,
                scale=args.gen_noise_scale,
            )
            gvals, transferred = apply_virtual_storage(
                gvals, dvals, STORAGE_SOURCE, STORAGE_TARGET, args.storage_transfer_mw
            )
            if transferred > 0:
                storage_records.append({
                    "trial": trial,
                    "year": year,
                    "month": month,
                    "transferred_mw": transferred,
                    "source": STORAGE_SOURCE,
                    "target": STORAGE_TARGET,
                })
            result = run_single_month(dvals, gvals, args.allow_slack_imports)
            result["trial"] = trial
            result["year"] = year
            result["month"] = month
            records.append(result)
    if not records:
        raise RuntimeError("No simulation records produced. Check inputs.")
    final_df = pd.concat(records, ignore_index=True)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(out_path, index=False)
    summary = final_df.groupby(["year", "month"]) ["curtailed_mw"].sum().reset_index()
    summary_path = out_path.with_name(out_path.stem + "_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Saved simulation results to {out_path}")
    print(f"Saved monthly totals to {summary_path}")
    if storage_records:
        storage_df = pd.DataFrame(storage_records)
        storage_path = out_path.with_name(out_path.stem + "_storage.csv")
        storage_df.to_csv(storage_path, index=False)
        print(f"Saved storage transfer log to {storage_path}")


if __name__ == "__main__":
    main()
