from __future__ import annotations
from pathlib import Path
import json
import math
import platform
import subprocess
import sys
import time
import numpy as np
import pandas as pd
import scipy
import scipy.optimize._highspy._core as highs_core
from src.data import generate_instance, save_instance, load_instance, clone_with_capacity_scales, SEED, VARIANT
from src.model import solve_instance, save_short_log
from src.verify import verify_solution, save_verification
from src.heuristic import greedy_assign
from src.plots import plot_assignment, plot_scenarios, plot_scaling, plot_bigm, plot_formulations

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
PLOTS = RESULTS / "plots"


def save_output(prefix, out):
    out.assignment.to_csv(RESULTS / f"{prefix}_assignment.csv", index=False, float_format="%.9f")
    out.servers.to_csv(RESULTS / f"{prefix}_servers.csv", index=False, float_format="%.9f")


def active_set(server_df):
    return set(server_df.loc[server_df["y"] > 0.5, "server_id"].tolist())


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    instance = generate_instance(seed=SEED, n_servers=8, n_tasks=15, max_active=5)
    save_instance(instance, DATA)
    instance = load_instance(DATA)
    highs_version = f"{highs_core.HIGHS_VERSION_MAJOR}.{highs_core.HIGHS_VERSION_MINOR}.{highs_core.HIGHS_VERSION_PATCH}"

    lp = solve_instance(instance, relax=True, formulation="aggregate", m_factor=1.0)
    milp_out = solve_instance(instance, relax=False, formulation="aggregate", m_factor=1.0)
    if not lp.result.success or not milp_out.result.success:
        raise RuntimeError("Base LP/MILP solve failed")
    save_output("base_lp", lp)
    save_output("base_milp", milp_out)
    save_short_log(lp, RESULTS / "solver_base_lp.log", scipy.__version__, highs_version)
    save_short_log(milp_out, RESULTS / "solver_base_milp.log", scipy.__version__, highs_version)

    with (RESULTS / "solver_base_milp_full.log").open("w", encoding="utf-8") as f:
        subprocess.run([sys.executable, "-m", "src.diagnostic", "--data", str(DATA)], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True)

    verification = verify_solution(instance, RESULTS / "base_milp_assignment.csv", RESULTS / "base_milp_servers.csv", milp_out.objective, require_integer=True)
    save_verification(verification, RESULTS / "verification_base.json")
    if not verification["feasible"]:
        raise RuntimeError("Base independent verification failed")

    gap = (milp_out.objective - lp.objective) / milp_out.objective * 100.0
    base_active = active_set(milp_out.servers)
    load_score = np.maximum(
        milp_out.servers["cpu_used"].to_numpy() / milp_out.servers["cpu_capacity"].to_numpy(),
        milp_out.servers["ram_used_gb"].to_numpy() / milp_out.servers["ram_capacity_gb"].to_numpy(),
    )
    active_indices = np.where(milp_out.servers["y"].to_numpy() > 0.5)[0]
    ranked = active_indices[np.argsort(-load_score[active_indices])]

    deficit_idx = int(ranked[0])
    deficit_instance = clone_with_capacity_scales(instance, {deficit_idx: 0.85})
    deficit = solve_instance(deficit_instance, relax=False, formulation="aggregate", m_factor=1.0)
    if not deficit.result.success:
        raise RuntimeError("Deficit scenario failed")
    save_output("deficit", deficit)
    save_short_log(deficit, RESULTS / "solver_deficit.log", scipy.__version__, highs_version)
    ver_def = verify_solution(deficit_instance, RESULTS / "deficit_assignment.csv", RESULTS / "deficit_servers.csv", deficit.objective, require_integer=True)
    save_verification(ver_def, RESULTS / "verification_deficit.json")
    if not ver_def["feasible"]:
        raise RuntimeError("Deficit verification failed")

    stress_indices = [int(ranked[0]), int(ranked[1])]
    stress_instance = clone_with_capacity_scales(instance, {stress_indices[0]: 0.80, stress_indices[1]: 0.80})
    stress = solve_instance(stress_instance, relax=False, formulation="aggregate", m_factor=1.0)
    if not stress.result.success:
        raise RuntimeError("Stress scenario failed")
    save_output("stress", stress)
    save_short_log(stress, RESULTS / "solver_stress.log", scipy.__version__, highs_version)
    ver_st = verify_solution(stress_instance, RESULTS / "stress_assignment.csv", RESULTS / "stress_servers.csv", stress.objective, require_integer=True)
    save_verification(ver_st, RESULTS / "verification_stress.json")
    if not ver_st["feasible"]:
        raise RuntimeError("Stress verification failed")

    scenario_rows = []
    set_rows = []
    for name, out in [("base", milp_out), ("deficit", deficit), ("stress", stress)]:
        aset = active_set(out.servers)
        scenario_rows.append({
            "scenario": name,
            "status": "optimal",
            "objective": out.objective,
            "delta_vs_base": out.objective - milp_out.objective,
            "active_servers": ", ".join(sorted(aset)),
        })
        set_rows.append({
            "scenario": name,
            "removed": ", ".join(sorted(base_active - aset)) or "-",
            "added": ", ".join(sorted(aset - base_active)) or "-",
        })
    scenarios = pd.DataFrame(scenario_rows)
    scenarios.to_csv(RESULTS / "scenarios.csv", index=False, float_format="%.9f")
    pd.DataFrame(set_rows).to_csv(RESULTS / "scenario_server_sets.csv", index=False)

    greedy = greedy_assign(instance)
    if greedy is None:
        raise RuntimeError("Greedy heuristic failed")
    greedy_obj, greedy_assignment, greedy_servers, greedy_time = greedy
    greedy_assignment.to_csv(RESULTS / "greedy_assignment.csv", index=False, float_format="%.9f")
    greedy_servers.to_csv(RESULTS / "greedy_servers.csv", index=False, float_format="%.9f")
    greedy_ver = verify_solution(instance, RESULTS / "greedy_assignment.csv", RESULTS / "greedy_servers.csv", greedy_obj, require_integer=True)
    save_verification(greedy_ver, RESULTS / "verification_greedy.json")
    if not greedy_ver["feasible"]:
        raise RuntimeError("Greedy verification failed")
    greedy_summary = {"objective": greedy_obj, "elapsed_s": greedy_time, "gap_pct": (greedy_obj - milp_out.objective) / milp_out.objective * 100.0}
    (RESULTS / "greedy_summary.json").write_text(json.dumps(greedy_summary, indent=2), encoding="utf-8")

    scale_rows = []
    for n_tasks, n_servers in [(10, 6), (20, 10), (40, 16)]:
        scaled = generate_instance(seed=SEED + n_tasks, n_servers=n_servers, n_tasks=n_tasks, max_active=max(3, math.ceil(n_servers * 0.6)))
        lp_s = solve_instance(scaled, relax=True, formulation="aggregate", m_factor=1.0)
        mi_s = solve_instance(scaled, relax=False, formulation="aggregate", m_factor=1.0)
        if not lp_s.result.success or not mi_s.result.success:
            raise RuntimeError(f"Scaling solve failed for {n_tasks} tasks")
        scale_rows.append({
            "n_tasks": n_tasks,
            "n_servers": n_servers,
            "lp_time_s": lp_s.elapsed_s,
            "milp_time_s": mi_s.elapsed_s,
            "integrality_gap_pct": (mi_s.objective - lp_s.objective) / mi_s.objective * 100.0,
            "mip_nodes": int(getattr(mi_s.result, "mip_node_count", 0) or 0),
        })
    scaling = pd.DataFrame(scale_rows)
    scaling.to_csv(RESULTS / "scaling.csv", index=False, float_format="%.9f")

    bigm_rows = []
    for factor in [1.0, 10.0, 100.0]:
        lp_m = solve_instance(instance, relax=True, formulation="aggregate", m_factor=factor)
        mi_m = solve_instance(instance, relax=False, formulation="aggregate", m_factor=factor)
        if not lp_m.result.success or not mi_m.result.success:
            raise RuntimeError("big-M experiment failed")
        bigm_rows.append({
            "m_factor": factor,
            "lp_objective": lp_m.objective,
            "lp_gap_pct": (mi_m.objective - lp_m.objective) / mi_m.objective * 100.0,
            "milp_objective": mi_m.objective,
            "milp_time_s": mi_m.elapsed_s,
            "mip_nodes": int(getattr(mi_m.result, "mip_node_count", 0) or 0),
        })
    bigm = pd.DataFrame(bigm_rows)
    bigm.to_csv(RESULTS / "bigm_experiment.csv", index=False, float_format="%.9f")

    form_rows = []
    for formulation in ["aggregate", "disaggregated"]:
        lp_f = solve_instance(instance, relax=True, formulation=formulation, m_factor=1.0)
        mi_f = solve_instance(instance, relax=False, formulation=formulation, m_factor=1.0, mip_rel_gap=1e-7)
        if not lp_f.result.success or not mi_f.result.success:
            raise RuntimeError("Formulation experiment failed")
        for mode, out in [("LP relaxation", lp_f), ("MILP", mi_f)]:
            form_rows.append({
                "formulation": formulation,
                "mode": mode,
                "objective": out.objective,
                "gap_pct": (mi_f.objective - out.objective) / mi_f.objective * 100.0,
                "time_s": out.elapsed_s,
                "n_constraints": out.n_constraints,
                "mip_nodes": int(getattr(out.result, "mip_node_count", 0) or 0),
            })
    formulations = pd.DataFrame(form_rows)
    formulations.to_csv(RESULTS / "formulation_comparison.csv", index=False, float_format="%.9f")

    objective_aggregate = float(formulations.loc[(formulations["formulation"] == "aggregate") & (formulations["mode"] == "MILP"), "objective"].iloc[0])
    objective_disaggregated = float(formulations.loc[(formulations["formulation"] == "disaggregated") & (formulations["mode"] == "MILP"), "objective"].iloc[0])
    lp_aggregate = float(formulations.loc[(formulations["formulation"] == "aggregate") & (formulations["mode"] == "LP relaxation"), "objective"].iloc[0])
    lp_disaggregated = float(formulations.loc[(formulations["formulation"] == "disaggregated") & (formulations["mode"] == "LP relaxation"), "objective"].iloc[0])
    if abs(objective_aggregate - objective_disaggregated) > 1e-6:
        raise RuntimeError("The formulations have different integer optima")
    if lp_disaggregated + 1e-7 < lp_aggregate:
        raise RuntimeError("The strengthened LP relaxation is unexpectedly weaker")

    for formulation in ("aggregate", "disaggregated"):
        with (RESULTS / f"solver_formulation_{formulation}_milp_full.log").open("w", encoding="utf-8") as stream:
            subprocess.run(
                [sys.executable, "-m", "src.diagnostic", "--data", str(DATA), "--formulation", formulation, "--tight-gap"],
                cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True,
            )
        with (RESULTS / f"solver_formulation_{formulation}_lp_full.log").open("w", encoding="utf-8") as stream:
            subprocess.run(
                [sys.executable, "-m", "src.diagnostic", "--data", str(DATA), "--formulation", formulation, "--relax"],
                cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True,
            )

    strengthened = solve_instance(instance, relax=False, formulation="disaggregated", m_factor=1.0, mip_rel_gap=1e-7)
    if not strengthened.result.success:
        raise RuntimeError("Strengthened MILP solution failed")
    save_output("formulation_disaggregated_milp", strengthened)
    strengthened_check = verify_solution(
        instance,
        RESULTS / "formulation_disaggregated_milp_assignment.csv",
        RESULTS / "formulation_disaggregated_milp_servers.csv",
        strengthened.objective,
        require_integer=True,
        check_disaggregated=True,
    )
    save_verification(strengthened_check, RESULTS / "verification_disaggregated.json")
    if not strengthened_check["feasible"]:
        raise RuntimeError("Strengthened formulation independent verification failed")

    plot_assignment(milp_out.assignment, PLOTS / "base_assignment.png", "MILP: назначение задач")
    plot_scenarios(scenarios, PLOTS / "scenarios.png")
    plot_scaling(scaling, PLOTS / "scaling_time.png", PLOTS / "scaling_gap.png")
    plot_bigm(bigm, PLOTS / "bigm_gap.png")
    plot_formulations(formulations, PLOTS / "formulation_gap.png")

    run_info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "highs": highs_version,
        "variant": VARIANT,
        "seed": SEED,
        "base_lp_objective": lp.objective,
        "base_milp_objective": milp_out.objective,
        "integrality_gap_pct": gap,
        "mip_gap": float(getattr(milp_out.result, "mip_gap", 0.0) or 0.0),
        "mip_node_count": int(getattr(milp_out.result, "mip_node_count", 0) or 0),
        "deficit_server": instance.servers.loc[deficit_idx, "server_id"],
        "stress_servers": [instance.servers.loc[i, "server_id"] for i in stress_indices],
    }
    (RESULTS / "run_info.json").write_text(json.dumps(run_info, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Variant: {VARIANT}; seed: {SEED}")
    print(f"LP relaxation: {lp.objective:.3f}")
    print(f"MILP: {milp_out.objective:.3f}; gap={gap:.3f}%")
    print(f"Base verification: {verification['feasible']}")
    print(f"Deficit: {deficit.objective:.3f}; active={', '.join(sorted(active_set(deficit.servers)))}")
    print(f"Stress: {stress.objective:.3f}; active={', '.join(sorted(active_set(stress.servers)))}")
    print(f"Greedy: {greedy_obj:.3f}; gap={greedy_summary['gap_pct']:.3f}%")

if __name__ == "__main__":
    main()
