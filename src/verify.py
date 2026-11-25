from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .data import Instance
from .model import tight_m_values


def verify_solution(instance: Instance, assignment_csv: Path, servers_csv: Path, expected_objective: float | None = None, tol: float = 1e-7, require_integer: bool = True, check_disaggregated: bool = False):
    assignment = pd.read_csv(assignment_csv)
    server_df = pd.read_csv(servers_csv)
    I = len(instance.servers)
    J = len(instance.tasks)
    server_map = {s: i for i, s in enumerate(instance.servers["server_id"])}
    task_map = {t: j for j, t in enumerate(instance.tasks["task_id"])}
    x = np.zeros((I, J), dtype=float)
    for row in assignment.itertuples(index=False):
        x[server_map[row.server_id], task_map[row.task_id]] = float(row.x)
    y = np.zeros(I, dtype=float)
    for row in server_df.itertuples(index=False):
        y[server_map[row.server_id]] = float(row.y)

    cpu = instance.tasks["cpu_demand"].to_numpy(dtype=float)
    ram = instance.tasks["ram_demand_gb"].to_numpy(dtype=float)
    cc = instance.servers["cpu_capacity"].to_numpy(dtype=float)
    rc = instance.servers["ram_capacity_gb"].to_numpy(dtype=float)
    task_eq_violation = float(np.max(np.abs(x.sum(axis=0) - 1.0)))
    cpu_violation = float(np.maximum(x @ cpu - cc, 0.0).max(initial=0.0))
    ram_violation = float(np.maximum(x @ ram - rc, 0.0).max(initial=0.0))
    m_values = tight_m_values(instance)
    linking_violation = float(np.maximum(x.sum(axis=1) - m_values * y, 0.0).max(initial=0.0))
    individual_link_violation = float(np.maximum(x - y[:, None], 0.0).max(initial=0.0)) if check_disaggregated else 0.0
    active_violation = max(0.0, float(y.sum() - instance.max_active))
    bound_violation = max(0.0, -float(min(x.min(), y.min())), float(max(x.max(), y.max()) - 1.0))
    integrality_violation = 0.0
    if require_integer:
        integrality_violation = float(max(np.max(np.abs(x - np.rint(x))), np.max(np.abs(y - np.rint(y)))))
    objective = float(np.sum(instance.assignment_cost * x) + np.dot(instance.servers["fixed_cost"].to_numpy(dtype=float), y))
    objective_difference = None if expected_objective is None else abs(objective - expected_objective)
    feasible = (
        task_eq_violation <= tol and cpu_violation <= tol and ram_violation <= tol and
        linking_violation <= tol and individual_link_violation <= tol and active_violation <= tol and bound_violation <= tol and
        integrality_violation <= tol and (objective_difference is None or objective_difference <= 1e-6)
    )
    return {
        "tolerance": tol,
        "task_assignment_violation": task_eq_violation,
        "cpu_violation": cpu_violation,
        "ram_violation": ram_violation,
        "linking_violation": linking_violation,
        "individual_link_violation": individual_link_violation,
        "active_server_violation": active_violation,
        "bound_violation": bound_violation,
        "integrality_violation": integrality_violation,
        "objective_recomputed": objective,
        "objective_difference": objective_difference,
        "feasible": bool(feasible),
    }


def save_verification(report: dict, path: Path) -> None:
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
