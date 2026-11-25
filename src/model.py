from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import lil_matrix, csr_matrix
from .data import Instance

@dataclass
class SolveOutput:
    result: object
    assignment: pd.DataFrame
    servers: pd.DataFrame
    objective: float | None
    elapsed_s: float
    formulation: str
    relax: bool
    m_factor: float
    n_constraints: int
    tight_m: np.ndarray


def tight_m_values(instance: Instance) -> np.ndarray:
    cpu = instance.tasks["cpu_demand"].to_numpy(dtype=float)
    ram = instance.tasks["ram_demand_gb"].to_numpy(dtype=float)
    cc = instance.servers["cpu_capacity"].to_numpy(dtype=float)
    rc = instance.servers["ram_capacity_gb"].to_numpy(dtype=float)
    j = len(cpu)
    values = []
    for i in range(len(cc)):
        m = min(j, int(np.floor(cc[i] / cpu.min())), int(np.floor(rc[i] / ram.min())))
        values.append(max(1, m))
    return np.asarray(values, dtype=float)


def build_model(instance: Instance, formulation: str = "aggregate", m_factor: float = 1.0):
    I = len(instance.servers)
    J = len(instance.tasks)
    n = I * J + I
    cpu = instance.tasks["cpu_demand"].to_numpy(dtype=float)
    ram = instance.tasks["ram_demand_gb"].to_numpy(dtype=float)
    cc = instance.servers["cpu_capacity"].to_numpy(dtype=float)
    rc = instance.servers["ram_capacity_gb"].to_numpy(dtype=float)
    fixed = instance.servers["fixed_cost"].to_numpy(dtype=float)
    c = np.concatenate([instance.assignment_cost.reshape(-1), fixed])
    lower = np.zeros(n, dtype=float)
    upper = np.ones(n, dtype=float)
    rows = []
    lows = []
    highs = []

    for j in range(J):
        row = lil_matrix((1, n), dtype=float)
        for i in range(I):
            row[0, i * J + j] = 1.0
        rows.append(row.tocsr())
        lows.append(1.0)
        highs.append(1.0)

    for i in range(I):
        row = lil_matrix((1, n), dtype=float)
        for j in range(J):
            row[0, i * J + j] = cpu[j]
        rows.append(row.tocsr())
        lows.append(-np.inf)
        highs.append(cc[i])

        row = lil_matrix((1, n), dtype=float)
        for j in range(J):
            row[0, i * J + j] = ram[j]
        rows.append(row.tocsr())
        lows.append(-np.inf)
        highs.append(rc[i])

    m_values = tight_m_values(instance)
    for i in range(I):
        row = lil_matrix((1, n), dtype=float)
        for j in range(J):
            row[0, i * J + j] = 1.0
        row[0, I * J + i] = -m_values[i] * m_factor
        rows.append(row.tocsr())
        lows.append(-np.inf)
        highs.append(0.0)

    row = lil_matrix((1, n), dtype=float)
    for i in range(I):
        row[0, I * J + i] = 1.0
    rows.append(row.tocsr())
    lows.append(-np.inf)
    highs.append(float(instance.max_active))

    if formulation == "disaggregated":
        for i in range(I):
            for j in range(J):
                row = lil_matrix((1, n), dtype=float)
                row[0, i * J + j] = 1.0
                row[0, I * J + i] = -1.0
                rows.append(row.tocsr())
                lows.append(-np.inf)
                highs.append(0.0)

    A = csr_matrix(np.vstack([r.toarray() for r in rows]))
    constraint = LinearConstraint(A, np.asarray(lows), np.asarray(highs))
    return c, Bounds(lower, upper), constraint, m_values, A.shape[0]


def solve_instance(instance: Instance, relax: bool = False, formulation: str = "aggregate", m_factor: float = 1.0, disp: bool = False, mip_rel_gap: float | None = None) -> SolveOutput:
    c, bounds, constraint, m_values, n_constraints = build_model(instance, formulation, m_factor)
    integrality = np.zeros(len(c), dtype=int) if relax else np.ones(len(c), dtype=int)
    t0 = time.perf_counter()
    options = {"presolve": True, "disp": disp}
    if not relax and mip_rel_gap is not None:
        options["mip_rel_gap"] = mip_rel_gap
    result = milp(c=c, integrality=integrality, bounds=bounds, constraints=constraint, options=options)
    elapsed = time.perf_counter() - t0
    if not result.success:
        return SolveOutput(result, pd.DataFrame(), pd.DataFrame(), None, elapsed, formulation, relax, m_factor, n_constraints, m_values)

    I = len(instance.servers)
    J = len(instance.tasks)
    x = np.asarray(result.x[:I * J]).reshape(I, J)
    y = np.asarray(result.x[I * J:])
    rows = []
    for i in range(I):
        for j in range(J):
            rows.append({
                "server_id": instance.servers.loc[i, "server_id"],
                "task_id": instance.tasks.loc[j, "task_id"],
                "x": x[i, j],
                "assignment_cost": instance.assignment_cost[i, j],
                "cost_contribution": x[i, j] * instance.assignment_cost[i, j],
            })
    assignment = pd.DataFrame(rows)
    cpu = instance.tasks["cpu_demand"].to_numpy(dtype=float)
    ram = instance.tasks["ram_demand_gb"].to_numpy(dtype=float)
    cpu_used = x @ cpu
    ram_used = x @ ram
    server_rows = []
    for i in range(I):
        server_rows.append({
            "server_id": instance.servers.loc[i, "server_id"],
            "y": y[i],
            "cpu_capacity": instance.servers.loc[i, "cpu_capacity"],
            "cpu_used": cpu_used[i],
            "ram_capacity_gb": instance.servers.loc[i, "ram_capacity_gb"],
            "ram_used_gb": ram_used[i],
            "fixed_cost": instance.servers.loc[i, "fixed_cost"],
            "assigned_tasks": x[i].sum(),
            "tight_M": m_values[i],
        })
    servers = pd.DataFrame(server_rows)
    return SolveOutput(result, assignment, servers, float(result.fun), elapsed, formulation, relax, m_factor, n_constraints, m_values)


def save_short_log(output: SolveOutput, path: Path, scipy_version: str, highs_version: str) -> None:
    r = output.result
    lines = [
        f"formulation={output.formulation}",
        f"relax={output.relax}",
        f"m_factor={output.m_factor}",
        f"scipy_version={scipy_version}",
        f"highs_version={highs_version}",
        f"status={getattr(r, 'status', None)}",
        f"success={getattr(r, 'success', None)}",
        f"message={getattr(r, 'message', '')}",
        f"objective={getattr(r, 'fun', None)}",
        f"mip_gap={getattr(r, 'mip_gap', None)}",
        f"mip_node_count={getattr(r, 'mip_node_count', None)}",
        f"elapsed_s={output.elapsed_s:.9f}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
