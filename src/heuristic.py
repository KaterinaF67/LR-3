from __future__ import annotations
import time
import numpy as np
import pandas as pd
from .data import Instance


def greedy_assign(instance: Instance):
    t0 = time.perf_counter()
    I = len(instance.servers)
    J = len(instance.tasks)
    cpu = instance.tasks["cpu_demand"].to_numpy(dtype=float)
    ram = instance.tasks["ram_demand_gb"].to_numpy(dtype=float)
    cc = instance.servers["cpu_capacity"].to_numpy(dtype=float)
    rc = instance.servers["ram_capacity_gb"].to_numpy(dtype=float)
    fixed = instance.servers["fixed_cost"].to_numpy(dtype=float)
    score = cpu / cpu.mean() + ram / ram.mean()
    orders = [np.argsort(-score), np.argsort(-cpu), np.argsort(-ram)]
    best = None
    for order in orders:
        rem_cpu = cc.copy()
        rem_ram = rc.copy()
        active = set()
        assignment = np.full(J, -1, dtype=int)
        total = 0.0
        feasible = True
        for j in order:
            candidates = []
            for i in range(I):
                if rem_cpu[i] + 1e-12 < cpu[j] or rem_ram[i] + 1e-12 < ram[j]:
                    continue
                if i not in active and len(active) >= instance.max_active:
                    continue
                incremental = instance.assignment_cost[i, j] + (fixed[i] if i not in active else 0.0)
                candidates.append((incremental, instance.assignment_cost[i, j], i))
            if not candidates:
                feasible = False
                break
            _, _, i = min(candidates)
            if i not in active:
                active.add(i)
                total += fixed[i]
            assignment[j] = i
            rem_cpu[i] -= cpu[j]
            rem_ram[i] -= ram[j]
            total += instance.assignment_cost[i, j]
        if feasible and (best is None or total < best[0]):
            best = (total, assignment.copy(), set(active))
    elapsed = time.perf_counter() - t0
    if best is None:
        return None
    total, assignment_idx, active = best
    rows = []
    for i in range(I):
        for j in range(J):
            x = 1.0 if assignment_idx[j] == i else 0.0
            rows.append({
                "server_id": instance.servers.loc[i, "server_id"],
                "task_id": instance.tasks.loc[j, "task_id"],
                "x": x,
                "assignment_cost": instance.assignment_cost[i, j],
                "cost_contribution": x * instance.assignment_cost[i, j],
            })
    server_rows = []
    for i in range(I):
        tasks = [j for j in range(J) if assignment_idx[j] == i]
        server_rows.append({
            "server_id": instance.servers.loc[i, "server_id"],
            "y": 1.0 if i in active else 0.0,
            "cpu_capacity": cc[i],
            "cpu_used": float(cpu[tasks].sum()) if tasks else 0.0,
            "ram_capacity_gb": rc[i],
            "ram_used_gb": float(ram[tasks].sum()) if tasks else 0.0,
            "fixed_cost": fixed[i],
            "assigned_tasks": float(len(tasks)),
        })
    return float(total), pd.DataFrame(rows), pd.DataFrame(server_rows), elapsed
