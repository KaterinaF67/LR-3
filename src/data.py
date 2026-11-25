from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd

N = 25
Q = 1
Y = 26
VARIANT = 1 + ((17 * N + 7 * Q + Y) % 20)
SEED = 10000 + 100 * Q + N

@dataclass
class Instance:
    servers: pd.DataFrame
    tasks: pd.DataFrame
    assignment_cost: np.ndarray
    max_active: int


def generate_instance(seed: int = SEED, n_servers: int = 8, n_tasks: int = 15, max_active: int = 5) -> Instance:
    rng = np.random.default_rng(seed)
    cpu_demand = rng.integers(2, 7, n_tasks)
    ram_demand = rng.integers(3, 11, n_tasks)
    cpu_capacity = rng.integers(14, 22, n_servers)
    ram_capacity = rng.integers(24, 39, n_servers)
    fixed_cost = np.round(rng.uniform(22, 45, n_servers), 2)
    assignment_cost = np.round(rng.uniform(3, 15, (n_servers, n_tasks)), 2)
    servers = pd.DataFrame({
        "server_id": [f"S{i+1}" for i in range(n_servers)],
        "cpu_capacity": cpu_capacity.astype(float),
        "ram_capacity_gb": ram_capacity.astype(float),
        "fixed_cost": fixed_cost,
    })
    tasks = pd.DataFrame({
        "task_id": [f"T{j+1}" for j in range(n_tasks)],
        "cpu_demand": cpu_demand.astype(float),
        "ram_demand_gb": ram_demand.astype(float),
    })
    return Instance(servers, tasks, assignment_cost, max_active)


def save_instance(instance: Instance, data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    instance.servers.to_csv(data_dir / "servers.csv", index=False)
    instance.tasks.to_csv(data_dir / "tasks.csv", index=False)
    pd.DataFrame(
        instance.assignment_cost,
        index=instance.servers["server_id"],
        columns=instance.tasks["task_id"],
    ).rename_axis("server_id").to_csv(data_dir / "assignment_cost.csv", float_format="%.4f")
    metadata = {
        "student_number_N": N,
        "group_index_Q": Q,
        "year_Y": Y,
        "variant_formula": "1 + ((17*N + 7*Q + Y) mod 20)",
        "variant": VARIANT,
        "seed_formula": "10000 + 100*Q + N",
        "seed": SEED,
        "n_servers": len(instance.servers),
        "n_tasks": len(instance.tasks),
        "max_active_servers": int(instance.max_active),
        "units": {
            "cpu_capacity": "CPU units",
            "ram_capacity_gb": "GB",
            "fixed_cost": "cost units per planning period",
            "cpu_demand": "CPU units",
            "ram_demand_gb": "GB",
            "assignment_cost": "cost units per task assignment",
        },
    }
    (data_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def load_instance(data_dir: Path) -> Instance:
    servers = pd.read_csv(data_dir / "servers.csv")
    tasks = pd.read_csv(data_dir / "tasks.csv")
    assignment_cost = pd.read_csv(data_dir / "assignment_cost.csv", index_col=0).to_numpy(dtype=float)
    metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    return Instance(servers, tasks, assignment_cost, int(metadata["max_active_servers"]))


def clone_with_capacity_scales(instance: Instance, scales: dict[int, float]) -> Instance:
    servers = instance.servers.copy(deep=True)
    for i, factor in scales.items():
        servers.loc[i, "cpu_capacity"] *= factor
        servers.loc[i, "ram_capacity_gb"] *= factor
    return Instance(servers, instance.tasks.copy(deep=True), instance.assignment_cost.copy(), instance.max_active)
