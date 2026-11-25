from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_assignment(assignment: pd.DataFrame, path: Path, title: str) -> None:
    pivot = assignment.pivot(index="server_id", columns="task_id", values="x")
    pivot = pivot.reindex(
        index=sorted(pivot.index, key=lambda label: int(label[1:])),
        columns=sorted(pivot.columns, key=lambda label: int(label[1:])),
    )
    fig, ax = plt.subplots(figsize=(10, 4.8))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.set_xlabel("Задачи")
    ax.set_ylabel("Серверы")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="x(i,j)")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_scenarios(scenarios: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(scenarios["scenario"], scenarios["objective"])
    ax.set_ylabel("Стоимость")
    ax.set_title("Сценарный анализ MILP")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_scaling(df: pd.DataFrame, time_path: Path, gap_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(df["n_tasks"], df["milp_time_s"], marker="o", label="MILP")
    ax.plot(df["n_tasks"], df["lp_time_s"], marker="o", label="LP relaxation")
    ax.set_xlabel("Число задач")
    ax.set_ylabel("Время, с")
    ax.set_title("Масштабирование: время решения")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(time_path, dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(df["n_tasks"], df["integrality_gap_pct"], marker="o")
    ax.set_xlabel("Число задач")
    ax.set_ylabel("Интегральный зазор, %")
    ax.set_title("Масштабирование: LP/MILP зазор")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(gap_path, dpi=170)
    plt.close(fig)


def plot_bigm(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(df["m_factor"].astype(str), df["lp_gap_pct"], marker="o")
    ax.set_xlabel("Множитель M")
    ax.set_ylabel("Зазор LP-релаксации, %")
    ax.set_title("Влияние завышенного big-M")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_formulations(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    lp = df[df["mode"] == "LP relaxation"]
    ax.bar(lp["formulation"], lp["gap_pct"])
    ax.set_ylabel("Зазор до MILP, %")
    ax.set_title("Доп. задание: сила LP-релаксаций")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
