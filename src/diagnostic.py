from __future__ import annotations
import argparse
from pathlib import Path
from .data import load_instance
from .model import solve_instance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--formulation", choices=("aggregate", "disaggregated"), default="aggregate")
    parser.add_argument("--relax", action="store_true")
    parser.add_argument("--tight-gap", action="store_true")
    args = parser.parse_args()
    instance = load_instance(args.data)
    out = solve_instance(
        instance, relax=args.relax, formulation=args.formulation, m_factor=1.0, disp=True,
        mip_rel_gap=1e-7 if args.tight_gap and not args.relax else None,
    )
    r = out.result
    print("\n--- SciPy/HiGHS result summary ---")
    print(f"formulation={args.formulation}")
    print(f"relaxation={args.relax}")
    print(f"n_constraints={out.n_constraints}")
    print(f"success={r.success}")
    print(f"status={r.status}")
    print(f"message={r.message}")
    print(f"objective={getattr(r, 'fun', None)}")
    print(f"mip_gap={getattr(r, 'mip_gap', None)}")
    print(f"mip_node_count={getattr(r, 'mip_node_count', None)}")
    if not r.success:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
