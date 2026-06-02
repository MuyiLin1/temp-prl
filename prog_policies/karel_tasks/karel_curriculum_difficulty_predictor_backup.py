#!/usr/bin/env python3
"""
Karel curriculum difficulty predictor for CleanHouse / PRL task generation.

This script implements a lightweight curriculum difficulty predictor for Karel
task files such as clean_house.py.

Outputs:
  1. Task descriptor features phi(tau)
  2. Optional RTD novelty score using prior task measurements
  3. Optional predicted success / return using ridge regression
  4. Curriculum utility score

Usage without prior measurements:
  python karel_curriculum_difficulty_predictor.py \
    --task-file clean_house.py \
    --task-name CleanHouse \
    --features-csv cleanhouse_features.csv \
    --out cleanhouse_difficulty_report.json

Create prior-measurement template:
  python karel_curriculum_difficulty_predictor.py \
    --task-file clean_house.py \
    --task-name CleanHouse \
    --write-prior-template prior_karel_measurements_template.csv

Usage with prior measurements:
  python karel_curriculum_difficulty_predictor.py \
    --task-file clean_house.py \
    --task-name CleanHouse \
    --prior-csv prior_karel_measurements.csv \
    --target-policy current_policy \
    --out cleanhouse_difficulty_report.json
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import pandas as pd
except Exception:
    pd = None


def read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def extract_world_map(source: str) -> Optional[List[List[Any]]]:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "world_map":
                    try:
                        value = ast.literal_eval(node.value)
                    except Exception:
                        return None
                    if isinstance(value, list) and value and isinstance(value[0], list):
                        return value
    return None


def extract_agent_pos(source: str) -> Optional[Tuple[int, int]]:
    m = re.search(r"agent_pos\s*=\s*\((\d+)\s*,\s*(\d+)\)", source)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def count_ast_nodes(source: str) -> Dict[str, float]:
    tree = ast.parse(source)
    counts = {
        "src_num_classes": 0.0,
        "src_num_functions": 0.0,
        "src_num_ifs": 0.0,
        "src_num_fors": 0.0,
        "src_num_whiles": 0.0,
        "src_num_assigns": 0.0,
        "src_num_calls": 0.0,
        "src_num_literals": 0.0,
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            counts["src_num_classes"] += 1
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            counts["src_num_functions"] += 1
        elif isinstance(node, ast.If):
            counts["src_num_ifs"] += 1
        elif isinstance(node, ast.For):
            counts["src_num_fors"] += 1
        elif isinstance(node, ast.While):
            counts["src_num_whiles"] += 1
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            counts["src_num_assigns"] += 1
        elif isinstance(node, ast.Call):
            counts["src_num_calls"] += 1
        elif isinstance(node, ast.Constant):
            counts["src_num_literals"] += 1

    return counts


def free_cells_from_world_map(
    world_map: List[List[Any]],
) -> Tuple[List[Tuple[int, int]], set[Tuple[int, int]]]:
    free = []
    walls = set()

    for y, row in enumerate(world_map):
        for x, val in enumerate(row):
            if val == "-":
                walls.add((y, x))
            else:
                free.append((y, x))

    return free, walls


def build_grid_graph(
    free_cells: Sequence[Tuple[int, int]],
) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
    free_set = set(free_cells)
    graph = {c: [] for c in free_cells}

    for y, x in free_cells:
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nb = (y + dy, x + dx)
            if nb in free_set:
                graph[(y, x)].append(nb)

    return graph


def bfs_distances(
    graph: Dict[Tuple[int, int], List[Tuple[int, int]]],
    start: Tuple[int, int],
) -> Dict[Tuple[int, int], int]:
    if start not in graph:
        return {}

    dist = {start: 0}
    q = deque([start])

    while q:
        u = q.popleft()
        for v in graph[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)

    return dist


def connected_components(
    graph: Dict[Tuple[int, int], List[Tuple[int, int]]],
) -> List[List[Tuple[int, int]]]:
    seen = set()
    comps = []

    for node in graph:
        if node in seen:
            continue
        dist = bfs_distances(graph, node)
        comp = list(dist.keys())
        seen.update(comp)
        comps.append(comp)

    return comps


def bridges_and_articulations(
    graph: Dict[Tuple[int, int], List[Tuple[int, int]]],
) -> Tuple[int, int]:
    time = 0
    tin = {}
    low = {}
    bridges = 0
    articulations = set()

    def dfs(u: Tuple[int, int], parent: Optional[Tuple[int, int]]) -> None:
        nonlocal time, bridges

        tin[u] = low[u] = time
        time += 1
        children = 0

        for v in graph[u]:
            if v == parent:
                continue

            if v in tin:
                low[u] = min(low[u], tin[v])
            else:
                children += 1
                dfs(v, u)
                low[u] = min(low[u], low[v])

                if low[v] > tin[u]:
                    bridges += 1

                if parent is not None and low[v] >= tin[u]:
                    articulations.add(u)

        if parent is None and children > 1:
            articulations.add(u)

    for u in graph:
        if u not in tin:
            dfs(u, None)

    return bridges, len(articulations)


def graph_laplacian_eigs(
    graph: Dict[Tuple[int, int], List[Tuple[int, int]]],
    k: int = 6,
) -> List[float]:
    nodes = list(graph.keys())
    n = len(nodes)

    if n == 0:
        return [0.0] * k

    idx = {node: i for i, node in enumerate(nodes)}
    A = np.zeros((n, n), dtype=float)

    for u, nbrs in graph.items():
        for v in nbrs:
            A[idx[u], idx[v]] = 1.0

    deg = A.sum(axis=1)
    D_inv_sqrt = np.zeros_like(deg)
    mask = deg > 0
    D_inv_sqrt[mask] = 1.0 / np.sqrt(deg[mask])

    L = np.eye(n) - (D_inv_sqrt[:, None] * A * D_inv_sqrt[None, :])
    vals = np.linalg.eigvalsh(L)
    vals = np.sort(np.real(vals))
    vals = vals[:k]

    if len(vals) < k:
        vals = np.pad(vals, (0, k - len(vals)))

    return [float(x) for x in vals]


def estimate_cleanhouse_marker_count(source: str) -> float:
    m = re.search(r"possible_marker_locations\s*\[:\s*(\d+)\s*\]", source)
    base = float(m.group(1)) if m else 10.0
    bonus = 1.0 if "put 1 marker near start" in source or "agent_pos[0]+1" in source else 0.0
    return base + bonus


def infer_reward_features(source: str, task_name: str) -> Dict[str, float]:
    lower = (source + "\n" + task_name).lower()

    return {
        "reward_pick_markers": float(
            "pickmarker" in lower
            or "picked-up" in lower
            or "pick up" in lower
            or "markers_grid.sum" in lower
        ),
        "reward_put_markers": float("putmarker" in lower or "put marker" in lower),
        "reward_reach_goal": float("reach" in lower or "goal" in lower),
        "reward_visit_cells": float("visited" in lower or "visit" in lower),
        "reward_sparse": float("sparse" in lower),
        "reward_crash_penalty_used": float("crash_penalty" in lower),
        "reward_num_subgoals_est": estimate_cleanhouse_marker_count(source)
        if "cleanhouse" in lower
        else 1.0,
        "reward_terminal_only": float("reward = 1." in lower and "num_markers == 0" in lower),
    }


def cleanhouse_descriptor(
    task_file: str | Path,
    task_name: str = "CleanHouse",
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    source = read_text(task_file)
    world_map = extract_world_map(source)

    if world_map is None:
        raise ValueError(f"Could not find a literal world_map assignment in {task_file}")

    h = len(world_map)
    w = len(world_map[0]) if h else 0

    free_cells, walls = free_cells_from_world_map(world_map)
    graph = build_grid_graph(free_cells)
    comps = connected_components(graph)

    largest = max((len(c) for c in comps), default=0)
    degrees = np.array([len(nbrs) for nbrs in graph.values()], dtype=float) if graph else np.array([0.0])
    bridges, arts = bridges_and_articulations(graph)

    agent_pos = extract_agent_pos(source)
    start_dist = bfs_distances(graph, agent_pos) if agent_pos else {}
    start_reachable = len(start_dist)
    start_ecc = max(start_dist.values()) if start_dist else 0

    diameter = 0
    mean_pair_dist_accum = 0.0
    mean_pair_count = 0

    for node in graph:
        d = bfs_distances(graph, node)
        if d:
            diameter = max(diameter, max(d.values()))
            mean_pair_dist_accum += sum(d.values())
            mean_pair_count += len(d)

    mean_pair_dist = mean_pair_dist_accum / max(1, mean_pair_count)

    eigs = graph_laplacian_eigs(graph, k=6)

    graph_features = {
        "grid_height": float(h),
        "grid_width": float(w),
        "grid_area": float(h * w),
        "graph_num_free_cells": float(len(free_cells)),
        "graph_num_wall_cells": float(len(walls)),
        "graph_wall_fraction": float(len(walls) / max(1, h * w)),
        "graph_num_edges": float(sum(len(v) for v in graph.values()) / 2.0),
        "graph_num_components": float(len(comps)),
        "graph_largest_component_frac": float(largest / max(1, len(free_cells))),
        "graph_mean_degree": float(np.mean(degrees)),
        "graph_min_degree": float(np.min(degrees)),
        "graph_max_degree": float(np.max(degrees)),
        "graph_num_deadends": float(np.sum(degrees <= 1)),
        "graph_num_corridor_cells": float(np.sum(degrees == 2)),
        "graph_num_branch_cells": float(np.sum(degrees >= 3)),
        "graph_diameter": float(diameter),
        "graph_mean_pair_distance": float(mean_pair_dist),
        "graph_start_reachable_frac": float(start_reachable / max(1, len(free_cells))),
        "graph_start_eccentricity": float(start_ecc),
        "graph_num_bridges": float(bridges),
        "graph_num_articulation_points": float(arts),
    }

    spectral_features = {f"spectral_lambda_{i}": eigs[i] for i in range(len(eigs))}

    src_features = count_ast_nodes(source)
    src_features.update(
        {
            "src_num_lines": float(len(source.splitlines())),
            "src_num_chars": float(len(source)),
        }
    )

    reward_features = infer_reward_features(source, task_name)

    features = {}
    features.update(reward_features)
    features.update(graph_features)
    features.update(spectral_features)
    features.update(src_features)

    metadata = {
        "task_name": task_name,
        "task_file": str(task_file),
        "agent_pos_yx": agent_pos,
        "feature_blocks": {
            "reward": list(reward_features.keys()),
            "graph": list(graph_features.keys()),
            "spectral": list(spectral_features.keys()),
            "programmatic": list(src_features.keys()),
        },
    }

    return features, metadata


@dataclass
class ScoreResult:
    predicted_return: Optional[float]
    predicted_success_prob: Optional[float]
    rtd: Optional[float]
    failure_difficulty: Optional[float]
    total_predictive_difficulty: Optional[float]
    curriculum_utility: Optional[float]
    n_prior_rows_used: int
    n_features_used: int
    lambda_ridge: float
    alpha: float
    beta: float


def load_prior_csv(path: str | Path) -> Any:
    if pd is None:
        raise RuntimeError(
            "pandas is required to read prior CSV files. Install pandas or run without --prior-csv."
        )
    return pd.read_csv(path)


def fit_and_score(
    candidate_features: Dict[str, float],
    prior_csv: str | Path,
    target_policy: Optional[str],
    lambda_ridge: float,
    alpha: float,
    beta: float,
) -> ScoreResult:
    df = load_prior_csv(prior_csv)

    if target_policy and "policy_id" in df.columns:
        df = df[df["policy_id"].astype(str) == str(target_policy)].copy()

    if df.empty:
        return ScoreResult(None, None, None, None, None, None, 0, 0, lambda_ridge, alpha, beta)

    y_col = "success" if "success" in df.columns else "return"

    if y_col not in df.columns:
        raise ValueError("prior CSV must include a 'return' column, optionally a 'success' column")

    feature_names = [k for k in candidate_features.keys() if k in df.columns]

    if not feature_names:
        raise ValueError(
            "No descriptor feature columns in prior CSV matched the candidate descriptor. "
            "Run with --write-prior-template to see the expected feature columns."
        )

    X = df[feature_names].astype(float).to_numpy()
    y = df[y_col].astype(float).to_numpy()
    x = np.array([candidate_features[k] for k in feature_names], dtype=float)

    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma < 1e-8] = 1.0

    Xs = (X - mu) / sigma
    xs = (x - mu) / sigma

    d = Xs.shape[1]
    V = lambda_ridge * np.eye(d) + Xs.T @ Xs
    Vinv = np.linalg.pinv(V)
    rtd = float(math.sqrt(max(0.0, xs.T @ Vinv @ xs)))

    X_aug = np.column_stack([np.ones(Xs.shape[0]), Xs])
    x_aug = np.concatenate([[1.0], xs])

    A = X_aug.T @ X_aug + lambda_ridge * np.eye(d + 1)
    A[0, 0] -= lambda_ridge

    coef = np.linalg.pinv(A) @ X_aug.T @ y
    pred_return = float(np.clip(x_aug @ coef, 0.0, 1.0))

    p_success = pred_return if y_col == "success" else float(np.clip(pred_return, 0.0, 1.0))

    failure = 1.0 - p_success
    total_diff = alpha * failure + beta * rtd
    utility = p_success * (1.0 - p_success) + beta * rtd

    return ScoreResult(
        predicted_return=pred_return,
        predicted_success_prob=p_success,
        rtd=rtd,
        failure_difficulty=float(failure),
        total_predictive_difficulty=float(total_diff),
        curriculum_utility=float(utility),
        n_prior_rows_used=int(X.shape[0]),
        n_features_used=int(d),
        lambda_ridge=lambda_ridge,
        alpha=alpha,
        beta=beta,
    )


def write_prior_template(
    path: str | Path,
    features: Dict[str, float],
    task_name: str,
    target_policy: str,
) -> None:
    fieldnames = ["task_name", "policy_id", "return", "success"] + list(features.keys())

    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        row = {
            "task_name": task_name,
            "policy_id": target_policy,
            "return": "",
            "success": "",
        }
        row.update(features)

        writer.writerow(row)


def write_candidate_feature_csv(
    path: str | Path,
    features: Dict[str, float],
    task_name: str,
) -> None:
    fieldnames = ["task_name"] + list(features.keys())

    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        row = {"task_name": task_name}
        row.update(features)

        writer.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compute PRL/Karel curriculum difficulty descriptors and RTD scores."
    )

    ap.add_argument("--task-file", required=True, help="Path to a Karel task file such as clean_house.py")
    ap.add_argument("--task-name", default="CleanHouse", help="Human-readable task name")
    ap.add_argument("--prior-csv", default=None, help="CSV of prior task descriptor/performance measurements")
    ap.add_argument("--target-policy", default=None, help="Policy/program id to filter prior rows by")
    ap.add_argument("--lambda-ridge", type=float, default=1.0, help="Ridge/RTD regularization lambda")
    ap.add_argument("--alpha", type=float, default=1.0, help="Weight on predicted failure difficulty")
    ap.add_argument("--beta", type=float, default=0.25, help="Weight on RTD novelty in difficulty/utility")
    ap.add_argument("--out", default="karel_difficulty_report.json", help="Output JSON report")
    ap.add_argument("--features-csv", default=None, help="Optional output CSV containing only candidate features")
    ap.add_argument("--write-prior-template", default=None, help="Write a prior CSV template and exit")

    args = ap.parse_args()

    features, metadata = cleanhouse_descriptor(args.task_file, args.task_name)

    if args.features_csv:
        write_candidate_feature_csv(args.features_csv, features, args.task_name)

    if args.write_prior_template:
        write_prior_template(
            args.write_prior_template,
            features,
            args.task_name,
            args.target_policy or "current_policy",
        )
        print(f"Wrote prior-measurement template to {args.write_prior_template}")
        return

    if args.prior_csv:
        scores = fit_and_score(
            features,
            args.prior_csv,
            args.target_policy,
            args.lambda_ridge,
            args.alpha,
            args.beta,
        )
    else:
        scores = ScoreResult(
            None,
            None,
            None,
            None,
            None,
            None,
            0,
            0,
            args.lambda_ridge,
            args.alpha,
            args.beta,
        )

    report = {
        "metadata": metadata,
        "features": features,
        "scores": asdict(scores),
        "interpretation": {
            "rtd": "Large means CleanHouse is poorly covered by prior task descriptors; this is novelty/uncertainty, not pure difficulty.",
            "predicted_success_prob": "Estimated from prior rows for the same policy_id when --prior-csv is provided.",
            "curriculum_utility": "p_success*(1-p_success) + beta*RTD; high values mean competence-boundary plus novelty.",
        },
    }

    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report["scores"], indent=2))
    print(f"Wrote report to {args.out}")


if __name__ == "__main__":
    main()
