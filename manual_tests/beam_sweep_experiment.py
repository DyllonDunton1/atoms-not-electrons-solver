"""Benchmark full-horizon search breadth and generate a self-contained report.

This is intentionally a *joint* beam/candidate-width sweep rather than a pure
beam-width ablation.  For every requested width W, the experiment runs the
scheduled solver with ``beam_width=W`` and ``candidate_width=W`` on the same
prefix of BIG_ORDER, validates it through the existing experiment runner, then
collects the generated planner/legacy metrics.

Outputs include CSV and JSON for analysis, Markdown for quick reading, a
self-contained HTML report with SVG charts, and one log per solver run.
"""

from __future__ import annotations

from argparse import ArgumentParser
import csv
import html
import json
from pathlib import Path
import subprocess
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "manual_tests" / "scheduled_solver_experiment.py"
DEFAULT_WIDTHS = (2, 4, 8, 12, 16, 20, 24)


def _output_stem(
    *,
    beam_width: int,
    candidate_width: int,
    candidate_cap: int,
    padding: int,
    robots: int,
    orders: int,
) -> str:
    return (
        f"scheduled_v1_full_horizon_beam{beam_width}_cand{candidate_width}_"
        f"cap{candidate_cap}_pad{padding}_{robots}r_{orders}o"
    )


def _run_one(command: Sequence[str], log_path: Path) -> int:
    """Run one benchmark while teeing combined stdout/stderr to console + log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            list(command),
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()
        return process.wait()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _collect_result(
    *,
    width: int,
    legacy_metrics_path: Path,
    planner_metrics_path: Path,
) -> Dict[str, object]:
    legacy = _read_json(legacy_metrics_path)
    planner_metrics = _read_json(planner_metrics_path)
    planner = planner_metrics["planner"]
    action_counts = legacy["action_counts"]
    movement = legacy["movement"]

    return {
        "status": "ok",
        "beam_width": width,
        "candidate_width": width,
        "makespan": int(legacy["end_timestep"]),
        "planning_seconds": float(planner["planning_seconds"]),
        "astar_seconds": float(planner["astar_seconds"]),
        "astar_calls": int(planner["astar_calls"]),
        "astar_expansions": int(planner["astar_expansions"]),
        "astar_capped_calls": int(planner["astar_capped_calls"]),
        "beam_expansions": int(planner["beam_expansions"]),
        "beam_generated": int(planner["beam_generated"]),
        "beam_pruned": int(planner["beam_pruned"]),
        "candidate_seconds": float(planner["candidate_seconds"]),
        "candidate_expansions_skipped": int(planner["candidate_expansions_skipped"]),
        "capped_candidate_rejections": int(planner["capped_candidate_rejections"]),
        "candidate_full_budget_rescues": int(planner["candidate_full_budget_rescues"]),
        "order_full_budget_rescues": int(planner["order_full_budget_rescues"]),
        "row_fast_path_hits": int(planner["row_fast_path_hits"]),
        "point_fast_path_hits": int(planner["point_fast_path_hits"]),
        "wait_timesteps": int(legacy["wait_timesteps"]),
        "explicit_actions": int(legacy["explicit_actions"]),
        "moves": int(action_counts["move"]),
        "picks": int(action_counts["pick"]),
        "collection_moves": int(movement["collection"]),
        "refill_moves": int(movement["refill"]),
        "fulfillment_moves": int(movement["fulfillment"]),
        "refill_trips": int(legacy["refill_trips"]),
        "aisle_visits": int(legacy["aisle_visits"]),
        "aisle_reentries": int(legacy["aisle_reentries"]),
    }


def add_derived_metrics(results: List[Dict[str, object]]) -> None:
    """Add quality/runtime comparisons and Pareto-frontier labels in place."""
    valid = [result for result in results if result.get("status") == "ok"]
    if not valid:
        return
    valid.sort(key=lambda result: int(result["beam_width"]))
    baseline = valid[0]
    baseline_makespan = int(baseline["makespan"])
    baseline_seconds = float(baseline["planning_seconds"])

    previous: Optional[Dict[str, object]] = None
    for result in valid:
        makespan = int(result["makespan"])
        seconds = float(result["planning_seconds"])
        result["timesteps_saved_vs_baseline"] = baseline_makespan - makespan
        result["makespan_improvement_pct_vs_baseline"] = (
            0.0 if baseline_makespan == 0 else 100.0 * (baseline_makespan - makespan) / baseline_makespan
        )
        result["runtime_multiplier_vs_baseline"] = (
            0.0 if baseline_seconds == 0 else seconds / baseline_seconds
        )
        if previous is None:
            result["timesteps_saved_vs_previous"] = 0
            result["makespan_improvement_pct_vs_previous"] = 0.0
            result["extra_seconds_vs_previous"] = 0.0
            result["timesteps_saved_per_extra_second"] = None
        else:
            previous_makespan = int(previous["makespan"])
            previous_seconds = float(previous["planning_seconds"])
            saved = previous_makespan - makespan
            extra_seconds = seconds - previous_seconds
            result["timesteps_saved_vs_previous"] = saved
            result["makespan_improvement_pct_vs_previous"] = (
                0.0
                if previous_makespan == 0
                else 100.0 * saved / previous_makespan
            )
            result["extra_seconds_vs_previous"] = extra_seconds
            result["timesteps_saved_per_extra_second"] = (
                None if extra_seconds <= 0 else saved / extra_seconds
            )
        previous = result

    # Pareto-optimal means no completed run is both at least as fast to plan and
    # at least as good in makespan, with one of those inequalities strict.
    for result in valid:
        rt = float(result["planning_seconds"])
        ms = int(result["makespan"])
        dominated = any(
            other is not result
            and float(other["planning_seconds"]) <= rt
            and int(other["makespan"]) <= ms
            and (
                float(other["planning_seconds"]) < rt
                or int(other["makespan"]) < ms
            )
            for other in valid
        )
        result["pareto_optimal"] = not dominated


def _fmt_number(value: object, decimals: int = 0) -> str:
    if value is None:
        return "—"
    if decimals:
        return f"{float(value):,.{decimals}f}"
    return f"{int(value):,}"


def _summary_rows(results: Iterable[Dict[str, object]]) -> List[List[str]]:
    rows: List[List[str]] = []
    for result in sorted(results, key=lambda r: int(r["beam_width"])):
        if result.get("status") != "ok":
            rows.append([
                str(result["beam_width"]),
                "FAILED",
                "—",
                "—",
                "—",
                "—",
                "—",
            ])
            continue
        rows.append([
            str(result["beam_width"]),
            _fmt_number(result["makespan"]),
            _fmt_number(result["planning_seconds"], 2),
            f"{float(result['makespan_improvement_pct_vs_baseline']):.2f}%",
            f"{float(result['runtime_multiplier_vs_baseline']):.2f}×",
            _fmt_number(result["astar_expansions"]),
            "yes" if result.get("pareto_optimal") else "no",
        ])
    return rows


def render_markdown(
    results: List[Dict[str, object]],
    *,
    orders: int,
    robots: int,
    candidate_cap: int,
) -> str:
    valid = [result for result in results if result.get("status") == "ok"]
    lines = [
        "# Full-Horizon Search Breadth Sweep",
        "",
        f"Same first **{orders} orders**, **{robots} robots**, candidate A* cap **{candidate_cap:,}**.",
        "",
        "> This is a joint search-breadth experiment: candidate width is set equal to beam width. "
        "It demonstrates the quality/compute tradeoff of increasing both breadth knobs together; "
        "it is not a pure beam-width ablation.",
        "",
        "| Beam = candidates | Makespan | Planning time (s) | Improvement vs baseline | Runtime vs baseline | A* expansions | Pareto |",
        "|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in _summary_rows(results):
        lines.append("| " + " | ".join(row) + " |")

    lines.extend(["", "## Interpretation", ""])
    if valid:
        baseline = min(valid, key=lambda r: int(r["beam_width"]))
        best = min(valid, key=lambda r: int(r["makespan"]))
        fastest = min(valid, key=lambda r: float(r["planning_seconds"]))
        improvement = float(best["makespan_improvement_pct_vs_baseline"])
        runtime_mult = float(best["runtime_multiplier_vs_baseline"])
        lines.extend([
            f"- Baseline: **{int(baseline['beam_width'])}/{int(baseline['candidate_width'])}**, "
            f"makespan **{int(baseline['makespan']):,}**, planning **{float(baseline['planning_seconds']):.2f}s**.",
            f"- Best schedule: **{int(best['beam_width'])}/{int(best['candidate_width'])}**, "
            f"makespan **{int(best['makespan']):,}** — **{improvement:.2f}%** better than the baseline at "
            f"**{runtime_mult:.2f}×** baseline planning time.",
            f"- Fastest completed run: **{int(fastest['beam_width'])}/{int(fastest['candidate_width'])}** at "
            f"**{float(fastest['planning_seconds']):.2f}s**.",
        ])
        pareto = [r for r in valid if r.get("pareto_optimal")]
        lines.append(
            "- Pareto-optimal completed widths: "
            + ", ".join(f"**{int(r['beam_width'])}**" for r in pareto)
            + "."
        )

        # Find the largest marginal gain and first clear diminishing-return step.
        ordered = sorted(valid, key=lambda r: int(r["beam_width"]))
        marginal = ordered[1:]
        positive = [r for r in marginal if int(r["timesteps_saved_vs_previous"]) > 0]
        if positive:
            strongest = max(positive, key=lambda r: int(r["timesteps_saved_vs_previous"]))
            lines.append(
                f"- Largest single-step quality gain: moving to width **{int(strongest['beam_width'])}** saved "
                f"**{int(strongest['timesteps_saved_vs_previous']):,} timesteps** versus the previous tested width."
            )
    else:
        lines.append("No run completed successfully.")

    lines.extend([
        "",
        "## Interview-safe conclusion",
        "",
        "Use this sweep to defend **the search mechanism**, not to claim that the full-horizon architecture beats the submitted reactive solver. "
        "If larger widths reduce makespan, the experiment shows that preserving more candidate/order-route alternatives improves the scheduler's "
        "solution quality. The full 1,000-order benchmark is a separate architectural comparison.",
        "",
    ])
    return "\n".join(lines)


def _svg_line_chart(
    points: Sequence[Tuple[float, float, str]],
    *,
    title: str,
    x_label: str,
    y_label: str,
    lower_is_better: bool = False,
    width: int = 760,
    height: int = 300,
) -> str:
    if not points:
        return "<p>No completed data.</p>"
    margin_left, margin_right, margin_top, margin_bottom = 72, 24, 42, 54
    chart_w = width - margin_left - margin_right
    chart_h = height - margin_top - margin_bottom
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_min == x_max:
        x_min -= 1
        x_max += 1
    if y_min == y_max:
        y_min -= 1
        y_max += 1
    y_pad = max((y_max - y_min) * 0.08, 1.0)
    y_min -= y_pad
    y_max += y_pad

    def sx(x: float) -> float:
        return margin_left + (x - x_min) * chart_w / (x_max - x_min)

    def sy(y: float) -> float:
        return margin_top + (y_max - y) * chart_h / (y_max - y_min)

    coords = [(sx(x), sy(y), label, x, y) for x, y, label in points]
    polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y, _, _, _ in coords)
    subtitle = "lower is better" if lower_is_better else ""
    out = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
        f'<text x="{width/2}" y="20" text-anchor="middle" class="chart-title">{html.escape(title)}</text>',
        f'<text x="{width/2}" y="36" text-anchor="middle" class="chart-subtitle">{html.escape(subtitle)}</text>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height-margin_bottom}" class="axis"/>',
        f'<line x1="{margin_left}" y1="{height-margin_bottom}" x2="{width-margin_right}" y2="{height-margin_bottom}" class="axis"/>',
        f'<polyline points="{polyline}" class="series"/>',
    ]
    for px, py, label, x, y in coords:
        out.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4.5" class="point"><title>{html.escape(label)}: {y:,.2f}</title></circle>')
        out.append(f'<text x="{px:.1f}" y="{height-margin_bottom+18}" text-anchor="middle" class="tick">{x:g}</text>')
        out.append(f'<text x="{px:.1f}" y="{py-9:.1f}" text-anchor="middle" class="value">{y:,.0f}</text>')
    out.extend([
        f'<text x="{width/2}" y="{height-8}" text-anchor="middle" class="label">{html.escape(x_label)}</text>',
        f'<text transform="translate(17 {height/2}) rotate(-90)" text-anchor="middle" class="label">{html.escape(y_label)}</text>',
        "</svg>",
    ])
    return "".join(out)


def render_html(
    results: List[Dict[str, object]],
    *,
    orders: int,
    robots: int,
    candidate_cap: int,
) -> str:
    valid = sorted(
        [result for result in results if result.get("status") == "ok"],
        key=lambda r: int(r["beam_width"]),
    )
    table_rows = []
    for row in _summary_rows(results):
        table_rows.append("<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>")

    quality_points = [
        (float(r["beam_width"]), float(r["makespan"]), f"width {r['beam_width']}")
        for r in valid
    ]
    runtime_points = [
        (float(r["beam_width"]), float(r["planning_seconds"]), f"width {r['beam_width']}")
        for r in valid
    ]
    tradeoff_points = [
        (float(r["planning_seconds"]), float(r["makespan"]), f"width {r['beam_width']}")
        for r in valid
    ]

    interpretation = render_markdown(
        results,
        orders=orders,
        robots=robots,
        candidate_cap=candidate_cap,
    )
    interpretation_lines = []
    in_interpretation = False
    for line in interpretation.splitlines():
        if line == "## Interpretation":
            in_interpretation = True
            continue
        if line.startswith("## ") and in_interpretation:
            break
        if in_interpretation and line.startswith("- "):
            interpretation_lines.append(f"<li>{html.escape(line[2:])}</li>")

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Full-Horizon Search Breadth Sweep</title>
<style>
:root{{--ink:#172033;--muted:#667085;--panel:#fff;--line:#d9e2ec;--accent:#2463eb;--bg:#f5f7fb;--good:#0a7a42}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font-family:system-ui,-apple-system,Segoe UI,sans-serif}}
main{{max-width:1120px;margin:0 auto;padding:34px 22px 60px}} h1{{margin:0 0 6px;font-size:32px}} h2{{margin-top:32px}}
.sub{{color:var(--muted);margin:0 0 24px}} .note{{background:#eef4ff;border-left:4px solid var(--accent);padding:14px 16px;border-radius:6px;margin:20px 0}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px;margin:16px 0;box-shadow:0 2px 8px rgba(20,30,50,.04)}}
table{{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}} th,td{{padding:10px 8px;border-bottom:1px solid var(--line);text-align:right}} th:first-child,td:first-child{{text-align:left}} th{{color:var(--muted);font-size:13px}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}} svg{{width:100%;height:auto}} .axis{{stroke:#98a2b3;stroke-width:1}} .series{{fill:none;stroke:var(--accent);stroke-width:2.5}} .point{{fill:var(--accent)}} .chart-title{{font-size:15px;font-weight:700;fill:var(--ink)}} .chart-subtitle,.tick,.label{{font-size:11px;fill:var(--muted)}} .value{{font-size:10px;fill:var(--ink)}}
.badge{{display:inline-block;padding:3px 8px;border-radius:999px;background:#e9f8ef;color:var(--good);font-weight:700;font-size:12px}} li{{margin:7px 0}} code{{background:#eef1f5;padding:2px 5px;border-radius:4px}}
@media(max-width:780px){{.grid{{grid-template-columns:1fr}} table{{font-size:12px}}}}
</style>
</head>
<body><main>
<h1>Full-Horizon Search Breadth Sweep</h1>
<p class="sub">{orders} orders · {robots} robots · candidate A* cap {candidate_cap:,} · beam width = candidate width</p>
<div class="note"><strong>Experiment meaning.</strong> This is a joint search-breadth sweep, not a pure beam-width ablation. Each larger setting both preserves more beam states and exposes more candidate children per state.</div>
<div class="card"><h2>Results</h2><table><thead><tr><th>Beam = candidates</th><th>Makespan</th><th>Planning time (s)</th><th>Improvement vs baseline</th><th>Runtime vs baseline</th><th>A* expansions</th><th>Pareto</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table></div>
<div class="grid">
<div class="card">{_svg_line_chart(quality_points,title='Schedule Quality vs Search Breadth',x_label='Beam = candidate width',y_label='Makespan',lower_is_better=True)}</div>
<div class="card">{_svg_line_chart(runtime_points,title='Planning Cost vs Search Breadth',x_label='Beam = candidate width',y_label='Planning seconds')}</div>
</div>
<div class="card">{_svg_line_chart(tradeoff_points,title='Quality / Compute Tradeoff',x_label='Planning seconds',y_label='Makespan',lower_is_better=True)}</div>
<div class="card"><h2>Interpretation</h2><ul>{''.join(interpretation_lines)}</ul><p><span class="badge">Interview framing</span> Larger breadth improving this 50-order schedule demonstrates that beam/candidate search is effective inside the full-horizon formulation. It does <em>not</em> by itself prove that the full-horizon architecture beats the submitted reactive solver on all 1,000 orders.</p></div>
</main></body></html>"""


def _write_csv(results: List[Dict[str, object]], path: Path) -> None:
    keys: List[str] = []
    for result in results:
        for key in result:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)


def write_reports(
    results: List[Dict[str, object]],
    output_dir: Path,
    *,
    orders: int,
    robots: int,
    candidate_cap: int,
    widths: Sequence[int],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    add_derived_metrics(results)
    payload = {
        "experiment": {
            "orders": orders,
            "robots": robots,
            "widths": list(widths),
            "beam_equals_candidate_width": True,
            "candidate_max_path_expansions": candidate_cap,
        },
        "results": results,
    }
    (output_dir / "beam_sweep_results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(results, output_dir / "beam_sweep_results.csv")
    (output_dir / "beam_sweep_report.md").write_text(
        render_markdown(
            results,
            orders=orders,
            robots=robots,
            candidate_cap=candidate_cap,
        ),
        encoding="utf-8",
    )
    (output_dir / "beam_sweep_report.html").write_text(
        render_html(
            results,
            orders=orders,
            robots=robots,
            candidate_cap=candidate_cap,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = ArgumentParser(
        description="Sweep beam/candidate breadth and generate CSV/JSON/Markdown/HTML reports."
    )
    parser.add_argument("--orders", type=int, default=50, choices=range(1, 1001))
    parser.add_argument("--robots", type=int, default=5, choices=range(1, 6))
    parser.add_argument("--widths", type=int, nargs="+", default=list(DEFAULT_WIDTHS))
    parser.add_argument("--padding", type=int, default=1)
    parser.add_argument("--path-horizon", type=int, default=512)
    parser.add_argument("--max-path-expansions", type=int, default=250_000)
    parser.add_argument("--candidate-max-path-expansions", type=int, default=30_000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: outputs/beam_sweep_<orders>o",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse existing metrics for a width instead of rerunning it.",
    )
    args = parser.parse_args()

    if any(width <= 0 for width in args.widths):
        parser.error("every --widths value must be positive")
    if len(set(args.widths)) != len(args.widths):
        parser.error("--widths must not contain duplicates")
    if args.padding < 0:
        parser.error("--padding must be nonnegative")
    if args.max_path_expansions <= 0 or args.candidate_max_path_expansions <= 0:
        parser.error("A* expansion caps must be positive")

    widths = sorted(args.widths)
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else REPO_ROOT / "outputs" / f"beam_sweep_{args.orders}o"
    )
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, object]] = []
    interrupted = False
    for width in widths:
        stem = _output_stem(
            beam_width=width,
            candidate_width=width,
            candidate_cap=args.candidate_max_path_expansions,
            padding=args.padding,
            robots=args.robots,
            orders=args.orders,
        )
        legacy_metrics_path = REPO_ROOT / "outputs" / f"{stem}_metrics.json"
        planner_metrics_path = REPO_ROOT / "outputs" / f"{stem}_planner_metrics.json"
        log_path = output_dir / f"beam{width}_cand{width}.log"

        print("\n" + "=" * 78)
        print(f"BEAM={width}  CANDIDATES={width}  ORDERS={args.orders}")
        print("=" * 78, flush=True)

        should_run = not (
            args.reuse_existing
            and legacy_metrics_path.exists()
            and planner_metrics_path.exists()
        )
        if should_run:
            command = [
                sys.executable,
                str(RUNNER),
                "--orders", str(args.orders),
                "--robots", str(args.robots),
                "--beam-width", str(width),
                "--candidate-width", str(width),
                "--padding", str(args.padding),
                "--path-horizon", str(args.path_horizon),
                "--max-path-expansions", str(args.max_path_expansions),
                "--candidate-max-path-expansions", str(args.candidate_max_path_expansions),
            ]
            try:
                return_code = _run_one(command, log_path)
            except KeyboardInterrupt:
                print("\nSweep interrupted; writing report for completed runs...", flush=True)
                interrupted = True
                break
            if return_code != 0:
                results.append({
                    "status": "failed",
                    "beam_width": width,
                    "candidate_width": width,
                    "return_code": return_code,
                    "log": str(log_path.relative_to(REPO_ROOT)),
                })
                write_reports(
                    results,
                    output_dir,
                    orders=args.orders,
                    robots=args.robots,
                    candidate_cap=args.candidate_max_path_expansions,
                    widths=widths,
                )
                print(f"Run {width}/{width} failed with exit code {return_code}; continuing.")
                continue
        else:
            print("Reusing existing completed metrics.", flush=True)

        if not legacy_metrics_path.exists() or not planner_metrics_path.exists():
            results.append({
                "status": "failed",
                "beam_width": width,
                "candidate_width": width,
                "error": "expected metrics files were not produced",
                "log": str(log_path.relative_to(REPO_ROOT)) if log_path.exists() else "",
            })
            continue

        result = _collect_result(
            width=width,
            legacy_metrics_path=legacy_metrics_path,
            planner_metrics_path=planner_metrics_path,
        )
        result["log"] = str(log_path.relative_to(REPO_ROOT)) if log_path.exists() else "reused existing metrics"
        results.append(result)

        # Rewrite after every completed run so an interrupted long sweep still
        # leaves a usable partial report.
        write_reports(
            results,
            output_dir,
            orders=args.orders,
            robots=args.robots,
            candidate_cap=args.candidate_max_path_expansions,
            widths=widths,
        )
        print(
            f"Completed {width}/{width}: makespan={result['makespan']:,}, "
            f"planning={result['planning_seconds']:.2f}s",
            flush=True,
        )

    write_reports(
        results,
        output_dir,
        orders=args.orders,
        robots=args.robots,
        candidate_cap=args.candidate_max_path_expansions,
        widths=widths,
    )

    valid = [result for result in results if result.get("status") == "ok"]
    print("\n" + "=" * 78)
    print("SWEEP SUMMARY")
    print("=" * 78)
    for row in _summary_rows(results):
        print(
            f"width={row[0]:>2s}  makespan={row[1]:>8s}  "
            f"plan={row[2]:>9s}s  improve={row[3]:>8s}  "
            f"runtime={row[4]:>7s}  pareto={row[6]}"
        )
    if valid:
        best = min(valid, key=lambda result: int(result["makespan"]))
        print(
            f"Best completed schedule: {best['beam_width']}/{best['candidate_width']} "
            f"at {best['makespan']:,} timesteps."
        )
    print(f"HTML report: {output_dir / 'beam_sweep_report.html'}")
    print(f"Markdown:    {output_dir / 'beam_sweep_report.md'}")
    print(f"CSV:         {output_dir / 'beam_sweep_results.csv'}")
    print(f"JSON:        {output_dir / 'beam_sweep_results.json'}")
    if interrupted:
        raise SystemExit(130)


if __name__ == "__main__":
    main()
