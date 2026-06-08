from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from model import load_model, predict_flare_risk


def _percent(value: float) -> float:
    return round(float(value) * 100, 2)


def evaluate(manifest_path: Path, limit: int | None = None, wayback_only: bool = False) -> dict:
    rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    if wayback_only:
        rows = [row for row in rows if "web.archive.org" in row.get("snapshot_url", "")]
    if limit:
        rows = rows[:limit]

    model = load_model()
    if model is None:
        raise RuntimeError("Modelo nao carregado. Use HERMES_MODEL_MODE=forecast e confira o checkpoint.")

    examples = []
    absolute_errors = {"C": [], "M": [], "X": []}
    within = {5: {"C": 0, "M": 0, "X": 0}, 10: {"C": 0, "M": 0, "X": 0}, 15: {"C": 0, "M": 0, "X": 0}}

    for row in rows:
        image_path = Path(row["path"])
        if not image_path.exists():
            continue
        result = predict_flare_risk(image_path.read_bytes(), model)
        predicted = result.get("flare_probabilities") or {}
        actual = row["probabilities"]
        item_errors = {}
        for flare_class in ("C", "M", "X"):
            pred_pct = _percent(predicted.get(flare_class, 0.0))
            actual_pct = float(actual[flare_class])
            error = abs(pred_pct - actual_pct)
            absolute_errors[flare_class].append(error)
            item_errors[flare_class] = round(error, 2)
            for tolerance in within:
                if error <= tolerance:
                    within[tolerance][flare_class] += 1
        examples.append(
            {
                "date": row["date"],
                "image": image_path.name,
                "actual": actual,
                "predicted": {key: _percent(value) for key, value in predicted.items()},
                "absolute_error": item_errors,
            }
        )

    count = len(examples)
    if count == 0:
        raise RuntimeError("Nenhuma imagem valida encontrada para avaliacao.")

    per_class_mae = {key: round(mean(values), 2) for key, values in absolute_errors.items()}
    all_errors = [value for values in absolute_errors.values() for value in values]
    summary = {
        "samples": count,
        "model_architecture": getattr(model, "hermes_metadata", {}).get("architecture"),
        "mae_points": {
            "general": round(mean(all_errors), 2),
            **per_class_mae,
        },
        "within_tolerance_percent": {
            f"+/-{tolerance}_points": {
                key: round((value / count) * 100, 2)
                for key, value in counts.items()
            }
            for tolerance, counts in within.items()
        },
        "worst_examples": sorted(
            examples,
            key=lambda item: mean(item["absolute_error"].values()),
            reverse=True,
        )[:10],
        "best_examples": sorted(
            examples,
            key=lambda item: mean(item["absolute_error"].values()),
        )[:10],
    }
    return {"summary": summary, "examples": examples}


def main() -> None:
    parser = argparse.ArgumentParser(description="Avalia forecast C/M/X contra historico Wayback SpaceWeatherLive.")
    parser.add_argument("--manifest", default="data/wayback_swl_0131/manifest.json", type=Path)
    parser.add_argument("--output", default="reports/wayback_forecast_evaluation.json", type=Path)
    parser.add_argument("--report", default="reports/wayback_forecast_evaluation.md", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--wayback-only", action="store_true")
    args = parser.parse_args()

    result = evaluate(args.manifest, args.limit, args.wayback_only)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = result["summary"]
    report = [
        "# Hermes SolarShield - Avaliacao historica C/M/X",
        "",
        f"Amostras avaliadas: {summary['samples']}",
        f"Arquitetura ativa: {summary['model_architecture']}",
        "",
        "## Margem de erro media",
        "",
        "| Classe | MAE |",
        "|---|---:|",
        f"| Geral | {summary['mae_points']['general']} p.p. |",
        f"| C | {summary['mae_points']['C']} p.p. |",
        f"| M | {summary['mae_points']['M']} p.p. |",
        f"| X | {summary['mae_points']['X']} p.p. |",
        "",
        "## Percentual dentro da margem",
        "",
        "| Margem | C | M | X |",
        "|---|---:|---:|---:|",
    ]
    for tolerance, values in summary["within_tolerance_percent"].items():
        report.append(f"| {tolerance.replace('_', ' ')} | {values['C']}% | {values['M']}% | {values['X']}% |")
    report.extend(
        [
            "",
            "## Piores exemplos",
            "",
            "| Data | Imagem | Real C/M/X | Hermes C/M/X | Erro C/M/X |",
            "|---|---|---|---|---|",
        ]
    )
    for item in summary["worst_examples"]:
        report.append(
            "| {date} | `{image}` | {a[C]}/{a[M]}/{a[X]} | {p[C]}/{p[M]}/{p[X]} | {e[C]}/{e[M]}/{e[X]} |".format(
                date=item["date"],
                image=item["image"],
                a=item["actual"],
                p=item["predicted"],
                e=item["absolute_error"],
            )
        )
    report.extend(
        [
            "",
            "## Melhores exemplos",
            "",
            "| Data | Imagem | Real C/M/X | Hermes C/M/X | Erro C/M/X |",
            "|---|---|---|---|---|",
        ]
    )
    for item in summary["best_examples"]:
        report.append(
            "| {date} | `{image}` | {a[C]}/{a[M]}/{a[X]} | {p[C]}/{p[M]}/{p[X]} | {e[C]}/{e[M]}/{e[X]} |".format(
                date=item["date"],
                image=item["image"],
                a=item["actual"],
                p=item["predicted"],
                e=item["absolute_error"],
            )
        )
    args.report.write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
