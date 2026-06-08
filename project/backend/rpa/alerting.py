from __future__ import annotations

import csv
from email.message import EmailMessage
from pathlib import Path
from typing import Any

from database.db import utc_now

BASE_DIR = Path(__file__).resolve().parents[1]
ALERT_DIR = BASE_DIR / "reports" / "operational_alerts"
ALERT_CSV = ALERT_DIR / "shared_ops_alerts.csv"


def record_operational_alert(analysis: dict[str, Any], report: dict[str, Any] | None = None) -> dict[str, str]:
    ALERT_DIR.mkdir(parents=True, exist_ok=True)
    created_at = utc_now()
    row = {
        "created_at": created_at,
        "analysis_id": str(analysis.get("id", "")),
        "risk_level": str(analysis.get("risk_level", "")),
        "score": f"{float(analysis.get('score', 0.0)):.3f}",
        "confidence": f"{float(analysis.get('confidence', 0.0)):.3f}",
        "source": str(analysis.get("source", "")),
        "report_id": str((report or {}).get("id", "")),
    }
    write_header = not ALERT_CSV.exists()
    with ALERT_CSV.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    message = EmailMessage()
    message["From"] = "hermes-solarshield@local"
    message["To"] = "orbital-ops-team@example.com"
    message["Subject"] = f"Hermes SolarShield alert - {row['risk_level']} risk"
    message.set_content(
        "\n".join(
            [
                "Hermes SolarShield operational alert",
                "",
                f"Analysis ID: {row['analysis_id']}",
                f"Risk level: {row['risk_level']}",
                f"Score: {row['score']}",
                f"Confidence: {row['confidence']}",
                f"Source: {row['source']}",
                f"Report ID: {row['report_id']}",
                "",
                "Recommended action: review the generated report and apply the mitigation plan.",
            ]
        )
    )
    email_path = ALERT_DIR / f"alert_analysis_{row['analysis_id']}_{created_at.replace(':', '-')}.eml"
    email_path.write_text(message.as_string(), encoding="utf-8")
    return {"csv_path": str(ALERT_CSV), "email_path": str(email_path)}
