from __future__ import annotations

import os
from typing import Any

from .prompts import EXECUTIVE_REPORT_PROMPT, MITIGATION_PLAN_PROMPT, PLAIN_LANGUAGE_PROMPT, TECHNICAL_SUMMARY_PROMPT


def _risk_label(risk_level: str) -> str:
    return {"LOW": "baixo", "MEDIUM": "medio", "HIGH": "alto"}.get(risk_level, risk_level.lower())


def _recommendations(risk_level: str) -> list[str]:
    if risk_level == "HIGH":
        return [
            "Ativar monitoramento reforcado de satelites e enlaces criticos.",
            "Preparar janela de contingencia para cargas sensiveis e comunicacoes.",
            "Emitir alerta operacional para equipes de infraestrutura orbital.",
        ]
    if risk_level == "MEDIUM":
        return [
            "Aumentar frequencia de verificacao por 24 horas.",
            "Manter equipes informadas sobre possivel escalada.",
            "Validar disponibilidade de relatorios e canais de alerta.",
        ]
    return [
        "Manter monitoramento padrao.",
        "Registrar a analise como baseline operacional.",
        "Reavaliar ao receber nova imagem solar.",
    ]


def generate_sections(analysis: dict[str, Any]) -> dict[str, str]:
    risk = analysis["risk_level"]
    score = analysis["score"]
    confidence = analysis["confidence"]
    risk_pt = _risk_label(risk)
    recs = _recommendations(risk)

    return {
        "technical_summary": (
            f"O Hermes SolarShield classificou a janela de 24 horas com risco {risk_pt}. "
            f"O score de atividade foi {score:.3f}, com confianca {confidence:.3f}. "
            "A leitura combina brilho, contraste, densidade de bordas e concentracao de regioes ativas, "
            "servindo como triagem operacional para priorizar observacao solar e continuidade de missao."
        ),
        "plain_summary": (
            f"A chance de uma erupcao solar relevante foi avaliada como risco {risk_pt}. "
            "Isso nao significa certeza de evento, mas indica o quanto a imagem solar atual merece atencao."
        ),
        "mitigation_plan": " ".join(recs),
        "executive_report": (
            f"Status Hermes SolarShield: risco {risk_pt} para as proximas 24 horas. "
            f"Score {score:.3f}; confianca {confidence:.3f}. Acao recomendada: {recs[0]}"
        ),
    }


def answer_question(question: str, analysis: dict[str, Any] | None) -> dict[str, str]:
    if not analysis:
        return {
            "answer": "Ainda nao ha uma previsao ativa. Execute uma analise de imagem solar para eu contextualizar a resposta.",
            "technical_summary": "Sem analise disponivel.",
            "plain_summary": "Envie ou selecione uma imagem solar primeiro.",
            "mitigation_plan": "Aguardando resultado do modelo.",
        }

    sections = generate_sections(analysis)
    api_key = os.getenv("GENAI_API_KEY")
    provider = "template"
    if api_key:
        provider = "external-ready"

    answer = (
        f"Pergunta recebida: {question}\n\n"
        f"{sections['plain_summary']}\n\n"
        f"Resumo tecnico: {sections['technical_summary']}\n\n"
        f"Plano de mitigacao: {sections['mitigation_plan']}"
    )
    return {**sections, "answer": answer, "provider": provider}


def prompt_catalog() -> dict[str, str]:
    return {
        "technical_summary": TECHNICAL_SUMMARY_PROMPT,
        "plain_language": PLAIN_LANGUAGE_PROMPT,
        "mitigation_plan": MITIGATION_PLAN_PROMPT,
        "executive_report": EXECUTIVE_REPORT_PROMPT,
    }
