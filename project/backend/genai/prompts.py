SYSTEM_CONTEXT = """
Voce e o Hermes Copilot, um assistente tecnico para clima espacial, infraestrutura orbital,
seguranca operacional e resposta a solar flares.
"""

TECHNICAL_SUMMARY_PROMPT = """
Gere um resumo tecnico para engenharia com base no risco {risk_level}, score {score}
e confianca {confidence}. Foque em telemetria, continuidade de missao e monitoramento.
"""

PLAIN_LANGUAGE_PROMPT = """
Explique para uma pessoa nao tecnica o que significa risco {risk_level} de solar flare
nas proximas 24 horas.
"""

MITIGATION_PLAN_PROMPT = """
Crie um plano de mitigacao operacional para risco {risk_level}, incluindo comunicacao,
infraestrutura orbital, sistemas terrestres e escalonamento.
"""

EXECUTIVE_REPORT_PROMPT = """
Crie um relatorio executivo curto para decisores sobre o risco {risk_level}, score {score}
e recomendacoes para as proximas 24 horas.
"""

