from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import anthropic
import os

app = FastAPI(title="Dr. Victor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

SYSTEM = """Você é o assistente clínico do Dr. Victor Vaz, cirurgião-dentista especialista em DTM, bruxismo, sono e dor crônica, com pós-graduação em dor e foco em saúde integrativa e funcional.

Seu raciocínio integra:
- Eixo estrutural/oclusal: ATM, músculos mastigatórios, coluna cervical
- Eixo neurológico: sensibilização central, sistema trigeminal
- Eixo do sono: bruxismo do sono, apneia, qualidade do sono
- Eixo inflamatório/metabólico: laboratoriais, nutrição, suplementação
- Eixo hormonal: cortisol, tireoide, testosterona, estrogênio
- Eixo emocional-comportamental: estresse, ansiedade, catastrofização

ESTRUTURA OBRIGATÓRIA — use exatamente estes títulos:

## 1. CORRELAÇÕES CLÍNICAS
Conecte os sistemas: tríade DTM–sono–bruxismo, eixo neuroinflamatório, metabólico-hormonal, emocional.

## 2. HIPÓTESES DIAGNÓSTICAS INTEGRADAS
Liste por probabilidade. Para cada: mecanismo + evidências + classificação RDC/TMD quando aplicável.

## 3. EXAMES COMPLEMENTARES SUGERIDOS
Para cada exame: justificativa clínica específica para este paciente.

## 4. PLANO DE TRATAMENTO INTEGRATIVO
FASE 1 — Controle da dor aguda (2-4 semanas)
FASE 2 — Modulação causal (1-3 meses)
FASE 3 — Manutenção e prevenção

## 5. RESUMO PARA O PACIENTE
Linguagem simples, empática, sem jargão técnico."""


class DadosPaciente(BaseModel):
    nome: Optional[str] = ""
    idade: Optional[str] = ""
    sexo: Optional[str] = ""
    profissao: Optional[str] = ""
    estresse: Optional[str] = ""
    queixa: Optional[str] = ""
    evolucao: Optional[str] = ""
    evento: Optional[str] = ""
    dor_local: Optional[str] = ""
    eva: Optional[str] = ""
    dor_padrao: Optional[str] = ""
    dor_piora: Optional[str] = ""
    sintomas: Optional[str] = ""
    sono_qual: Optional[str] = ""
    sono_horas: Optional[str] = ""
    insonia: Optional[str] = ""
    posicao: Optional[str] = ""
    sono_obs: Optional[str] = ""
    bruxismo: Optional[str] = ""
    placa: Optional[str] = ""
    parafuncoes: Optional[str] = ""
    dtm_sinais: Optional[str] = ""
    abertura: Optional[str] = ""
    diag_dtm: Optional[str] = ""
    sistemicas: Optional[str] = ""
    meds: Optional[str] = ""
    sups: Optional[str] = ""
    labs: Optional[str] = ""
    img: Optional[str] = ""
    poli: Optional[str] = ""
    hist: Optional[str] = ""
    ctx: Optional[str] = ""
    obs: Optional[str] = ""


def montar_prompt(d: dict) -> str:
    linhas = ["DADOS DO PACIENTE:\n"]
    campos = [
        ("Nome", d.get("nome")), ("Idade", f"{d.get('idade')} anos" if d.get("idade") else ""),
        ("Sexo", d.get("sexo")), ("Profissão", d.get("profissao")),
        ("Estresse percebido", f"{d.get('estresse')}/10" if d.get("estresse") else ""),
        ("Queixa principal", d.get("queixa")), ("Tempo de evolução", d.get("evolucao")),
        ("Evento desencadeante", d.get("evento")), ("Localização da dor", d.get("dor_local")),
        ("EVA", f"{d.get('eva')}/10" if d.get("eva") else ""),
        ("Padrão temporal", d.get("dor_padrao")), ("Fatores de piora", d.get("dor_piora")),
        ("Sintomas associados", d.get("sintomas")), ("Qualidade do sono", d.get("sono_qual")),
        ("Horas de sono", f"{d.get('sono_horas')}h" if d.get("sono_horas") else ""),
        ("Insônia", d.get("insonia")), ("Posição ao dormir", d.get("posicao")),
        ("Obs. sono", d.get("sono_obs")), ("Bruxismo", d.get("bruxismo")),
        ("Dispositivo intraoral", d.get("placa")), ("Parafunções", d.get("parafuncoes")),
        ("Sinais DTM", d.get("dtm_sinais")), ("Abertura bucal", d.get("abertura")),
        ("Diagnóstico DTM prévio", d.get("diag_dtm")), ("Doenças sistêmicas", d.get("sistemicas")),
        ("Medicamentos", d.get("meds")), ("Suplementos", d.get("sups")),
        ("Exames laboratoriais", d.get("labs")), ("Exames de imagem", d.get("img")),
        ("Polissonografia", d.get("poli")), ("Histórico clínico", d.get("hist")),
        ("Contexto de vida", d.get("ctx")), ("Obs. dentista", d.get("obs")),
    ]
    for label, val in campos:
        if val and val.strip():
            linhas.append(f"• {label}: {val}")
    linhas.append("\nGere a análise clínica integrativa completa.")
    return "\n".join(linhas)


def parsear_secoes(texto: str) -> dict:
    import re
    secoes = {}
    mapa = {
        "correlacoes": ["1.", "CORRELAÇÕES", "CORRELACOES"],
        "hipoteses": ["2.", "HIPÓTESES", "HIPOTESES"],
        "exames": ["3.", "EXAMES"],
        "plano": ["4.", "PLANO", "TRATAMENTO"],
        "resumo_paciente": ["5.", "RESUMO", "PACIENTE"],
    }
    partes = re.split(r'\n##\s+', texto)
    for parte in partes:
        if not parte.strip():
            continue
        h = parte.split('\n')[0].upper()
        body = '\n'.join(parte.split('\n')[1:]).strip()
        for chave, marcadores in mapa.items():
            if any(m in h for m in marcadores):
                secoes[chave] = body
                break
    if not secoes:
        secoes["correlacoes"] = texto
    return secoes


@app.get("/status")
def status():
    return {"status": "online", "modelo": "claude-sonnet-4-6", "versao": "1.0"}


@app.post("/analisar")
def analisar(dados: DadosPaciente):
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="API key não configurada no servidor")
    if not dados.queixa:
        raise HTTPException(status_code=400, detail="Queixa principal é obrigatória")
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        prompt = montar_prompt(dados.dict())
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4000,
            system=SYSTEM,
            messages=[{"role": "user", "content": prompt}]
        )
        texto = message.content[0].text
        secoes = parsear_secoes(texto)
        return {
            "correlacoes": secoes.get("correlacoes", ""),
            "hipoteses": secoes.get("hipoteses", ""),
            "exames": secoes.get("exames", ""),
            "plano": secoes.get("plano", ""),
            "resumo_paciente": secoes.get("resumo_paciente", ""),
            "areas_detectadas": [],
            "fontes": [],
            "analise_bruta": texto
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
