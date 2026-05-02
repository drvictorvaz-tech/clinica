from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import anthropic
import os
import base64
import json
import re

app = FastAPI(title="Dr. Victor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

SYSTEM = """Voce e o assistente clinico do Dr. Victor Vaz, cirurgiao-dentista especialista em DTM, bruxismo, sono e dor cronica, com pos-graduacao em dor e foco em saude integrativa e funcional.

Seu raciocinio integra:
- Eixo estrutural/oclusal: ATM, musculos mastigatorios, coluna cervical
- Eixo neurologico: sensibilizacao central, sistema trigeminal
- Eixo do sono: bruxismo do sono, apneia, qualidade do sono
- Eixo inflamatorio/metabolico: laboratoriais, nutricao, suplementacao
- Eixo hormonal: cortisol, tireoide, testosterona, estrogenio
- Eixo emocional-comportamental: estresse, ansiedade, catastrofizacao

ESTRUTURA OBRIGATORIA - use exatamente estes titulos:

## 1. CORRELACOES CLINICAS
Conecte os sistemas: triade DTM-sono-bruxismo, eixo neuroinflamatorio, metabolico-hormonal, emocional.

## 2. HIPOTESES DIAGNOSTICAS INTEGRADAS
Liste por probabilidade. Para cada: mecanismo + evidencias + classificacao RDC/TMD quando aplicavel.

## 3. EXAMES COMPLEMENTARES SUGERIDOS
Para cada exame: justificativa clinica especifica para este paciente.

## 4. PLANO DE TRATAMENTO INTEGRATIVO
FASE 1 - Controle da dor aguda (2-4 semanas)
FASE 2 - Modulacao causal (1-3 meses)
FASE 3 - Manutencao e prevencao

## 5. RESUMO PARA O PACIENTE
Linguagem simples, empatica, sem jargao tecnico."""


class DadosPaciente(BaseModel):
    nome: Optional[str] = ""
    idade: Optional[str] = ""
    sexo: Optional[str] = ""
    profissao: Optional[str] = ""
    ecivil: Optional[str] = ""
    estresse: Optional[str] = ""
    queixa: Optional[str] = ""
    evolucao: Optional[str] = ""
    evento: Optional[str] = ""
    dor_local: Optional[str] = ""
    eva: Optional[str] = ""
    dpadrao: Optional[str] = ""
    dpiora: Optional[str] = ""
    sintomas: Optional[str] = ""
    sonoq: Optional[str] = ""
    sonoh: Optional[str] = ""
    insonia: Optional[str] = ""
    posicao: Optional[str] = ""
    sonoobs: Optional[str] = ""
    brux: Optional[str] = ""
    placa: Optional[str] = ""
    parafuncoes: Optional[str] = ""
    dtm_sinais: Optional[str] = ""
    abertura: Optional[str] = ""
    diagdtm: Optional[str] = ""
    sistemicas: Optional[str] = ""
    tratant: Optional[str] = ""
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
        ("Sexo", d.get("sexo")), ("Profissao", d.get("profissao")),
        ("Estado civil", d.get("ecivil")),
        ("Estresse percebido", f"{d.get('estresse')}/10" if d.get("estresse") else ""),
        ("Queixa principal", d.get("queixa")), ("Tempo de evolucao", d.get("evolucao")),
        ("Evento desencadeante", d.get("evento")), ("Localizacao da dor", d.get("dor_local")),
        ("EVA", f"{d.get('eva')}/10" if d.get("eva") else ""),
        ("Padrao temporal", d.get("dpadrao")), ("Fatores de piora", d.get("dpiora")),
        ("Sintomas associados", d.get("sintomas")), ("Qualidade do sono", d.get("sonoq")),
        ("Horas de sono", f"{d.get('sonoh')}h" if d.get("sonoh") else ""),
        ("Insonia", d.get("insonia")), ("Posicao ao dormir", d.get("posicao")),
        ("Obs. sono", d.get("sonoobs")), ("Bruxismo", d.get("brux")),
        ("Dispositivo intraoral", d.get("placa")), ("Parafuncoes", d.get("parafuncoes")),
        ("Sinais DTM", d.get("dtm_sinais")), ("Abertura bucal", d.get("abertura")),
        ("Diagnostico DTM previo", d.get("diagdtm")), ("Doencas sistemicas", d.get("sistemicas")),
        ("Tratamento anterior", d.get("tratant")),
        ("Medicamentos", d.get("meds")), ("Suplementos", d.get("sups")),
        ("Exames laboratoriais", d.get("labs")), ("Exames de imagem", d.get("img")),
        ("Polissonografia", d.get("poli")), ("Historico clinico", d.get("hist")),
        ("Contexto de vida", d.get("ctx")), ("Obs. dentista", d.get("obs")),
    ]
    for label, val in campos:
        if val and str(val).strip():
            linhas.append(f"• {label}: {val}")
    linhas.append("\nGere a analise clinica integrativa completa.")
    return "\n".join(linhas)


def parsear_secoes(texto: str) -> dict:
    secoes = {}
    mapa = {
        "correlacoes": ["1.", "CORRELACOES", "CORRELAÇÕES"],
        "hipoteses": ["2.", "HIPOTESES", "HIPÓTESES"],
        "exames": ["3.", "EXAMES"],
        "plano": ["4.", "PLANO"],
        "resumo_paciente": ["5.", "RESUMO"],
    }
    partes = re.split(r'\n(?=##)', texto)
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
        raise HTTPException(status_code=500, detail="API key nao configurada no servidor")
    if not dados.queixa:
        raise HTTPException(status_code=400, detail="Queixa principal e obrigatoria")
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


@app.post("/extrair-dados-audio")
async def extrair_dados_audio(transcricao: str = Form(...)):
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="API key nao configurada")
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        prompt = f"Voce e o assistente clinico do Dr. Victor Vaz, especialista em DTM.\nTranscricao da consulta:\n---\n{transcricao}\n---\nRetorne SOMENTE JSON com campos: nome, idade, sexo, profissao, ecivil, estresse, queixa, evolucao, evento, dpadrao, dpiora, sonoq, sonoh, insonia, posicao, sonoobs, brux, placa, abertura, diagdtm, sistemicas, tratant, meds, sups, labs, hist, ctx, obs. String vazia se nao mencionado."
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        texto = message.content[0].text
        json_match = re.search(r'\{.*\}', texto, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return {}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analisar-com-arquivos")
async def analisar_com_arquivos(
    dados: str = Form(...),
    arquivos: List[UploadFile] = File(default=[])
):
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="API key nao configurada")
    dados_dict = json.loads(dados)
    if not dados_dict.get("queixa"):
        raise HTTPException(status_code=400, detail="Queixa principal e obrigatoria")
    content = []
    prompt = montar_prompt(dados_dict)
    content.append({"type": "text", "text": prompt})
    arquivos_adicionados = []
    for arquivo in arquivos:
        file_bytes = await arquivo.read()
        if not file_bytes:
            continue
        file_b64 = base64.standard_b64encode(file_bytes).decode("utf-8")
        media_type = arquivo.content_type or ""
        if media_type.startswith("image/"):
            content.append({"type": "image", "source": {"type": "base64", "media_type": media_type, "data": file_b64}})
            arquivos_adicionados.append(arquivo.filename)
        elif media_type == "application/pdf":
            content.append({"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": file_b64}})
            arquivos_adicionados.append(arquivo.filename)
    if arquivos_adicionados:
        content.append({"type": "text", "text": f"\nArquivos: {', '.join(arquivos_adicionados)}. Analise e incorpore os achados."})
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        message = client.messages.create(model="claude-sonnet-4-6", max_tokens=4000, system=SYSTEM, messages=[{"role": "user", "content": content}])
        texto = message.content[0].text
        secoes = parsear_secoes(texto)
        return {"correlacoes": secoes.get("correlacoes", ""), "hipoteses": secoes.get("hipoteses", ""), "exames": secoes.get("exames", ""), "plano": secoes.get("plano", ""), "resumo_paciente": secoes.get("resumo_paciente", ""), "areas_detectadas": arquivos_adicionados, "fontes": [], "analise_bruta": texto}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



MODALIDADES_TRATAMENTO = [
    {"id": "avaliacao_multidisciplinar", "nome": "Avaliação Multidisciplinar", "descricao": "Avaliação integrada com dentista especializado em dor orofacial, fisioterapeuta e outros especialistas conforme necessidade."},
    {"id": "terapia_manual", "nome": "Terapia Manual (Fisioterapia)", "descricao": "Técnica para aliviar dor e tensão muscular na região da mandíbula e pescoço, realizada pelo fisioterapeuta."},
    {"id": "exercicios", "nome": "Exercícios de Fortalecimento e Alongamento", "descricao": "Exercícios específicos para fortalecer e alongar os músculos da mandíbula e pescoço, ajudando a aliviar dor e tensão."},
    {"id": "placa_oclusal", "nome": "Placa Oclusal / Placa Estabilizadora", "descricao": "Dispositivo personalizado que alivia a pressão na articulação da mandíbula, reduzindo dor e protegendo os dentes."},
    {"id": "agulhamento_seco", "nome": "Agulhamento Seco (Dry Needling)", "descricao": "Técnica que alivia a dor muscular reduzindo tensão na região da mandíbula e pescoço através de agulhas finas."},
    {"id": "laser_terapia", "nome": "Laser Terapia", "descricao": "Terapia com luz laser para aliviar dor, reduzir inflamação e promover cicatrização dos tecidos."},
    {"id": "viscossuplementacao", "nome": "Viscosuplementação", "descricao": "Procedimento que melhora a lubrificação e reduz a inflamação dentro da articulação da mandíbula."},
    {"id": "disbiose_intestinal", "nome": "Controle da Disbiose Intestinal", "descricao": "Orientação sobre dieta e suplementação com probióticos para melhorar o equilíbrio intestinal e reduzir inflamação."},
    {"id": "regulacao_sono", "nome": "Regulação do Sono", "descricao": "Orientações sobre higiene do sono e estratégias para melhorar a qualidade e duração do sono."},
    {"id": "controle_estresse", "nome": "Controle do Estresse e Ansiedade", "descricao": "Técnicas de relaxamento e gerenciamento do estresse, que agravam os sintomas de DTM e bruxismo."},
    {"id": "atividade_fisica", "nome": "Orientações sobre Atividade Física", "descricao": "Exercícios físicos regulares que reduzem a dor muscular e melhoram o bem-estar geral."}
]


class PropostaRequest(BaseModel):
    nome: Optional[str] = ""
    queixa: Optional[str] = ""
    hipoteses: Optional[str] = ""
    plano: Optional[str] = ""
    resumo_paciente: Optional[str] = ""


@app.post("/gerar-proposta")
def gerar_proposta(dados: PropostaRequest):
    """Gera proposta de tratamento personalizada para o paciente"""
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="API key não configurada")

    modalidades_lista = "\n".join([f"- {m['id']}: {m['nome']}" for m in MODALIDADES_TRATAMENTO])

    prompt = f"""Você é o assistente clínico do Dr. Victor Vaz, especialista em DTM, bruxismo, sono e dor crônica.

Com base nos dados clínicos abaixo, selecione quais modalidades são indicadas e gere uma proposta personalizada.

DADOS:
Nome: {dados.nome}
Queixa: {dados.queixa}
Hipóteses: {dados.hipoteses}
Plano clínico: {dados.plano}
Resumo: {dados.resumo_paciente}

MODALIDADES DISPONÍVEIS:
{modalidades_lista}

Retorne SOMENTE JSON válido:
{{
  "introducao": "Texto personalizado para {dados.nome} (2-3 frases, linguagem simples)",
  "modalidades_indicadas": ["id1", "id2"],
  "justificativas": {{"id": "justificativa em 1-2 frases acessíveis"}},
  "timeline": "tempo estimado para este caso",
  "proximos_passos": "2-3 passos concretos para o paciente",
  "observacoes": "expectativas realistas (máx 3 linhas)"
}}"""

    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        texto = message.content[0].text
        json_match = re.search(r'\{.*\}', texto, re.DOTALL)
        if not json_match:
            raise ValueError("JSON não encontrado")
        proposta_ia = json.loads(json_match.group())
        modalidades_completas = []
        for mid in proposta_ia.get("modalidades_indicadas", []):
            modal = next((m for m in MODALIDADES_TRATAMENTO if m["id"] == mid), None)
            if modal:
                modalidades_completas.append({
                    "id": modal["id"],
                    "nome": modal["nome"],
                    "descricao": modal["descricao"],
                    "justificativa": proposta_ia.get("justificativas", {}).get(mid, "")
                })
        return {
            "nome_paciente": dados.nome,
            "introducao": proposta_ia.get("introducao", ""),
            "modalidades": modalidades_completas,
            "timeline": proposta_ia.get("timeline", "Aproximadamente 60 dias"),
            "proximos_passos": proposta_ia.get("proximos_passos", ""),
            "observacoes": proposta_ia.get("observacoes", "")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
