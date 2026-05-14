from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import anthropic
import os
import base64
import json
import re
import datetime
try:
    from supabase import create_client, Client as SupabaseClient
    SUPABASE_DISPONIVEL = True
except ImportError:
    SUPABASE_DISPONIVEL = False

app = FastAPI(title="Dr. Victor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
_db_client = None

def get_db():
    global _db_client
    if _db_client is None and SUPABASE_DISPONIVEL and SUPABASE_URL and SUPABASE_KEY:
        _db_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _db_client


SYSTEM = """Você é o assistente clínico do Dr. Victor Vaz, cirurgião-dentista especialista em DTM, bruxismo, sono e dor orofacial, com pós-graduação em Dor Orofacial e foco em Saúde Integrativa e Funcional. CRO: 4923 SC — Balneário Camboriú, SC.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FILOSOFIA CLÍNICA — LEIA PRIMEIRO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Dr. Victor pratica medicina integrativa e funcional. Sua abordagem vai além do sintoma local e considera o organismo como sistema interconectado. Ele NUNCA usa Botox/Toxina Botulínica — não inclua esta modalidade em nenhuma hipótese ou plano.

TRATAMENTOS DO PROTOCOLO DO DR. VICTOR:
• Placa Oclusal / Placa Interoclusal (NÃO "placa miorrelaxante")
• Placa Estabilizadora — indicada especificamente para disco articular deslocado sem redução (bloqueio de abertura bucal)
• Placa de Avanço Mandibular (ronco e apneia)
• Terapia Manual e Fisioterapia Orofacial
• Exercícios de Fortalecimento e Alongamento Mandibular/Cervical
• Agulhamento Seco (Dry Needling) — pontos gatilho miofasciais
• Laser Terapia (Fotobiomodulação) — anti-inflamatório e analgésico
• Ozonioterapia — protocolo integrativo-funcional
• Viscosuplementação da ATM
• LDN (Baixa Dose de Naltrexona) — modulação neuroimune, dor crônica, fibromialgia
• Suplementação Funcional e Ortomolecular: Magnésio (Di-malato), CoQ10/Ubiquinol, Resveratrol trans, Vitamina D3+K2+E, Acetil-L-Carnitina, D-Ribose
• Controle da Disbiose Intestinal (probióticos, dieta anti-inflamatória)
• Regulação do Sono (higiene do sono + dispositivos)
• Controle do Estresse e Ansiedade
• Orientações de Atividade Física e Postura
• Avaliação Multidisciplinar (fisioterapeuta, fonoaudiólogo, neurologista, reumatologista)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODELO DE RACIOCÍNIO — 6 EIXOS INTEGRATIVOS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. EIXO ESTRUTURAL/OCLUSAL: ATM, músculos mastigatórios, coluna cervical, postura, oclusão
2. EIXO NEUROLÓGICO: sensibilização central, dor neuropática, nociplasticidade, zumbido somatossensorial
3. EIXO DO SONO: bruxismo do sono (STAB Eixo A), apneia (STOP-Bang, Epworth), qualidade do sono
4. EIXO INFLAMATÓRIO/METABÓLICO: PCR, IL-6, disbiose intestinal, deficiências (Mg, D3, B12, ferritina)
5. EIXO HORMONAL: cortisol, tireoide (T3/T4/TSH), testosterona, estrogênio, insulina
6. EIXO EMOCIONAL/PSICOSSOCIAL: estresse crônico, ansiedade (GAD-7), depressão (PHQ-9), catastrofização (PCS), cinesiofobia (Tampa)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AVALIAÇÃO DO BRUXISMO — STAB (Standardized Tool for Assessment of Bruxism)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instrumento padronizado utilizado pelo Dr. Victor. Considere sempre:

EIXO A — Status e Consequências:
• A1: Bruxismo do sono (frequência, histórico — autorrelato)
• A2: Bruxismo de vigília (grinding, clenching, tooth contact, mandible bracing)
• A3: Queixas (dor TMD, rigidez ao acordar, travamento, ruídos, dor muscular)
• A4–A6: Avaliação clínica (articulações, músculos, desgaste dental, tecidos orais)

EIXO B — Fatores de Risco e Comorbidades:
• B1: Psicossocial (estresse, GAD-7, PHQ-9)
• B2: Condições do sono (apneia, insônia, parassonias)
• B3: Condições não-sono (doenças sistêmicas, lifestyle)
• B4: Medicamentos e substâncias (cafeína, álcool, tabaco, SSRIs, benzodiazepínicos)
• B5: Fatores adicionais (genética, neurológicos, biomecânicos)

Classificar bruxismo como: PROVÁVEL (autorrelato + clínica), POSSÍVEL (apenas clínica) ou DEFINIDO (PSG/bruxômetro).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ESTRUTURA OBRIGATÓRIA DA ANÁLISE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REGRA DE ESCRITA: Sempre que usar siglas ou abreviações, escreva o significado entre parênteses na primeira vez que aparecer. Ex: ATM (Articulação Temporomandibular), DTM (Disfunção Temporomandibular), PSG (Polissonografia), EVA (Escala Visual Analógica), SNC (Sistema Nervoso Central).
REGRA DE FORMATAÇÃO: Sempre que apresentar dados comparativos, múltiplas variáveis, escalas, rankings ou parâmetros lado a lado, use tabelas em formato Markdown (| Coluna | Coluna |\n|---|---|\n| dado | dado |). Nunca liste em texto corrido quando uma tabela deixaria mais claro.

Use EXATAMENTE estes títulos:

## 1. CORRELAÇÕES CLÍNICAS
Conecte os sistemas: tríade DTM–sono–bruxismo, eixo neuroinflamatório, metabólico-hormonal, psicossocial (STAB Eixo B). Identifique ciclos de retroalimentação e mecanismos de perpetuação da dor.

## 2. HIPÓTESES DIAGNÓSTICAS INTEGRADAS
Liste por probabilidade. Para cada: mecanismo fisiopatológico + evidências + classificação DC/TMD + estadiamento STAB quando bruxismo envolvido.

## 3. EXAMES COMPLEMENTARES SUGERIDOS
Por exame: justificativa clínica específica. Incluir quando indicado: perfil inflamatório, painel hormonal, vitaminas/minerais, polissonografia, cone beam, RNM da ATM.

## 4. PLANO DE TRATAMENTO INTEGRATIVO
Baseado no protocolo do Dr. Victor. NUNCA incluir Botox.
FASE 1 — Controle da dor aguda (2–4 semanas)
FASE 2 — Modulação causal (1–3 meses)
FASE 3 — Manutenção e prevenção (contínuo)

## 5. RESUMO PARA O PACIENTE
Linguagem simples, empática, sem jargão técnico. Explique de forma clara e acolhedora: o que o paciente tem e por que está acontecendo, como o tratamento proposto vai ajudar, o que ele pode fazer em casa (hábitos, postura, qualidade do sono, manejo do estresse), o que deve evitar, e o que esperar ao longo do processo. Seja explicativo, motivador e tranquilizador. Máx 350 palavras."""


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
        "resumo": ["5.", "RESUMO"],
    }
    partes = re.split(r'\n(?=## )', texto)
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
    retur@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    try:
        with open("pacientes.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>App not found</h1>", status_code=404)

n {"status": "online", "modelo": "claude-sonnet-4-6", "versao": "2.0-integrativo-stab"}


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
            max_tokens=8000,
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
            "resumo": secoes.get("resumo", ""),
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
        message = client.messages.create(model="claude-sonnet-4-6", max_tokens=8000, system=SYSTEM, messages=[{"role": "user", "content": content}])
        texto = message.content[0].text
        secoes = parsear_secoes(texto)
        return {"correlacoes": secoes.get("correlacoes", ""), "hipoteses": secoes.get("hipoteses", ""), "exames": secoes.get("exames", ""), "plano": secoes.get("plano", ""), "resumo": secoes.get("resumo", ""), "areas_detectadas": arquivos_adicionados, "fontes": [], "analise_bruta": texto}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



MODALIDADES_TRATAMENTO = [
    {"id":"avaliacao","nome":"Avaliação Multidisciplinar","descricao":"Avaliação integrada com dentista especializado em dor orofacial, fisioterapeuta e outros especialistas (fonoaudiólogo, neurologista, reumatologista) conforme o caso."},
    {"id":"placa_estab","nome":"Placa Oclusal / Placa Estabilizadora","descricao":"Dispositivo intraoral personalizado que alivia a pressão na ATM e protege os dentes do desgaste. Necessidade avaliada durante a evolução do tratamento."},
    {"id":"placa_avanco","nome":"Placa de Avanço Mandibular (Ronco/Apneia)","descricao":"Dispositivo que avança a mandíbula durante o sono, desobstruindo as vias aéreas e reduzindo ronco e apneia leve a moderada."},
    {"id":"terapia_manual","nome":"Terapia Manual e Fisioterapia Orofacial","descricao":"Técnicas manuais para aliviar dor e tensão muscular na região da mandíbula, pescoço e coluna cervical, realizadas pelo fisioterapeuta."},
    {"id":"exercicios","nome":"Exercícios de Fortalecimento e Alongamento","descricao":"Protocolo específico para fortalecer e alongar os músculos da mandíbula, pescoço e postura, reduzindo dor e tensão crônica."},
    {"id":"agulhamento","nome":"Agulhamento Seco (Dry Needling)","descricao":"Técnica com agulhas finas em pontos gatilho miofasciais para aliviar dor muscular e restaurar função na região da mandíbula e pescoço."},
    {"id":"laser","nome":"Laser Terapia (Fotobiomodulação)","descricao":"Terapia com luz laser de baixa intensidade para aliviar dor, reduzir inflamação e promover regeneração tecidual e modulação neurológica."},
    {"id":"ozonio","nome":"Ozonioterapia","descricao":"Aplicação terapêutica do ozônio com propriedades anti-inflamatórias, antimicrobianas e regenerativas — parte do protocolo integrativo do Dr. Victor."},
    {"id":"viscosupl","nome":"Viscosuplementação da ATM","descricao":"Injeção intra-articular que melhora a lubrificação, reduz inflamação e alivia dor na articulação temporomandibular."},
    {"id":"ldn","nome":"LDN – Baixa Dose de Naltrexona","descricao":"Modulação neuroimune com Naltrexona em baixa dose (1,5–4,5 mg). Indicada para dor crônica, sensibilização central, fibromialgia e condições autoimunes associadas."},
    {"id":"suplementacao","nome":"Suplementação Funcional e Ortomolecular","descricao":"Protocolo individualizado com magnésio (Di-malato), CoQ10/Ubiquinol, vitamina D3+K2, resveratrol, acetil-L-carnitina e outros conforme perfil laboratorial."},
    {"id":"disbiose","nome":"Controle da Disbiose Intestinal / Probióticos","descricao":"Intervenção no eixo intestino-cérebro: dieta anti-inflamatória, probióticos e suplementos para restaurar microbiota e reduzir inflamação sistêmica."},
    {"id":"sono","nome":"Regulação do Sono (Higiene + Dispositivos)","descricao":"Protocolo completo de higiene do sono, orientações comportamentais e dispositivos intraorais quando indicados para bruxismo do sono e apneia."},
    {"id":"estresse","nome":"Controle do Estresse e Ansiedade","descricao":"Orientações sobre relaxamento, mindfulness, biofeedback e encaminhamento para psicólogo quando indicado pelo GAD-7 ou PCS."},
    {"id":"atividade","nome":"Orientações de Atividade Física e Postura","descricao":"Exercícios físicos regulares, correção postural e orientações ergonômicas para reduzir tensão muscular e melhorar qualidade de vida."},
    {"id":"retorno","nome":"Consultas de Retorno e Acompanhamento","descricao":"Consultas periódicas para monitorar evolução, ajustar dispositivos e protocolos, e garantir manutenção dos resultados a longo prazo."}
]

class PropostaRequest(BaseModel):
    nome: Optional[str] = ""
    queixa: Optional[str] = ""
    hipoteses: Optional[str] = ""
    plano: Optional[str] = ""
    resumo_paciente: Optional[str] = ""
    modalidades_selecionadas: Optional[List[str]] = []


@app.post("/gerar-proposta")
def gerar_proposta(dados: PropostaRequest):
    """Gera proposta de tratamento personalizada e integrativa para o paciente"""
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="API key nao configurada")

    if dados.modalidades_selecionadas:
        mods_disponiveis = [m for m in MODALIDADES_TRATAMENTO if m["nome"] in dados.modalidades_selecionadas]
        if not mods_disponiveis:
            mods_disponiveis = MODALIDADES_TRATAMENTO
    else:
        mods_disponiveis = MODALIDADES_TRATAMENTO

    modalidades_lista = "\n".join([f"- {m['id']}: {m['nome']}" for m in mods_disponiveis])

    prompt = f"""Voce e o assistente clinico do Dr. Victor Vaz, especialista em DTM, bruxismo, sono e dor cronica, com abordagem integrativa e funcional.
Com base nos dados clinicos abaixo, gere uma proposta de tratamento personalizada e humanizada.
IMPORTANTE: Dr. Victor NUNCA usa Botox/Toxina Botulinica. Nao mencione esta modalidade.

DADOS DO PACIENTE:
Nome: {dados.nome}
Queixa principal: {dados.queixa}

HIPOTESES DIAGNOSTICAS:
{dados.hipoteses}

PLANO DE TRATAMENTO CLINICO:
{dados.plano}

RESUMO PARA O PACIENTE:
{dados.resumo_paciente}

MODALIDADES SELECIONADAS PELO DR. VICTOR:
{modalidades_lista}

Retorne SOMENTE um JSON valido com esta estrutura:
{{
  "introducao": "Texto de boas-vindas personalizado para {dados.nome}, explicando o diagnostico em linguagem simples e empatica (2-3 frases)",
  "modalidades_indicadas": ["id1", "id2"],
  "justificativas": {{"id_modalidade": "justificativa clinica especifica para este paciente (1-2 frases)"}},
  "timeline": "Cronograma estimado especifico para este caso (Fase 1, 2 e 3)",
  "proximos_passos": "O que o paciente deve fazer agora (2-3 passos concretos)",
  "observacoes": "Informacoes importantes sobre o tratamento integrativo (max 3 linhas)"
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
            raise ValueError("JSON nao encontrado na resposta")

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



class SalvarRequest(BaseModel):
    dados_paciente: dict = {}
    resultado: dict = {}
    analise_bruta: str = ""
    notas: str = ""

class AtualizarRequest(BaseModel):
    resultado: Optional[dict] = None
    notas: Optional[str] = None
    dados_paciente: Optional[dict] = None

@app.get("/status-banco")
def status_banco():
    db = get_db()
    if not db:
        return {"configurado": False, "msg": "Adicione SUPABASE_URL e SUPABASE_KEY no Railway"}
    try:
        db.table("analises").select("id").limit(1).execute()
        return {"configurado": True, "msg": "Banco operacional"}
    except Exception as e:
        return {"configurado": False, "msg": str(e)}

@app.post("/salvar-analise")
def salvar_analise(req: SalvarRequest):
    db = get_db()
    if not db:
        raise HTTPException(503, "Banco nao configurado. Adicione SUPABASE_URL e SUPABASE_KEY no Railway.")
    row = {
        "nome_paciente": req.dados_paciente.get("nome", "Paciente sem nome"),
        "dados_paciente": req.dados_paciente,
        "resultado": req.resultado,
        "analise_bruta": req.analise_bruta,
        "notas": req.notas
    }
    resp = db.table("analises").insert(row).execute()
    if not resp.data:
        raise HTTPException(500, "Erro ao salvar analise")
    return {"id": resp.data[0]["id"], "criado_em": resp.data[0]["criado_em"]}

@app.get("/historico")
def historico(pagina: int = 1, limite: int = 20):
    db = get_db()
    if not db:
        raise HTTPException(503, "Banco nao configurado.")
    offset = (pagina - 1) * limite
    resp = db.table("analises").select("id,nome_paciente,criado_em,atualizado_em,notas").order("criado_em", desc=True).range(offset, offset + limite - 1).execute()
    count_resp = db.table("analises").select("id", count="exact").execute()
    return {"analises": resp.data, "total": count_resp.count or 0, "pagina": pagina, "limite": limite}

@app.get("/buscar-paciente")
def buscar_paciente_hist(nome: str):
    db = get_db()
    if not db:
        raise HTTPException(503, "Banco nao configurado.")
    resp = db.table("analises").select("id,nome_paciente,criado_em,atualizado_em,notas").ilike("nome_paciente", f"%{nome}%").order("criado_em", desc=True).execute()
    return {"analises": resp.data, "total": len(resp.data)}

@app.get("/analise/{analise_id}")
def obter_analise(analise_id: str):
    db = get_db()
    if not db:
        raise HTTPException(503, "Banco nao configurado.")
    resp = db.table("analises").select("*").eq("id", analise_id).execute()
    if not resp.data:
        raise HTTPException(404, "Analise nao encontrada")
    return resp.data[0]

@app.put("/analise/{analise_id}")
def atualizar_analise(analise_id: str, req: AtualizarRequest):
    db = get_db()
    if not db:
        raise HTTPException(503, "Banco nao configurado.")
    update_data: dict = {}
    if req.resultado is not None:
        update_data["resultado"] = req.resultado
    if req.notas is not None:
        update_data["notas"] = req.notas
    if req.dados_paciente is not None:
        update_data["dados_paciente"] = req.dados_paciente
    if not update_data:
        raise HTTPException(400, "Nenhum dado para atualizar")
    update_data["atualizado_em"] = datetime.datetime.utcnow().isoformat()
    resp = db.table("analises").update(update_data).eq("id", analise_id).execute()
    if not resp.data:
        raise HTTPException(404, "Analise nao encontrada")
    return resp.data[0]

@app.delete("/analise/{analise_id}")
def deletar_analise(analise_id: str):
    db = get_db()
    if not db:
        raise HTTPException(503, "Banco nao configurado.")
    db.table("analises").delete().eq("id", analise_id).execute()
    return {"ok": True, "id": analise_id}

class ChatPayload(BaseModel):
    mensagem: str
    contexto_analise: Optional[str] = ""
    historico: Optional[list] = []

@app.post("/chat")
async def chat_ia(payload: ChatPayload):
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    system_prompt = """Você é o assistente clínico do Dr. Victor Vaz, cirurgião-dentista especialista em DTM (Disfunção Temporomandibular), bruxismo, sono e dor orofacial. 
Auxilie o doutor a raciocinar clinicamente sobre o caso em questão.
Seja objetivo, técnico mas acessível. Use tabelas Markdown quando comparar dados.
Sempre que usar siglas, coloque o significado entre parênteses na primeira ocorrência."""

    messages = []
    for h in payload.historico:
        messages.append({"role": h["role"], "content": h["content"]})

    contexto = ""
    if payload.contexto_analise:
        contexto = f"\n\nCONTEXTO DA ANÁLISE ATUAL:\n{payload.contexto_analise}"

    messages.append({"role": "user", "content": payload.mensagem + contexto})

    resp = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        system=system_prompt,
        messages=messages
    )
    return {"resposta": resp.content[0].text}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))


class ChatProfessorPayload(BaseModel):
    mensagem: str
    historico: Optional[list] = []
    arquivo_texto: Optional[str] = ""
    arquivo_nome: Optional[str] = ""
    imagem_base64: Optional[str] = ""
    imagem_tipo: Optional[str] = ""

@app.post("/chat-professor")
async def chat_professor(payload: ChatProfessorPayload):
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    system_prompt = """Voce e um assistente clinico especializado em DTM (Disfuncao Temporomandibular), bruxismo, sono e dor orofacial.
Auxiliando professores, pesquisadores e profissionais da area odontologica e medica.
Seja tecnico, preciso e didatico. Use tabelas Markdown para comparacoes e dados estruturados.
Sempre que usar siglas, coloque o significado entre parenteses na primeira ocorrencia.
Responda sempre em portugues brasileiro."""
    messages = []
    for h in payload.historico:
        messages.append({"role": h["role"], "content": h["content"]})
    user_content = []
    if payload.imagem_base64 and payload.imagem_tipo:
        user_content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": payload.imagem_tipo,
                "data": payload.imagem_base64
            }
        })
    texto_msg = payload.mensagem or ""
    if payload.arquivo_texto:
        texto_msg += f"\n\n[Arquivo: {payload.arquivo_nome}]\n{payload.arquivo_texto[:10000]}"
    user_content.append({"type": "text", "text": texto_msg or "(arquivo enviado)"})
    messages.append({"role": "user", "content": user_content})
    resp = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=2048,
        system=system_prompt,
        messages=messages
    )
    return {"resposta": resp.content[0].text}

# ============================================================
# IA PACIENTES — Endpoints
# ============================================================

class PacienteCreate(BaseModel):
    nome: str
    cpf: Optional[str] = ""
    celular: Optional[str] = ""
    email: Optional[str] = ""
    data_nascimento: Optional[str] = None
    sexo: Optional[str] = ""
    profissao: Optional[str] = ""
    estado_civil: Optional[str] = ""
    plano_saude: Optional[str] = ""
    endereco: Optional[str] = ""
    como_conheceu: Optional[str] = ""
    observacoes: Optional[str] = ""

class PacienteUpdate(BaseModel):
    nome: Optional[str] = None
    cpf: Optional[str] = None
    celular: Optional[str] = None
    email: Optional[str] = None
    data_nascimento: Optional[str] = None
    sexo: Optional[str] = None
    profissao: Optional[str] = None
    estado_civil: Optional[str] = None
    plano_saude: Optional[str] = None
    endereco: Optional[str] = None
    como_conheceu: Optional[str] = None
    observacoes: Optional[str] = None

class EvolucaoCreate(BaseModel):
    paciente_id: str
    data_consulta: Optional[str] = None
    tipo: Optional[str] = "consulta"
    queixa_principal: Optional[str] = ""
    anamnese: Optional[str] = ""
    exame_clinico: Optional[str] = ""
    diagnostico: Optional[str] = ""
    plano_tratamento: Optional[str] = ""
    procedimentos_realizados: Optional[str] = ""
    orientacoes: Optional[str] = ""
    proxima_consulta: Optional[str] = None
    valor: Optional[float] = None
    status_pagamento: Optional[str] = "pendente"
    forma_pagamento: Optional[str] = ""

class EvolucaoUpdate(BaseModel):
    data_consulta: Optional[str] = None
    tipo: Optional[str] = None
    queixa_principal: Optional[str] = None
    anamnese: Optional[str] = None
    exame_clinico: Optional[str] = None
    diagnostico: Optional[str] = None
    plano_tratamento: Optional[str] = None
    procedimentos_realizados: Optional[str] = None
    orientacoes: Optional[str] = None
    proxima_consulta: Optional[str] = None
    valor: Optional[float] = None
    status_pagamento: Optional[str] = None
    forma_pagamento: Optional[str] = None

class AgendaCreate(BaseModel):
    paciente_id: Optional[str] = None
    nome_paciente: Optional[str] = ""
    data_hora: str
    duracao_minutos: Optional[int] = 60
    tipo: Optional[str] = "consulta"
    observacoes: Optional[str] = ""
    status: Optional[str] = "agendado"

class AgendaUpdate(BaseModel):
    paciente_id: Optional[str] = None
    nome_paciente: Optional[str] = None
    data_hora: Optional[str] = None
    duracao_minutos: Optional[int] = None
    tipo: Optional[str] = None
    observacoes: Optional[str] = None
    status: Optional[str] = None

# Pacientes
@app.post("/pacientes")
def criar_paciente(req: PacienteCreate):
    db = get_db()
    if not db:
        raise HTTPException(503, "Banco nao configurado.")
    row = req.dict()
    if not row.get("data_nascimento"):
        row["data_nascimento"] = None
    resp = db.table("pacientes").insert(row).execute()
    if not resp.data:
        raise HTTPException(500, "Erro ao criar paciente")
    return resp.data[0]

@app.get("/pacientes")
def listar_pacientes(busca: str = "", pagina: int = 1, limite: int = 50):
    db = get_db()
    if not db:
        raise HTTPException(503, "Banco nao configurado.")
    offset = (pagina - 1) * limite
    q = db.table("pacientes").select("id,nome,cpf,celular,email,data_nascimento,plano_saude,ativo,criado_em").eq("ativo", True)
    if busca:
        q = q.ilike("nome", f"%{busca}%")
    resp = q.order("nome").range(offset, offset + limite - 1).execute()
    count_q = db.table("pacientes").select("id", count="exact").eq("ativo", True)
    if busca:
        count_q = count_q.ilike("nome", f"%{busca}%")
    count_resp = count_q.execute()
    return {"pacientes": resp.data, "total": count_resp.count or 0, "pagina": pagina, "limite": limite}

@app.get("/pacientes/{paciente_id}")
def obter_paciente(paciente_id: str):
    db = get_db()
    if not db:
        raise HTTPException(503, "Banco nao configurado.")
    resp = db.table("pacientes").select("*").eq("id", paciente_id).execute()
    if not resp.data:
        raise HTTPException(404, "Paciente nao encontrado")
    return resp.data[0]

@app.put("/pacientes/{paciente_id}")
def atualizar_paciente(paciente_id: str, req: PacienteUpdate):
    db = get_db()
    if not db:
        raise HTTPException(503, "Banco nao configurado.")
    update_data = {k: v for k, v in req.dict().items() if v is not None}
    update_data["atualizado_em"] = datetime.datetime.utcnow().isoformat()
    resp = db.table("pacientes").update(update_data).eq("id", paciente_id).execute()
    if not resp.data:
        raise HTTPException(404, "Paciente nao encontrado")
    return resp.data[0]

@app.delete("/pacientes/{paciente_id}")
def deletar_paciente(paciente_id: str):
    db = get_db()
    if not db:
        raise HTTPException(503, "Banco nao configurado.")
    db.table("pacientes").update({"ativo": False, "atualizado_em": datetime.datetime.utcnow().isoformat()}).eq("id", paciente_id).execute()
    return {"ok": True}

# Evolucoes
@app.post("/evolucoes")
def criar_evolucao(req: EvolucaoCreate):
    db = get_db()
    if not db:
        raise HTTPException(503, "Banco nao configurado.")
    row = req.dict()
    if not row.get("data_consulta"):
        row["data_consulta"] = datetime.date.today().isoformat()
    if not row.get("proxima_consulta"):
        row["proxima_consulta"] = None
    if not row.get("paciente_id"):
        raise HTTPException(400, "paciente_id e obrigatorio")
    resp = db.table("evolucoes").insert(row).execute()
    if not resp.data:
        raise HTTPException(500, "Erro ao criar evolucao")
    evolucao = resp.data[0]
    if req.valor and req.valor > 0 and req.status_pagamento and req.status_pagamento != "pendente":
        try:
            paciente = db.table("pacientes").select("nome").eq("id", req.paciente_id).execute()
            nome_pac = paciente.data[0]["nome"] if paciente.data else "Paciente"
            lancamento = {
                "descricao": f"Consulta odontologica - {nome_pac}",
                "valor": req.valor,
                "tipo": "receita",
                "categoria": "Consulta Odontologica",
                "data": req.data_consulta or datetime.date.today().isoformat(),
                "status": req.status_pagamento,
                "forma_pagamento": req.forma_pagamento or "",
                "origem": "ia_pacientes"
            }
            db.table("transactions").insert(lancamento).execute()
            db.table("evolucoes").update({"enviado_financeiro": True}).eq("id", evolucao["id"]).execute()
        except Exception:
            pass
    return evolucao

@app.get("/pacientes/{paciente_id}/evolucoes")
def listar_evolucoes(paciente_id: str):
    db = get_db()
    if not db:
        raise HTTPException(503, "Banco nao configurado.")
    resp = db.table("evolucoes").select("*").eq("paciente_id", paciente_id).order("data_consulta", desc=True).execute()
    return {"evolucoes": resp.data, "total": len(resp.data)}

@app.get("/evolucoes/{evolucao_id}")
def obter_evolucao(evolucao_id: str):
    db = get_db()
    if not db:
        raise HTTPException(503, "Banco nao configurado.")
    resp = db.table("evolucoes").select("*").eq("id", evolucao_id).execute()
    if not resp.data:
        raise HTTPException(404, "Evolucao nao encontrada")
    return resp.data[0]

@app.put("/evolucoes/{evolucao_id}")
def atualizar_evolucao(evolucao_id: str, req: EvolucaoUpdate):
    db = get_db()
    if not db:
        raise HTTPException(503, "Banco nao configurado.")
    update_data = {k: v for k, v in req.dict().items() if v is not None}
    update_data["atualizado_em"] = datetime.datetime.utcnow().isoformat()
    resp = db.table("evolucoes").update(update_data).eq("id", evolucao_id).execute()
    if not resp.data:
        raise HTTPException(404, "Evolucao nao encontrada")
    return resp.data[0]

@app.delete("/evolucoes/{evolucao_id}")
def deletar_evolucao(evolucao_id: str):
    db = get_db()
    if not db:
        raise HTTPException(503, "Banco nao configurado.")
    db.table("evolucoes").delete().eq("id", evolucao_id).execute()
    return {"ok": True}

# Agenda
@app.post("/agenda")
def criar_agendamento(req: AgendaCreate):
    db = get_db()
    if not db:
        raise HTTPException(503, "Banco nao configurado.")
    row = req.dict()
    if not row.get("paciente_id"):
        row["paciente_id"] = None
    resp = db.table("agenda").insert(row).execute()
    if not resp.data:
        raise HTTPException(500, "Erro ao criar agendamento")
    return resp.data[0]

@app.get("/agenda")
def listar_agenda(data_inicio: Optional[str] = None, data_fim: Optional[str] = None):
    db = get_db()
    if not db:
        raise HTTPException(503, "Banco nao configurado.")
    q = db.table("agenda").select("*")
    if data_inicio:
        q = q.gte("data_hora", data_inicio)
    if data_fim:
        q = q.lte("data_hora", data_fim)
    resp = q.order("data_hora").execute()
    return {"agendamentos": resp.data, "total": len(resp.data)}

@app.put("/agenda/{agenda_id}")
def atualizar_agendamento(agenda_id: str, req: AgendaUpdate):
    db = get_db()
    if not db:
        raise HTTPException(503, "Banco nao configurado.")
    update_data = {k: v for k, v in req.dict().items() if v is not None}
    update_data["atualizado_em"] = datetime.datetime.utcnow().isoformat()
    resp = db.table("agenda").update(update_data).eq("id", agenda_id).execute()
    if not resp.data:
        raise HTTPException(404, "Agendamento nao encontrado")
    return resp.data[0]

@app.delete("/agenda/{agenda_id}")
def deletar_agendamento(agenda_id: str):
    db = get_db()
    if not db:
        raise HTTPException(503, "Banco nao configurado.")
    db.table("agenda").delete().eq("id", agenda_id).execute()
    return {"ok": True}


# ─── Arquivos ────────────────────────────────────────────────────────────────

class ArquivoCreate(BaseModel):
    paciente_id: str
    nome: str
    tipo_arquivo: Optional[str] = ""
    descricao: Optional[str] = ""
    conteudo_base64: Optional[str] = ""

@app.post("/pacientes/{paciente_id}/arquivos")
def criar_arquivo(paciente_id: str, req: ArquivoCreate):
    db = get_db()
    if not db:
        raise HTTPException(503, "Banco nao configurado.")
    dados = {
        "paciente_id": paciente_id,
        "nome": req.nome,
        "tipo_arquivo": req.tipo_arquivo or "",
        "descricao": req.descricao or "",
        "conteudo_base64": req.conteudo_base64 or "",
        "tamanho_bytes": int(len(req.conteudo_base64 or "") * 3 / 4),
        "criado_em": datetime.datetime.utcnow().isoformat(),
        "atualizado_em": datetime.datetime.utcnow().isoformat(),
    }
    try:
        resp = db.table("arquivos").insert(dados).execute()
        res = dict(resp.data[0]) if resp.data else dados
        res.pop("conteudo_base64", None)
        return res
    except Exception as e:
        raise HTTPException(500, f"Erro ao criar arquivo: {str(e)}")

@app.get("/pacientes/{paciente_id}/arquivos")
def listar_arquivos(paciente_id: str):
    db = get_db()
    if not db:
        raise HTTPException(503, "Banco nao configurado.")
    try:
        resp = db.table("arquivos").select("id,nome,tipo_arquivo,descricao,tamanho_bytes,criado_em").eq("paciente_id", paciente_id).order("criado_em", desc=True).execute()
        return {"arquivos": resp.data or [], "total": len(resp.data or [])}
    except Exception:
        return {"arquivos": [], "total": 0}

@app.get("/arquivos/{arquivo_id}/conteudo")
def get_arquivo_conteudo(arquivo_id: str):
    db = get_db()
    if not db:
        raise HTTPException(503, "Banco nao configurado.")
    try:
        resp = db.table("arquivos").select("*").eq("id", arquivo_id).execute()
        if not resp.data:
            raise HTTPException(404, "Arquivo nao encontrado")
        return resp.data[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))

@app.delete("/arquivos/{arquivo_id}")
def deletar_arquivo(arquivo_id: str):
    db = get_db()
    if not db:
        raise HTTPException(503, "Banco nao configurado.")
    try:
        db.table("arquivos").delete().eq("id", arquivo_id).execute()
    except Exception:
        pass
    return {"ok": True}

# ─── Importacao em massa de evolucoes ────────────────────────────────────────

class EvolucaoBulkItem(BaseModel):
    paciente_cpf: Optional[str] = ""
    paciente_nome: Optional[str] = ""
    data_consulta: Optional[str] = None
    tipo: Optional[str] = "consulta"
    queixa_principal: Optional[str] = ""
    anamnese: Optional[str] = ""
    exame_clinico: Optional[str] = ""
    diagnostico: Optional[str] = ""
    plano_tratamento: Optional[str] = ""
    procedimentos_realizados: Optional[str] = ""
    orientacoes: Optional[str] = ""
    valor: Optional[float] = None
    status_pagamento: Optional[str] = "pendente"

class EvolucaoBulkCreate(BaseModel):
    evolucoes: List[EvolucaoBulkItem]

@app.post("/evolucoes/importar")
def importar_evolucoes_bulk(req: EvolucaoBulkCreate):
    """Importa evolucoes em massa. Resolve paciente por CPF ou nome."""
    db = get_db()
    if not db:
        raise HTTPException(503, "Banco nao configurado.")
    ok, erros = 0, []
    cache_pac: dict = {}
    for i, evo in enumerate(req.evolucoes):
        try:
            pac_id = None
            if evo.paciente_cpf:
                cpf_clean = re.sub(r"\D", "", evo.paciente_cpf)
                if cpf_clean in cache_pac:
                    pac_id = cache_pac[cpf_clean]
                else:
                    res = db.table("pacientes").select("id,cpf").execute()
                    for p in (res.data or []):
                        if re.sub(r"\D", "", p.get("cpf","")) == cpf_clean:
                            pac_id = p["id"]
                            cache_pac[cpf_clean] = pac_id
                            break
            if not pac_id and evo.paciente_nome:
                key = evo.paciente_nome.strip().lower()
                if key in cache_pac:
                    pac_id = cache_pac[key]
                else:
                    nome0 = evo.paciente_nome.split()[0] if evo.paciente_nome.split() else ""
                    res = db.table("pacientes").select("id,nome").ilike("nome", f"%{nome0}%").execute()
                    for p in (res.data or []):
                        if p.get("nome","").strip().lower() == key:
                            pac_id = p["id"]
                            cache_pac[key] = pac_id
                            break
                    if not pac_id and res.data:
                        pac_id = res.data[0]["id"]
                        cache_pac[key] = pac_id
            if not pac_id:
                erros.append({"linha": i, "msg": f"Paciente nao encontrado: {evo.paciente_nome or evo.paciente_cpf}"})
                continue
            dados = {
                "paciente_id": pac_id,
                "data_consulta": evo.data_consulta,
                "tipo": evo.tipo or "consulta",
                "queixa_principal": evo.queixa_principal or "",
                "anamnese": evo.anamnese or "",
                "exame_clinico": evo.exame_clinico or "",
                "diagnostico": evo.diagnostico or "",
                "plano_tratamento": evo.plano_tratamento or "",
                "procedimentos_realizados": evo.procedimentos_realizados or "",
                "orientacoes": evo.orientacoes or "",
                "valor": evo.valor,
                "status_pagamento": evo.status_pagamento or "pendente",
                "criado_em": datetime.datetime.utcnow().isoformat(),
                "atualizado_em": datetime.datetime.utcnow().isoformat(),
            }
            db.table("evolucoes").insert(dados).execute()
            ok += 1
        except Exception as e:
            erros.append({"linha": i, "msg": str(e)[:100]})
    return {"importados": ok, "erros": len(erros), "detalhes_erros": erros[:20]}



# ─── Migração — endpoint temporário ─────────────────────────────────────────

@app.post("/admin/migrate-arquivos")
def admin_migrate_arquivos(token: str = ""):
    """Cria tabela arquivos via pg-meta (Supabase internal) ou Management API."""
    import urllib.request as _ur
    import urllib.error as _ue
    
    sql = """CREATE TABLE IF NOT EXISTS public.arquivos (
    id              UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    paciente_id     UUID REFERENCES public.pacientes(id) ON DELETE CASCADE,
    nome            TEXT NOT NULL,
    tipo_arquivo    TEXT DEFAULT '',
    descricao       TEXT DEFAULT '',
    conteudo_base64 TEXT DEFAULT '',
    tamanho_bytes   INTEGER DEFAULT 0,
    criado_em       TIMESTAMPTZ DEFAULT NOW(),
    atualizado_em   TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_arquivos_paciente ON public.arquivos(paciente_id);
ALTER TABLE public.arquivos ENABLE ROW LEVEL SECURITY;"""
    
    results = []
    sb_url = os.environ.get("SUPABASE_URL", "")
    sb_key = os.environ.get("SUPABASE_KEY", "")
    mgmt_token = token or os.environ.get("SUPABASE_MGMT_TOKEN", "")
    
    import re as _re
    proj_ref_match = _re.search(r'https://([^.]+)\.supabase\.co', sb_url)
    proj_ref = proj_ref_match.group(1) if proj_ref_match else ""
    
    # Método 1: pg-meta v0 query endpoint (usado pelo Supabase Studio internamente)
    for key_name, key_val in [("sb_key", sb_key), ("mgmt_token", mgmt_token)]:
        if not key_val:
            continue
        try:
            data = json.dumps({"query": sql}).encode("utf-8")
            req = _ur.Request(
                f"{sb_url}/pg-meta/v0/query",
                data=data,
                method="POST",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {key_val}",
                }
            )
            with _ur.urlopen(req, timeout=15) as resp:
                body = resp.read().decode()[:300]
                results.append({"method": f"pg_meta/{key_name}", "ok": True, "body": body})
                break
        except _ue.HTTPError as e:
            results.append({"method": f"pg_meta/{key_name}", "ok": False, "status": e.code, "error": e.read().decode()[:150]})
        except Exception as e:
            results.append({"method": f"pg_meta/{key_name}", "ok": False, "error": str(e)[:150]})
    
    # Método 2: Management API
    if proj_ref and mgmt_token:
        try:
            data = json.dumps({"query": sql}).encode("utf-8")
            req = _ur.Request(
                f"https://api.supabase.com/v1/projects/{proj_ref}/database/query",
                data=data,
                method="POST",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {mgmt_token}",
                }
            )
            with _ur.urlopen(req, timeout=15) as resp:
                results.append({"method": "mgmt_api", "ok": True, "body": resp.read().decode()[:200]})
        except _ue.HTTPError as e:
            results.append({"method": "mgmt_api", "ok": False, "status": e.code, "error": e.read().decode()[:150]})
        except Exception as e:
            results.append({"method": "mgmt_api", "ok": False, "error": str(e)[:150]})
    
    # Método 3: tenta via RPC (funções utilitarias que podem existir)
    db = get_db()
    if db:
        for fn in ["exec_sql", "query", "run_sql", "execute"]:
            try:
                db.rpc(fn, {"query": sql}).execute()
                results.append({"method": f"rpc/{fn}", "ok": True})
                break
            except Exception as e:
                results.append({"method": f"rpc/{fn}", "ok": False, "error": str(e)[:80]})
    
    # Verifica resultado
    table_exists = False
    if db:
        try:
            db.table("arquivos").select("id").limit(1).execute()
            table_exists = True
        except Exception:
            pass
    
    # Decodifica JWT para ver o role
    jwt_role = ""
    if sb_key:
        try:
            import base64
            parts = sb_key.split(".")
            if len(parts) == 3:
                payload = parts[1] + "=="
                decoded = json.loads(base64.urlsafe_b64decode(payload).decode())
                jwt_role = decoded.get("role", "")
        except Exception:
            pass
    
    return {
        "proj_ref": proj_ref,
        "jwt_role": jwt_role,
        "has_mgmt_token": bool(mgmt_token),
        "results": results,
        "table_exists": table_exists,
    }
