from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional, Any, Dict
from datetime import datetime
import json, os, anthropic

router_anamnese = APIRouter()

class AnamnesePayload(BaseModel):
    fonte: Optional[str] = "ficha_online"
    ts: Optional[str] = None
    paciente: Optional[Dict[str, Any]] = {}
    anotacoes: Optional[Dict[str, Any]] = {}
    scores: Optional[Dict[str, Any]] = {}

@router_anamnese.post("/anamnese-paciente")
async def receber_anamnese(payload: AnamnesePayload):
    """
    Recebe ficha preenchida pelo paciente online.
    Salva localmente + dispara análise da IA Saúde automaticamente.
    """
    paciente = payload.paciente or {}
    scores = payload.scores or {}
    anotacoes = payload.anotacoes or {}

    nome = paciente.get("nome", "Paciente")
    ts = payload.ts or datetime.now().isoformat()

    # 1. Monta o contexto clínico completo para a IA Saúde
    contexto_clinico = f"""
ANAMNESE COMPLETA — {nome}
Recebida em: {ts}
Fonte: {payload.fonte}

=== IDENTIFICAÇÃO ===
Nome: {paciente.get('nome','—')}
Nascimento: {paciente.get('nasc','—')} | Sexo: {paciente.get('sexo','—')}
WhatsApp: {paciente.get('tel','—')} | Profissão: {paciente.get('prof','—')}
Estresse percebido: {paciente.get('est','—')}/10

=== DOR E QUEIXA PRINCIPAL ===
{paciente.get('queixa','—')}
Tempo de evolução: {paciente.get('tempo','—')}
Evento desencadeante: {paciente.get('evento','—')}
Locais: {', '.join(paciente.get('locais',[]) or [])}
EVA atual: {scores.get('eva','—')}/10
Tipo: {paciente.get('tipodor','—')} | Padrão: {paciente.get('padrao','—')}
Piora com: {', '.join(paciente.get('piora',[]) or [])}
Melhora com: {paciente.get('melhora','—')}
Outros sintomas: {', '.join(paciente.get('outrosint',[]) or [])}

=== SONO ===
Qualidade: {paciente.get('qsono','—')} | Horas/noite: {paciente.get('horas','—')}
Insônia: {', '.join(paciente.get('insonia',[]) or [])}
Ronco: {paciente.get('ronco','—')} | Apneia relatada: {paciente.get('apneia','—')}
Posição: {paciente.get('posicao','—')}
Epworth: {scores.get('epworth','—')}/24

=== BRUXISMO / ATM ===
Bruxismo noturno: {paciente.get('bsono','—')}
Bruxismo diurno: {paciente.get('bdia','—')}
Placa dental: {paciente.get('placa','—')}
Parafunções: {', '.join(paciente.get('paraf',[]) or [])}
Estalo ATM: {paciente.get('estalo','—')} | Travamento: {paciente.get('trava','—')}
Dor ao mastigar: {paciente.get('dormastigar','—')} | DTM prévio: {paciente.get('dtmprev','—')}

=== SAÚDE GERAL ===
Doenças: {', '.join(paciente.get('doencas',[]) or [])}
Medicamentos: {paciente.get('meds','—')}
Suplementos: {paciente.get('sups','—')}
Tratamentos anteriores: {paciente.get('tratant','—')}
Exames imagem: {paciente.get('eximg','—')} | Polissonografia: {paciente.get('psg','—')}
Alimentação: {paciente.get('alim','—')} | Atividade física: {paciente.get('atfis','—')}

=== PSICOSSOCIAL ===
GAD-7 (ansiedade): {scores.get('gad7','—')}/21
PHQ-9 (humor/depressão): {scores.get('phq9','—')}/27
PCS (catastrofização): {scores.get('pcs','—')}/16
DN4 (neuropático): {scores.get('dn4','—')}/7 — Itens: {', '.join(paciente.get('dn4',[]) or [])}

=== CONTEXTO DE VIDA ===
{paciente.get('contexto','—')}
Expectativa: {paciente.get('expect','—')}
Observações: {paciente.get('obs','—')}

=== ANOTAÇÕES DA CONSULTA ===
Exame físico: {anotacoes.get('exame','—')}
Hipóteses (6 Eixos + STAB): {anotacoes.get('hipoteses','—')}
Exames solicitados: {anotacoes.get('exames_sol','—')}
Conduta: {anotacoes.get('conduta','—')}
Observações adicionais: {anotacoes.get('obs','—')}
"""

    # 2. Chama a IA Saúde automaticamente com o contexto completo
    prompt_ia_saude = f"""Você é a IA Saúde do Dr. Victor Vaz, especialista em DTM, Dor Orofacial e Saúde Integrativa.

Analise a anamnese abaixo e gere um prontuário clínico completo seguindo OBRIGATORIAMENTE:

1. RESUMO CLÍNICO (3-4 linhas)
2. ANÁLISE POR EIXO INTEGRATIVO:
   - Eixo 1 — Estrutural/Occlusal
   - Eixo 2 — Neurológico
   - Eixo 3 — Sono
   - Eixo 4 — Inflamatório/Metabólico
   - Eixo 5 — Hormonal
   - Eixo 6 — Psicossocial
3. CLASSIFICAÇÃO STAB (PROVÁVEL / POSSÍVEL / DEFINIDO)
4. HIPÓTESES DIAGNÓSTICAS PRINCIPAIS
5. ALERTAS CLÍNICOS (scores elevados, red flags)
6. SUGESTÕES DE CONDUTA INTEGRATIVA

REGRAS:
- NUNCA use "placa miorrelaxante" — use sempre "Placa Oclusal Estabilizadora"
- NUNCA sugira Botox/Toxina Botulínica
- Use linguagem técnica para o Dr. Victor

{contexto_clinico}"""

    # 3. Chama Claude API
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    
    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt_ia_saude}]
        )
        analise_ia = response.content[0].text
    except Exception as e:
        analise_ia = f"[Erro ao gerar análise: {str(e)}]"

    # 4. Salva no arquivo de fichas (ou banco de dados se existir)
    ficha_completa = {
        "id": f"ficha_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{nome[:10].replace(' ','_')}",
        "timestamp": ts,
        "paciente": paciente,
        "scores": scores,
        "anotacoes": anotacoes,
        "analise_ia_saude": analise_ia,
        "contexto_clinico": contexto_clinico
    }

    # Salva em fichas.json (adapte para seu banco se tiver)
    fichas_path = "fichas_pacientes.json"
    try:
        if os.path.exists(fichas_path):
            with open(fichas_path, "r") as f:
                fichas = json.load(f)
        else:
            fichas = []
        fichas.append(ficha_completa)
        with open(fichas_path, "w") as f:
            json.dump(fichas, f, ensure_ascii=False, indent=2)
    except Exception as e:
        pass  # log do erro

    # 5. AUTOMATICO: cria/acha o paciente e lanca a anamnese como evolucao na ficha
    if str(payload.fonte or "").startswith("ficha"):
        try:
            from supabase import create_client as _cc
            _u2 = os.environ.get("SUPABASE_URL", "")
            _k2 = os.environ.get("SUPABASE_KEY", "")
            if _u2 and _k2:
                _cli = _cc(_u2, _k2)
                _cel = paciente.get("tel", "") or paciente.get("celular", "")
                _pid = None
                try:
                    if _cel:
                        _ex = _cli.table("pacientes").select("id").eq("celular", _cel).limit(1).execute()
                        if _ex.data:
                            _pid = _ex.data[0]["id"]
                    if not _pid and nome:
                        _ex2 = _cli.table("pacientes").select("id").ilike("nome", nome).limit(1).execute()
                        if _ex2.data:
                            _pid = _ex2.data[0]["id"]
                except Exception:
                    _pid = None
                if not _pid:
                    _novo = _cli.table("pacientes").insert({
                        "nome": nome or "Paciente",
                        "celular": _cel,
                        "email": paciente.get("email", ""),
                        "data_nascimento": (paciente.get("nasc") or None),
                        "sexo": paciente.get("sexo", ""),
                        "profissao": paciente.get("prof", ""),
                        "observacoes": "Cadastrado pela ficha de anamnese online",
                    }).execute()
                    if _novo.data:
                        _pid = _novo.data[0]["id"]
                if _pid:
                    _cli.table("evolucoes").insert({
                        "paciente_id": _pid,
                        "tipo": "anamnese_online",
                        "queixa_principal": paciente.get("queixa", ""),
                        "anamnese": contexto_clinico,
                        "diagnostico": (analise_ia or "")[:8000],
                    }).execute()
        except Exception:
            pass

    return {
        "status": "ok",
        "id": ficha_completa["id"],
        "nome": nome,
        "analise_ia_saude": analise_ia,
        "message": f"Anamnese de {nome} recebida e analisada com sucesso."
    }
