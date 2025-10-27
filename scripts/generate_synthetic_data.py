import json
import random
import argparse
from datetime import datetime, timedelta
from pathlib import Path

CLUBS = [
    "Clube Alpha", "Clube Beta", "Clube Gama", "Clube Delta", "Clube Épsilon",
    "Bela Vista", "Gremio Fronteira", "Comercial Carazinho", "Comercial Cascavel",
    "Country POA", "Clube Gaúcho", "SR Ijui", "Caixeiral Campestre PF", "TC Rio Branco"
]

# Templates por prioridade
TEMPLATES = {
    1: {
        "pergunta": [
            "Quero emitir um relatório simples de pagamentos do mês",
            "Onde encontro a lista de associados ativos?",
            "Como faço para atualizar o endereço de um sócio?",
            "Quero saber os horários de atendimento do balcão",
            "Como consulto a política de cancelamento?",
            "Onde vejo os valores de mensalidade por categoria?",
            "Quero um passo a passo para baixar o aplicativo",
            "Como faço para consultar o histórico de pagamentos?",
            "Onde encontro o manual do usuário?",
            "Quero a lista dos dependentes de um associado",
            "Onde vejo a validade dos exames médicos?",
            "Quero consultar convites emitidos esta semana",
            "Onde verifico bloqueios de acesso do clube?",
            "Onde vejo atividades com vagas disponíveis?",
            "Quero exportar dados de associados para CSV",
            "Como ordenar relatório por nome?"
        ],
        "resposta": [
            "Use Relatórios/Financeiros/Pagamentos com filtro de mês.",
            "Menu Relatórios/Quadro Social/Associados Ativos.",
            "Cadastro/Quadro Social/Associados > Editar endereço.",
            "Horários no Manual Interno, aba Atendimento.",
            "Relatórios/Contratos > Política de cancelamento.",
            "Relatórios/Financeiros/Tabela de Mensalidades.",
            "Acesse loja Android/iOS e procure por Forza.",
            "Atendimento > Aba Pagamentos > Histórico.",
            "Ferramentas/Documentação > Manual do Usuário.",
            "Cadastro > Associados > Dependentes > Listar.",
            "Relatórios/Atividades/Exames > Vencimentos.",
            "Relatórios/Convites > Filtro semanal.",
            "Relatórios/Acessos > Bloqueios.",
            "Relatórios/Atividades > Vagas disponíveis.",
            "Ferramentas/Exportar > Associados (CSV).",
            "Coloque atributo Nome como primeiro da lista."
        ]
    },
    5: {
        "pergunta": [
            "Preciso cadastrar um novo associado no sistema",
            "Como configurar a impressora de recibos?",
            "Preciso alterar o vencimento da mensalidade",
            "Como faço para gerar mensalidades coletivas?",
            "Quero integrar o sistema com a contabilidade",
            "Preciso instalar uma atualização do sistema",
            "Como configurar permissões de acesso para um usuário?",
            "Preciso resetar a senha de um associado",
            "Como cadastro dependentes no sistema?",
            "Como configurar cobrador padrão para débitos?",
            "Como habilitar fila de espera em atividade?",
            "Preciso configurar carteira social para dependente",
            "Como trocar o plano de um associado?",
            "Preciso configurar desconto automático no PIX"
        ],
        "resposta": [
            "Cadastro/Quadro Social/Associados > Adicionar.",
            "Ferramentas/Configurações/Impressoras > Recibos.",
            "Cadastro/Associados > Financeiro > Vencimento.",
            "Rotinas/Financeiras/Realizar Gerações Financeiras.",
            "Relatórios/Financeiros/Exportar > Integração contábil.",
            "Ferramentas/Atualizações > Verificar e instalar.",
            "Ferramentas/Usuários/Permissões > Editar.",
            "Atendimento > Segurança > Resetar senha.",
            "Cadastro/Associados > Dependentes > Adicionar.",
            "Financeiro > Cobradores > Definir padrão.",
            "Atividades > Parâmetros > Fila de espera.",
            "Quadro Social > Carteiras > Adicionar dependente.",
            "Cadastro/Associados > Plano > Trocar.",
            "Financeiro > Descontos > Configurar automática."
        ]
    },
    10: {
        "pergunta": [
            "Erro ao gerar PIX no aplicativo, pagamento falhou",
            "Sistema travado, não abre a tela de login",
            "Falha crítica no servidor, app fora do ar",
            "Catraca não libera acesso, erro no leitor",
            "Aplicativo não funciona, tela preta ao abrir",
            "Licença expirada, sistema bloqueado",
            "Erro de banco de dados, não baixa boletos",
            "Caiu a internet do clube, pagamentos falhando",
            "Impressora não imprime, erro ao enviar recibo",
            "Bug na geração automática de mensalidades",
            "PIX não baixa automaticamente, erro constante",
            "Erro geral: não consigo abrir o sistema Forza"
        ],
        "resposta": [
            "Verificar conexão com o banco e reprocessar.",
            "Reiniciar serviço e validar licenças.",
            "Intervenção no servidor e restauração imediata.",
            "Verificar cabeamento e reiniciar controlador.",
            "Atualizar app e limpar cache.",
            "Gerar nova chave e aplicar desbloqueio.",
            "Checar integridade e reexecutar baixa.",
            "Restaurar conexão e reprocessar PIX.",
            "Reinstalar driver e testar fila.",
            "Aplicar patch e regenerar referência.",
            "Checar integração e reprocessar baixa.",
            "Validar servidor, licenças e serviços."
        ]
    }
}

AUG_SYNONYMS_5 = {
    "cadastrar": ["registrar", "incluir", "adicionar"],
    "associado": ["sócio", "membro"],
    "impressora": ["printer", "equipamento de impressão"],
    "recibos": ["comprovantes", "recibos de pagamento"],
    "mensalidade": ["mensalidades", "cobrança mensal"],
    "gerar": ["emitir", "criar", "processar"],
    "coletivas": ["em lote", "coletivamente"],
    "contabilidade": ["contador", "escritório contábil"],
    "atualização": ["update", "nova versão"],
    "permissões": ["acessos", "autorização"],
    "resetar": ["redefinir", "trocar"],
    "dependentes": ["dependente", "familiares"],
    "cobrador": ["responsável pela cobrança", "carteira"],
    "fila": ["lista de espera", "espera"],
    "carteira": ["cartão", "identificação"],
    "plano": ["categoria", "modalidade"],
    "desconto": ["abatimento", "redução"]
}

# Variações adicionais solicitadas para classe 5 (pagamentos, cadastro, mensalidades, políticas do clube)
EXTRA_SYNONYMS_5 = {
    "pagamento": ["pagamentos", "quitação", "cobrança", "baixa"],
    "cadastro": ["registro", "cadastros", "atualização cadastral"],
    "política": ["políticas do clube", "normas", "regulamento"],
    "politica": ["politicas do clube", "normas", "regulamento"],
}

AUG_SYNONYMS_5.update(EXTRA_SYNONYMS_5)

# Novos templates focados em classe 5 com variações específicas
EXTRA_QUESTIONS_5 = [
    "Como consultar políticas do clube sobre cancelamento e reembolso?",
    "Como ajustar política de mensalidades e multas no sistema?",
    "Como configurar regras de pagamento automático e baixa?",
    "Como atualizar cadastro completo de um associado (documentos, endereço)?",
    "Como revisar e publicar regulamento interno do clube no sistema?",
]

EXTRA_ANSWERS_5 = [
    "Ferramentas/Documentação > Políticas do Clube.",
    "Financeiro > Parâmetros > Mensalidades e Multas.",
    "Financeiro > Automatizações > Pagamento e Baixa.",
    "Cadastro/Associados > Documentos e Endereço > Editar.",
    "Ferramentas/Regulamentos > Publicar e versionar.",
]

TEMPLATES[5]["pergunta"].extend(EXTRA_QUESTIONS_5)
TEMPLATES[5]["resposta"].extend(EXTRA_ANSWERS_5)

FILLERS_NEUTRAL = [
    "por favor", "se possível", "gostaria de", "preciso", "poderiam",
    "eu queria", "tem como", "me avisem", "passo a passo", "detalhado"
]

def drop_accents(text: str) -> str:
    replacements = {
        "á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u",
        "â": "a", "ê": "e", "ô": "o", "ã": "a", "õ": "o",
        "ç": "c"
    }
    return "".join(replacements.get(ch, ch) for ch in text)

def augment_text_5(text: str, level: float = 0.4) -> str:
    words = text.split()
    out = []
    for w in words:
        base = w.lower()
        replaced = False
        for k, syns in AUG_SYNONYMS_5.items():
            if k in base and random.random() < level:
                out.append(random.choice(syns))
                replaced = True
                break
        if not replaced:
            # pequena chance de remover acento
            if random.random() < 0.15:
                out.append(drop_accents(w))
            else:
                out.append(w)
    # inserir filler neutro em algum ponto
    if len(out) > 3 and random.random() < 0.5:
        pos = random.randint(1, len(out)-1)
        out.insert(pos, random.choice(FILLERS_NEUTRAL))
    return " ".join(out)

def synth_record(idx: int, prio: int, augment: bool = False) -> dict:
    pergunta = random.choice(TEMPLATES[prio]["pergunta"])  # noqa
    resposta = random.choice(TEMPLATES[prio]["resposta"])  # noqa
    if augment and prio == 5:
        pergunta = augment_text_5(pergunta)
        resposta = augment_text_5(resposta, level=0.2)
    club = random.choice(CLUBS)
    ticket_id = f"synthetic-{prio}"
    motivo = pergunta
    ts = datetime(2025, 10, 27, 9, 0, 0) + timedelta(seconds=idx)
    return {
        "id": idx,
        "ticket_id": ticket_id,
        "pergunta": pergunta,
        "resposta": resposta,
        "agente_discord_id": "000000000000000000",
        "agente_discord_tag": "synthetic",
        "imagem_url": None,
        "imagem_cliente": None,
        "imagem_suporte": None,
        "imagem": None,
        "club_name": club,
        "motivo": motivo,
        "prioridade": prio,
        "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
    }

def main():
    parser = argparse.ArgumentParser(description="Gerar dados sintéticos balanceados para suporte QnA")
    parser.add_argument("--n", type=int, default=1000, help="Quantidade de registros sintéticos a gerar")
    parser.add_argument("--out", type=str, default="data/support_data_suporte_qna.json", help="Arquivo JSON a ser atualizado")
    parser.add_argument("--focus", type=int, choices=[1,5,10], default=None, help="Gerar apenas para uma prioridade específica")
    parser.add_argument("--augment", action="store_true", help="Ativar variação lexical para classe 5")
    parser.add_argument("--seed", type=int, default=42, help="Semente para reproducibilidade")
    args = parser.parse_args()

    random.seed(args.seed)
    out_path = Path(args.out)
    if not out_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {out_path}")

    with out_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Determina próximo id
    max_id = max([item.get("id", 0) for item in data]) if data else 0

    # Distribuição balanceada ou foco em uma classe
    n_total = args.n
    if args.focus is None:
        n_1 = n_total // 3
        n_5 = n_total // 3
        n_10 = n_total - n_1 - n_5
        plan = [(1, n_1), (5, n_5), (10, n_10)]
    else:
        plan = [(args.focus, n_total)]

    records = []
    idx_counter = max_id + 1

    for prio, qty in plan:
        for _ in range(qty):
            records.append(synth_record(idx_counter, prio, augment=args.augment))
            idx_counter += 1

    data.extend(records)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # Resumo
    counts = {1: 0, 5: 0, 10: 0}
    for r in records:
        counts[r["prioridade"]] += 1

    print(f"Gerados {len(records)} registros sintéticos")
    print(f"Distribuição: classe 1 = {counts[1]}, classe 5 = {counts[5]}, classe 10 = {counts[10]}")
    if args.augment:
        print("Augment ativo para classe 5 (variação lexical e ruído leve)")
    print(f"Arquivo atualizado: {out_path}")

if __name__ == "__main__":
    main()