# Priority Bot – Classificação de Prioridade de Chamados

Este projeto desenvolve um classificador automático de prioridade de chamados (1, 5, 10) para suporte técnico. O foco é identificar com alta confiabilidade casos críticos (prioridade 10) e organizar o atendimento com base em evidências de texto.

## Objetivos
- Automatizar a triagem de chamados por prioridade.
- Reduzir falsos negativos em urgências (classe 10).
- Entregar uma pipeline reprodutível e explicável (G1 – Ciência de Dados).

## Dados
- Fonte: `data/support_data_suporte_qna.json` (português, domínio de suporte).
- Campos chave: `pergunta`, `resposta`, `prioridade` (1/5/10), metadados.
- Observação: há desbalanceamento de classes e a classe 5 é mais ambígua.

## Metodologia
- Pré-processamento: lower case, remoção de pontuação, stopwords PT‑BR, tokens com len>2.
- Representação: TF‑IDF com n‑grams (até bigrams/trigrams), acentos normalizados (`strip_accents='unicode'`), `sublinear_tf`.
- Modelos comparados: `LogisticRegression` e `LinearSVC` com `class_weight='balanced'`.
- Validação: `StratifiedKFold (n=5)` + `GridSearchCV` otimizando hiperparâmetros. Métrica: `f1_macro`.
- Pós‑processamento: regra pragmática para elevar a prioridade a `10` quando aparecem termos críticos (ex.: "erro", "travado", "PIX", "catraca", "não funciona").

## Resultados
- Pipeline final (exemplo recente): `LogisticRegression (C=1, solver='saga')` + TF‑IDF `(1,2)`, `min_df=2`, `max_df=0.95`, acentos e `sublinear_tf`.
- Métricas no conjunto de teste (com regras):
  - `accuracy ≈ 0.31`
  - Classe `10`: `recall ≈ 0.67`, `f1 ≈ 0.47` (melhor foco em urgências).
- Interpretação: ganhos limitados pela quantidade/qualidade de dados e ambiguidade da classe 5; melhorias concentram-se em reduzir erro onde importa (prioridade 10).

## Estrutura do Projeto
- `main.py` → pipeline completa: limpeza, TF‑IDF, Grid Search, avaliação, matriz de confusão, testes manuais e função `predict_priority`.
- `models/best_model.pkl` → pipeline otimizada salva (TF‑IDF + classificador).
- `data/support_data_suporte_qna.json` → dataset.

## Como Rodar
1. Requisitos: Python 3.11.
2. Instale dependências:
   - `pip install pandas scikit-learn matplotlib seaborn`
3. Execute o treinamento e avaliação:
   - `python main.py`
4. Usar a função de predição em Python:
   - `from main import predict_priority; predict_priority("Não abre o sistema Forza")`

## Próximos Passos
- Expandir o dataset e definir um guia de rotulagem para distinguir melhor `5` vs `10`.
- Incluir features manuais (sinais binários de termos críticos) combinadas ao TF‑IDF.
- Testar modelos de linguagem em PT‑BR (ex.: BERTimbau) com fine‑tuning.
- Calibração e regras de segurança para casos sensíveis.

## Considerações Éticas
- Prioridade incorreta pode atrasar atendimentos críticos; por isso há regras de reforço para casos urgentes. Ainda assim, recomenda‑se supervisão humana nos primeiros ciclos.

## Licença
- MIT License (vide `LICENSE`).

Autor: Gabriel
