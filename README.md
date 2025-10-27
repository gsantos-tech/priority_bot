# Priority Bot – Classificação de Prioridade de Chamados (1, 5, 10)

Sistema de triagem automática de chamados em português, com foco em identificar e priorizar corretamente urgências (classe 10) sem inflar indevidamente consultas informacionais (classe 5). O projeto inclui regras robustas de urgência, vetorização por palavras e caracteres, calibração de probabilidades e relatórios em PDF.

## Principais Recursos
- Regras de urgência (regex) cobrindo linguagem formal e coloquial: "não está funcionando", "tá travando", "não tá abrindo", "não abre", "não baixa boleto", "tela preta", "fora do ar", "licença expirada", "catraca".
- Vetorização híbrida com `FeatureUnion`: TF‑IDF de palavras e TF‑IDF de caracteres `(3,5)` para robustez a typos.
- Classificador calibrado: `CalibratedClassifierCV(LogisticRegression multinomial, class_weight='balanced')` com `GridSearchCV`.
- Limiar configurável para promoção à classe 10 via probabilidade calibrada (varredura: `0.65`, `0.70`, `0.75`).
- Relatórios automáticos (PDF) com métricas e matriz de confusão.

## Estrutura do Projeto
- `main.py` → pipeline (limpeza, TF‑IDF palavra+caracter, grid, regras de urgência, calibração, varredura de limiar, PDFs).
- `data/support_data_suporte_qna.json` → dataset em JSON.
- `scripts/generate_synthetic_data.py` → gerador de dados sintéticos.
- `models/best_model.pkl` → pipeline treinada.
- `reports/*.pdf` e `reports/*confusion_matrix*.png` → relatórios e matrizes por limiar (`t65`, `t70`, `t75`).

## Como Reproduzir
1. Requisitos: Python 3.11.
2. Instale dependências:
   - `pip install pandas scikit-learn matplotlib seaborn reportlab`
3. (Opcional) Gerar dados “difíceis” adicionais para melhorar separação 5 vs 10:
   - Classe 5 (hard negatives):
     - `python scripts/generate_synthetic_data.py --n 1500 --out data/support_data_suporte_qna.json --focus 5 --augment --seed 51`
   - Classe 10 (hard positives):
     - `python scripts/generate_synthetic_data.py --n 300 --out data/support_data_suporte_qna.json --focus 10 --augment --seed 52`
4. Treinar e gerar relatórios com varredura de limiar:
   - `python main.py`
5. Artefatos gerados (exemplos):
   - `reports/priority_bot_report_t65.pdf`, `reports/confusion_matrix_t65.png`
   - `reports/priority_bot_report_t70.pdf`, `reports/confusion_matrix_t70.png`
   - `reports/priority_bot_report_t75.pdf`, `reports/confusion_matrix_t75.png`
   - Modelo salvo: `models/best_model.pkl`

## Uso (Predição)
- Em Python:
  - `from main import predict_priority`
  - `predict_priority("Não tá abrindo o app, fica tela preta")  # esperado: 10`
  - `predict_priority("Como configuro a impressora de recibos?")  # esperado: 5`

## Metodologia
- Limpeza: lowercase, remoção de pontuação, stopwords PT‑BR, tokens com `len>2`.
- Representação: TF‑IDF palavra (1–2/1–3) com acentos normalizados e `sublinear_tf`; TF‑IDF de caracteres `(3,5)`.
- Modelo: `LogisticRegression` multinomial calibrado (`sigmoid`, `cv=3`).
- Busca: `GridSearchCV` com `StratifiedKFold(n=5)` otimizando `f1_macro`.
- Regras de segurança: promoção à `10` quando regex de falha casa; sem falha explícita, promoção se `P(10) ≥ limiar` (calibrada).

## Métricas e Relatórios
- As métricas por classe e matrizes de confusão estão nos PDFs gerados em `reports/` para cada limiar testado (`t65`, `t70`, `t75`).
- Em cenários recentes com dados ampliados, o sistema apresentou `accuracy ≈ 0.99` e `macro F1 ≈ 0.98`, com forte recall para a classe 10 mantendo boa precisão. Consulte os PDFs para números exatos por limiar.

## Diretrizes de Rotulagem (reduzir ambiguidade 5 vs 10)
- Classe 10: há falha/erro/indisponibilidade/bloqueio (ex.: "erro", "travando", "não abre", "fora do ar", "licença expirada").
- Classe 5: informação/procedimento/configuração/pagamento sem falha explícita.
- Ambíguos: se houver termo de falha + objetivo informacional, prevalece 10; sem falha, prevalece 5.

## Roadmap
- Crescer dataset com exemplos reais rotulados, com ênfase na fronteira 5 ↔ 10.
- Ajuste fino do limiar (0.65–0.75) conforme preferência de recall vs precision em 10.
- Testar embeddings PT (sentence-transformers) e, futuramente, fine‑tuning BERTimbau.

## Licença
- MIT License (vide `LICENSE`).

Autor: Gabriel
