# Priority Bot - Classificação de Prioridade de Chamados

Este projeto cria um **classificador automático de prioridade de chamados** para ser integrado em um bot.
A partir de uma pergunta enviada pelo usuário, o sistema identifica se a prioridade do atendimento deve ser **baixa, média ou alta**.

---

## 📂 Estrutura
- `data/support_data_suporte_qna.json` → Dataset original.
- `models/best_model.pkl` → Modelo treinado final.
- `src/preprocess.py` → Pré-processamento de texto.
- `src/train_models.py` → Treinamento e comparação de modelos.
- `src/evaluate_models.py` → Avaliação de performance.
- `src/predict.py` → Função para prever prioridade em tempo real.
- `src/utils.py` → Funções auxiliares.

---

## 🚀 Como Rodar

1. Instale dependências:
   ```bash
   pip install -r requirements.txt
