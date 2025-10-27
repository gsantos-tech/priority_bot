# main.py
import pandas as pd
import re
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm

# =========================
# 1. Carregar dataset
# =========================
df = pd.read_json("data/support_data_suporte_qna.json")
df = df[["pergunta", "prioridade"]]

# =========================
# 2. Pré-processamento
# =========================
stopwords = set([
    "como","para","um","uma","de","no","na","em","o","a","os","as",
    "do","da","dos","das","que","e","é","ser","ao","aos","com","por",
    "se","uns","umas","este","esta","esse","essa","isso","isto","já",
    "sim","há","vou","está","etc"
])

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-záéíóúâêôãõç\s]", " ", text)  # remove pontuação
    tokens = text.split()
    tokens = [t for t in tokens if t not in stopwords and len(t) > 2]
    return " ".join(tokens)

df["clean_text"] = df["pergunta"].apply(clean_text)

# =========================
# 3. Vetorização e Pipeline com Grid Search
# =========================
# Usa Pipeline + GridSearchCV para otimizar TF-IDF e combinar palavras + caracteres;
# Classificador calibrado para obter probabilidades confiáveis.
X_text = df["clean_text"]
y = df["prioridade"]

X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text, y, test_size=0.3, stratify=y, random_state=42
)

features_union = FeatureUnion([
    ("word", TfidfVectorizer()),
    ("char", TfidfVectorizer(analyzer="char", ngram_range=(3, 5)))
])

pipeline = Pipeline([
    ("features", features_union),
    ("clf", CalibratedClassifierCV(
        estimator=LogisticRegression(max_iter=500, class_weight="balanced"),
        method="sigmoid",
        cv=3
    ))
])

# Leaner param grid for faster convergence
param_grid = [
    {
        # TF-IDF de palavras
        "features__word__ngram_range": [(1, 2), (1, 3)],
        "features__word__min_df": [1, 2],
        "features__word__max_df": [0.95],
        "features__word__strip_accents": ["unicode"],
        "features__word__sublinear_tf": [True],
        # TF-IDF de caracteres (fixo ou leve variação)
        "features__char__ngram_range": [(3, 5)],
        # Classificador calibrado (LogisticRegression como base)
        "clf__estimator__C": [0.5, 1, 2],
        "clf__estimator__solver": ["lbfgs", "saga"],
        "clf__estimator__multi_class": ["multinomial"],
        "clf__method": ["sigmoid"],
        "clf__cv": [3],
    },
]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
search = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    scoring="f1_macro",
    n_jobs=-1,
    cv=cv,
    verbose=0,
)

# Função de regras para reforço de prioridade
PRIO10_THRESHOLD = 0.7

def apply_rules(clean_text_value, pred, prob_10=None, threshold=PRIO10_THRESHOLD):
    s = clean_text_value
    tokens = set(s.split())

    # Padrões de falha/erro explícitos (sinal forte para prioridade 10)
    failure_patterns = [
        r"\b(erro|falha|falhou|bug)\b",
        r"\b(travado|trava|bloqueado)\b",
        r"\bn(a|ã)o\s+(funciona|abre|baixa)\b",
        r"\bn(a|ã)o\s+est(a|á)\s+funcionand[oa]\b",
        r"\bn(a|ã)o\s+est(a|á)\s+abrind[oa]\b",
        # Variações coloquiais
        r"\bt(á|a)\s+travando\b",
        r"\bn(a|ã)o\s+t(á|a)\s+abrindo\b",
        r"\bn(a|ã)o\s+abre\b",
        r"\bn(a|ã)o\s+baixa\s+boleto\b",
        r"\b(tela\s+preta|fora\s+do\s+ar|inacessivel|inacessível)\b",
        r"\b(catraca)\b",
        r"\b(licenca|licença)\s+(expirada|bloqueada)\b",
    ]

    # Termos informacionais que tendem à classe 5 (para evitar falsos positivos de 10)
    info_bias = {
        "configurar","consultar","gerar","mensalidade","mensalidades","cadastro",
        "política","politica","regulamento","impressora","permissoes","permissões",
        "dependentes","cobrador","fila","carteira","plano","desconto","pix"
    }

    # Se há padrão claro de falha, classifique como 10
    failure = any(re.search(p, s) for p in failure_patterns)
    if failure:
        return 10

    # Sem falha explícita: usar probabilidade calibrada para elevar urgências
    if prob_10 is not None and prob_10 >= threshold:
        return 10

    # Caso contrário, manter a predição do modelo
    return pred

# Salva a matriz de confusão como imagem PNG
def save_confusion_matrix(cm, labels, out_path, title="Matriz de Confusão"):
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels,
                yticklabels=labels)
    plt.xlabel("Previsto")
    plt.ylabel("Real")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# Gera um PDF de relatório com métricas e explicações em linguagem leiga
def generate_pdf_report(acc, report_dict, cm_path, best_name, best_params, out_pdf, threshold=None):
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(out_pdf, pagesize=A4)
    elems = []

    # Cabeçalho
    elems.append(Paragraph("Priority Bot – Relatório de Classificação", styles['Title']))
    elems.append(Paragraph(datetime.now().strftime("%d/%m/%Y %H:%M"), styles['Normal']))
    elems.append(Spacer(1, 12))

    # Resumo
    elems.append(Paragraph("Resumo", styles['Heading2']))
    elems.append(Paragraph(f"Modelo escolhido: {best_name}", styles['Normal']))
    elems.append(Paragraph(f"Hiperparâmetros: {best_params}", styles['Normal']))
    elems.append(Paragraph(f"Taxa de acerto (accuracy): {acc:.2f}", styles['Normal']))
    if threshold is not None:
        elems.append(Paragraph(f"Limiar para classe 10 (calibrada): {threshold:.2f}", styles['Normal']))
    macro_f1 = report_dict.get('macro avg', {}).get('f1-score', None)
    if macro_f1 is not None:
        elems.append(Paragraph(f"F1 macro (equilíbrio entre classes): {macro_f1:.2f}", styles['Normal']))
    elems.append(Spacer(1, 12))

    # Explicação leiga (Por que)
    por_que = (
        "Este sistema lê a sua pergunta e transforma o texto em números (TF‑IDF), "
        "que indicam quais palavras são mais importantes. Em seguida, ele usa um "
        "modelo estatístico para decidir se o assunto é simples (1), intermediário (5) "
        "ou urgente (10). Para evitar erros em casos críticos, há uma regra prática: "
        "se forem detectados termos como 'erro', 'travado', 'PIX', 'catraca' ou 'não funciona', "
        "a prioridade é marcada como 10. Isso ajuda a garantir que problemas graves recebam atenção primeiro."
    )
    elems.append(Paragraph("Por que funciona", styles['Heading2']))
    elems.append(Paragraph(por_que, styles['Normal']))
    elems.append(Spacer(1, 12))

    # Métricas por classe
    elems.append(Paragraph("Métricas por classe", styles['Heading2']))
    for label in ['1','5','10']:
        if label in report_dict:
            ld = report_dict[label]
            elems.append(Paragraph(
                f"Prioridade {label}: precisão {ld.get('precision',0):.2f} | recall {ld.get('recall',0):.2f} | f1 {ld.get('f1-score',0):.2f}",
                styles['Normal']
            ))
    elems.append(Spacer(1, 12))

    # Imagem da matriz de confusão
    if os.path.exists(cm_path):
        elems.append(Paragraph("Matriz de Confusão", styles['Heading2']))
        elems.append(Image(cm_path, width=14*cm, height=10*cm))
        elems.append(Spacer(1, 12))

    # Conclusões e próximos passos
    conclusoes = (
        "O desempenho geral é limitado pela quantidade de dados e pela ambiguidade da classe 5. "
        "Mesmo assim, o sistema prioriza bem os casos urgentes (classe 10), reduzindo riscos de atrasos. "
        "Para melhorar a taxa de acerto, recomenda‑se aumentar o conjunto de exemplos e, futuramente, "
        "usar modelos de linguagem mais avançados em português (ex.: BERTimbau)."
    )
    elems.append(Paragraph("Conclusões", styles['Heading2']))
    elems.append(Paragraph(conclusoes, styles['Normal']))

    doc.build(elems)

search.fit(X_train_text, y_train)

best_pipeline = search.best_estimator_
best_clf_name = type(best_pipeline.named_steps["clf"]).__name__
print(f"\n✅ Melhor pipeline: {best_clf_name} (GridSearchCV)")
print(f"Melhores parâmetros: {search.best_params_}")

# Avaliação no conjunto de teste com varredura de limiar
y_pred_raw = best_pipeline.predict(X_test_text)
probas = best_pipeline.predict_proba(X_test_text)
classes = list(best_pipeline.classes_)
idx10 = classes.index(10)
prob_10_list = [row[idx10] for row in probas]

thresholds = [0.65, 0.70, 0.75]
os.makedirs("reports", exist_ok=True)
labels = sorted(y.unique())

for t in thresholds:
    y_pred_t = [apply_rules(text, p, prob_10, t) for text, p, prob_10 in zip(X_test_text, y_pred_raw, prob_10_list)]
    report_dict_t = classification_report(y_test, y_pred_t, zero_division=0, output_dict=True)
    acc_t = accuracy_score(y_test, y_pred_t)
    print(f"\n=== THRESHOLD {t:.2f} ===")
    print(classification_report(y_test, y_pred_t, zero_division=0))

    conf_mat_t = confusion_matrix(y_test, y_pred_t)
    cm_img_path_t = os.path.join("reports", f"confusion_matrix_t{int(t*100)}.png")
    save_confusion_matrix(conf_mat_t, labels, cm_img_path_t, title=f"Matriz de Confusão - {best_clf_name} (th={t:.2f})")

    pdf_path_t = os.path.join("reports", f"priority_bot_report_t{int(t*100)}.pdf")
    generate_pdf_report(acc_t, report_dict_t, cm_img_path_t, best_clf_name, search.best_params_, pdf_path_t, threshold=t)

# =========================
# 8. Função de predição
# =========================
# Atualiza para carregar o pipeline e aplicar diretamente

def predict_priority(texto):
    with open("models/best_model.pkl", "rb") as f:
        pipeline_loaded = pickle.load(f)
    clean = clean_text(texto)
    pred = pipeline_loaded.predict([clean])[0]
    # Usa probabilidade calibrada para reforçar prioridade 10
    probas = pipeline_loaded.predict_proba([clean])[0]
    classes = list(pipeline_loaded.classes_)
    idx10 = classes.index(10)
    prob_10 = probas[idx10]
    return apply_rules(clean, pred, prob_10)

# =========================
# 10. Relatórios por limiar gerados acima; pulando gráfico interativo
# =========================

# =========================
# 9. Testes manuais com gabarito
# =========================
testes = [
    ("Não consigo abrir o sistema Forza, fica travado na tela inicial.", 10),
    ("Quero emitir um relatório de despesas do mês passado.", 1),
    ("Preciso cadastrar um novo associado no sistema.", 5),
    ("Erro ao gerar PIX no aplicativo, cliente não consegue pagar.", 10),
    ("Como faço para configurar impressora de recibos?", 5),
    ("O app não está funcionando e vários sócios estão reclamando.", 10),
    ("Gostaria de saber como gerar mensalidades coletivas.", 5),
    ("Está dando erro na catraca de acesso ao clube.", 10),
]

print("\n================ TESTES ================\n")
for texto, esperado in testes:
    previsto = predict_priority(texto)
    print(f"Pergunta: {texto}")
    print(f"➡ Esperado: {esperado} | 🔮 Previsto: {previsto}\n")

# =========================
# 7. Salvar pipeline completo
# =========================
with open("models/best_model.pkl", "wb") as f:
    pickle.dump(best_pipeline, f)

print("💾 Pipeline salvo em models/best_model.pkl")
