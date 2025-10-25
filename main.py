# main.py
import pandas as pd
import re
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import LinearSVC

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
    "não","sim","há","vou","está","etc"
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
# Usa Pipeline + GridSearchCV para otimizar TF-IDF e comparar Logistic Regression vs LinearSVC
X_text = df["clean_text"]
y = df["prioridade"]

X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text, y, test_size=0.3, stratify=y, random_state=42
)

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression(max_iter=500, class_weight="balanced"))
])

# Leaner param grid for faster convergence
param_grid = [
    {
        "tfidf__ngram_range": [(1, 2), (1, 3)],
        "tfidf__min_df": [1, 2],
        "tfidf__max_df": [0.95],
        "tfidf__strip_accents": ["unicode"],
        "tfidf__sublinear_tf": [True],
        "clf": [LogisticRegression(max_iter=500, class_weight="balanced")],
        "clf__C": [0.5, 1, 2],
        "clf__solver": ["liblinear", "saga"],
    },
    {
        "tfidf__ngram_range": [(1, 2), (1, 3)],
        "tfidf__min_df": [1, 2],
        "tfidf__max_df": [0.95],
        "tfidf__strip_accents": ["unicode"],
        "tfidf__sublinear_tf": [True],
        "clf": [LinearSVC(class_weight="balanced")],
        "clf__C": [0.5, 1, 2],
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
def apply_rules(clean_text_value, pred):
    tokens = set(clean_text_value.split())
    keywords_high = {
        "erro","travado","trava","falha","bug","nao","não","funciona","abrir","abre","bloqueado",
        "pix","catraca","pagamento","baixar","baixa","aplicativo","app","servidor","licenca","licença"
    }
    if tokens & keywords_high:
        return 10
    return pred

search.fit(X_train_text, y_train)

best_pipeline = search.best_estimator_
best_clf_name = type(best_pipeline.named_steps["clf"]).__name__
print(f"\n✅ Melhor pipeline: {best_clf_name} (GridSearchCV)")
print(f"Melhores parâmetros: {search.best_params_}")

# Avaliação no conjunto de teste
# Aplica reforço de regras na predição do conjunto de teste
y_pred_raw = best_pipeline.predict(X_test_text)
y_pred = [apply_rules(text, p) for text, p in zip(X_test_text, y_pred_raw)]
print("\nRelatório de classificação (teste):")
print(classification_report(y_test, y_pred, zero_division=0))

# =========================
# 8. Função de predição
# =========================
# Atualiza para carregar o pipeline e aplicar diretamente

def predict_priority(texto):
    with open("models/best_model.pkl", "rb") as f:
        pipeline_loaded = pickle.load(f)
    clean = clean_text(texto)
    pred = pipeline_loaded.predict([clean])[0]
    return apply_rules(clean, pred)

# =========================
# 10. Matriz de confusão do melhor modelo
# =========================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=sorted(y.unique()),
            yticklabels=sorted(y.unique()))
plt.xlabel("Previsto")
plt.ylabel("Real")
plt.title(f"Matriz de Confusão - {best_clf_name} (GridSearchCV)")
plt.show()

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
