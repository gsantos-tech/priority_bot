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
# 3. Vetorização TF-IDF
# =========================
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["clean_text"])
y = df["prioridade"]

# =========================
# 4. Divisão treino/teste
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# =========================
# 5. Treinar modelos
# =========================
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=200, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")
    results[name] = {"accuracy": acc, "f1_macro": f1}
    print(f"\n{name}")
    print(classification_report(y_test, preds))

# =========================
# 6. Escolher melhor modelo
# =========================
best_model_name = max(results, key=lambda x: results[x]["f1_macro"])
best_model = models[best_model_name]
print(f"\n✅ Melhor modelo: {best_model_name}")

# =========================
# 7. Salvar modelo
# =========================
with open("models/best_model.pkl", "wb") as f:
    pickle.dump((best_model, vectorizer), f)

print("💾 Modelo salvo em models/best_model.pkl")

# =========================
# 8. Função de predição
# =========================
def predict_priority(texto):
    with open("models/best_model.pkl", "rb") as f:
        model, vec = pickle.load(f)
    clean = clean_text(texto)
    X_new = vec.transform([clean])
    return model.predict(X_new)[0]

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
# 10. Matriz de confusão do melhor modelo
# =========================
y_pred_best = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=sorted(y.unique()),
            yticklabels=sorted(y.unique()))
plt.xlabel("Previsto")
plt.ylabel("Real")
plt.title(f"Matriz de Confusão - {best_model_name}")
plt.show()
