import json
import argparse
import random
from pathlib import Path


def build_balanced_subset(records, total=1000, seed=42):
    random.seed(seed)
    by_class = {1: [], 5: [], 10: []}
    for r in records:
        pr = r.get("prioridade")
        if pr in by_class:
            by_class[pr].append(r)

    # Distribuição balanceada: partes iguais; ajustar resto na classe 10
    base = total // 3
    remainder = total - base*3
    target = {1: base, 5: base, 10: base + remainder}

    subset = []
    for cls, tgt in target.items():
        pool = by_class[cls]
        if len(pool) < tgt:
            raise ValueError(f"Classe {cls} possui {len(pool)} exemplos, menos que o alvo {tgt}.")
        subset.extend(random.sample(pool, tgt))

    # Reordena por id para estabilidade, se houver
    subset.sort(key=lambda x: x.get("id", 0))
    return subset


def build_proportional_subset(records, total=1000, seed=42):
    random.seed(seed)
    by_class = {1: [], 5: [], 10: []}
    for r in records:
        pr = r.get("prioridade")
        if pr in by_class:
            by_class[pr].append(r)

    n1, n5, n10 = len(by_class[1]), len(by_class[5]), len(by_class[10])
    n_tot = n1 + n5 + n10
    # Alvos proporcionais às quantidades originais
    target = {
        1: max(1, round(total * n1 / n_tot)),
        5: max(1, round(total * n5 / n_tot)),
        10: max(1, round(total * n10 / n_tot)),
    }
    # Ajusta para somar exatamente 'total'
    diff = total - sum(target.values())
    if diff != 0:
        # Corrige na classe 5 por ser a maior
        target[5] = target[5] + diff

    subset = []
    for cls, tgt in target.items():
        pool = by_class[cls]
        if len(pool) < tgt:
            raise ValueError(f"Classe {cls} possui {len(pool)} exemplos, menos que o alvo {tgt}.")
        subset.extend(random.sample(pool, tgt))

    subset.sort(key=lambda x: x.get("id", 0))
    return subset


def main():
    parser = argparse.ArgumentParser(description="Criar subset de dados para treinamento")
    parser.add_argument("--in", dest="in_path", type=str, default="data/support_data_suporte_qna.json",
                        help="Arquivo JSON de entrada")
    parser.add_argument("--out", dest="out_path", type=str, default="data/support_data_subset.json",
                        help="Arquivo JSON de saída")
    parser.add_argument("--total", dest="total", type=int, default=1000, help="Tamanho total do subset")
    parser.add_argument("--seed", dest="seed", type=int, default=42, help="Semente para reproducibilidade")
    parser.add_argument("--strategy", dest="strategy", choices=["balanced", "proportional"], default="balanced",
                        help="Estratégia de seleção: balanceada ou proporcional")
    args = parser.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Arquivo de entrada não encontrado: {in_path}")

    with in_path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    if args.strategy == "balanced":
        subset = build_balanced_subset(records, total=args.total, seed=args.seed)
    else:
        subset = build_proportional_subset(records, total=args.total, seed=args.seed)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(subset, f, ensure_ascii=False, indent=2)

    print(f"Subset criado: {len(subset)} exemplos → {out_path}")


if __name__ == "__main__":
    main()