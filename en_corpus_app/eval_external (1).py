"""
eval_external.py: внешняя проверка на другом корпусе.

Обучение: Ott, положительные отзывы (те же модели, что в демо, плюс контрольные).
Тест: весь внешний корпус (например, Four-City, Li, Ott & Cardie 2013, или датасет с Kaggle).
Так проверяется, переносятся ли результаты на другие отели, города и авторов.

Настройка: поправьте блок CONFIG ниже, затем  python eval_external.py

Режимы данных:
  1) два файла: FAKE_PATH (ложные) и TRUE_PATH (правдивые, если есть);
  2) один файл с меткой: COMBINED_PATH, LABEL_COL, FAKE_VALUES.
Файл может быть CSV/TSV (колонка с текстом определяется автоматически как колонка с самыми
длинными строками, либо задайте TEXT_COL) или TXT (один отзыв на строку).
Если правдивых отзывов нет, считаются только доля распознанных ложных и средняя вероятность
(AUC и точность в этом случае посчитать нельзя).
"""
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import ott_pipeline_v2 as v2
from ott_pipeline_v2 import EXT_FEATURES, get_features, load_data
from train_demo_model import CORE_FEATURES

# ---------------- CONFIG ----------------
FAKE_PATH = "data/four_city/fake.csv"
TRUE_PATH = None                      # например, "data/four_city/true.csv"
COMBINED_PATH = None                  # вместо двух файлов: один файл с колонкой метки
LABEL_COL = None
FAKE_VALUES = {"fake", "deceptive", "1", "cg", "true_fake"}   # значения метки, означающие «ложь»
TEXT_COL = None                       # None: определить автоматически
TRAIN_POLARITY = "positive"           # внешние отзывы положительные, обучаем на положительных Ott
EXTERNAL_CACHE = "cache/external_features.csv"
RESULTS_PATH = "results/external_eval.csv"
N_BOOT = 1000
# ----------------------------------------


def read_table(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".csv", ".tsv"):
        sep = "\t" if ext == ".tsv" else None
        try:
            return pd.read_csv(path, sep=sep, engine="python")
        except UnicodeDecodeError:
            return pd.read_csv(path, sep=sep, engine="python", encoding="latin-1")
    with open(path, encoding="utf-8", errors="replace") as f:
        return pd.DataFrame({"text": [ln.strip() for ln in f if ln.strip()]})


def detect_text_col(table):
    if TEXT_COL:
        return TEXT_COL
    best, best_len = None, -1.0
    for col in table.columns:
        series = table[col]
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            mean_len = table[col].astype(str).str.len().mean()
            if mean_len > best_len:
                best, best_len = col, mean_len
    if best is None:
        raise SystemExit(f"Не нашёл текстовую колонку среди {list(table.columns)}. Задайте TEXT_COL.")
    print(f"  текстовая колонка: '{best}' (средняя длина {best_len:.0f} символов)")
    return best


def load_external():
    if COMBINED_PATH:
        if LABEL_COL is None:
            raise SystemExit("Для COMBINED_PATH задайте LABEL_COL и FAKE_VALUES.")
        t = read_table(COMBINED_PATH)
        print(f"Колонки файла: {list(t.columns)}")
        fake_values = {v.lower() for v in FAKE_VALUES}
        label = t[LABEL_COL].astype(str).str.strip().str.lower().isin(fake_values).astype(int).values
        out = pd.DataFrame({"text": t[detect_text_col(t)].astype(str).values, "label": label})
    else:
        parts = []
        for path, label in ((FAKE_PATH, 1), (TRUE_PATH, 0)):
            if path:
                t = read_table(path)
                print(f"{path}: колонки {list(t.columns)}")
                parts.append(pd.DataFrame({"text": t[detect_text_col(t)].astype(str).values, "label": label}))
        out = pd.concat(parts, ignore_index=True)
    out["text"] = out["text"].str.strip()
    return out[out["text"].str.len() > 0].reset_index(drop=True)


def external_features(ext_df):
    """Признаки внешнего корпуса с отдельным кэшем (кэш Ott не затрагивается)."""
    old = v2.CACHE_PATH
    v2.CACHE_PATH = EXTERNAL_CACHE
    try:
        return v2.get_features(ext_df)
    finally:
        v2.CACHE_PATH = old


def auc_ci(y, p, n_boot=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) == 2:
            aucs.append(roc_auc_score(y[idx], p[idx]))
    return np.percentile(aucs, [2.5, 97.5])


def main():
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(EXTERNAL_CACHE), exist_ok=True)

    # обучающая выборка: Ott (признаки берём из кэша v2 по полному корпусу)
    ott_all = load_data()
    X_ott_all = get_features(ott_all)
    mask = (ott_all["polarity"] == TRAIN_POLARITY).values
    X_tr = X_ott_all[mask].reset_index(drop=True)
    y_tr = ott_all.loc[mask, "label"].values
    texts_tr = ott_all.loc[mask, "text"].astype(str).values

    # внешний корпус
    ext = load_external()
    X_ext = external_features(ext)
    y_ext, texts_ext = ext["label"].values, ext["text"].values
    has_both = len(np.unique(y_ext)) == 2
    print(f"\nВнешний корпус: {len(ext)} текстов, ложных {int(y_ext.sum())}, правдивых {int((1 - y_ext).sum())}")

    # диагностика сдвига распределений
    diag_cols = ["Total_Words", "First_Sing_%", "Clauses"]
    rows = []
    for corpus, X, y in (("Ott (обучение)", X_tr, y_tr), ("внешний", X_ext, y_ext)):
        for lab, name in ((0, "правдивые"), (1, "ложные")):
            if (y == lab).any():
                sub = X[y == lab]
                row = {"корпус": corpus, "класс": name, "n": int((y == lab).sum())}
                row.update({c: sub[c].mean() for c in diag_cols})
                row["доля текстов без «I/me/my»"] = (sub["First_Sing_%"] == 0).mean()
                rows.append(row)
    print("\n--- Сдвиг распределений: средние значения ---")
    print(pd.DataFrame(rows).round(2).to_string(index=False))

    lr = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    bow = lambda: make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                                LogisticRegression(max_iter=2000, C=10))
    specs = {
        "только длина": (["Total_Words"], lr),
        "ext (18 признаков)": (EXT_FEATURES, lr),
        "core (7 признаков)": (CORE_FEATURES, lr),
        "BoW TF-IDF": (None, bow),
    }

    results = []
    for name, (cols, factory) in specs.items():
        inp_tr = texts_tr if cols is None else X_tr[cols].values
        inp_ext = texts_ext if cols is None else X_ext[cols].values
        p = factory().fit(inp_tr, y_tr).predict_proba(inp_ext)[:, 1]
        row = {"model": name, "mean_p_fake": p.mean(), "share_flagged_fake(p>0.5)": (p > 0.5).mean()}
        if has_both:
            lo, hi = auc_ci(y_ext, p)
            row.update({"AUC": roc_auc_score(y_ext, p), "CI95_low": lo, "CI95_high": hi,
                        "accuracy": accuracy_score(y_ext, p > 0.5),
                        "recall_fake": (p[y_ext == 1] > 0.5).mean(),
                        "specificity": (p[y_ext == 0] <= 0.5).mean()})
        results.append(row)

    res = pd.DataFrame(results)
    res.to_csv(RESULTS_PATH, index=False)
    print("\n--- Обучение на Ott (positive), тест на внешнем корпусе ---")
    print(res.round(3).to_string(index=False))
    if not has_both:
        print("\nПравдивых отзывов нет: AUC и точность не считаются. Добавьте TRUE_PATH.")
    print(f"\nТаблица сохранена: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
