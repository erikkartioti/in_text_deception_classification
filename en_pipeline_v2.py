"""
ott_pipeline_v2.py: признаки Хэнкока + Де Санто + LIWC-подобные, корпус Ott.

Что делает:
  1. Извлекает признаки (кэшируются в cache/) для всех текстов корпуса.
  2. Для положительных и отрицательных отзывов отдельно:
     - различия признаков между классами: Манн-Уитни, rank-biserial (размер эффекта),
       поправка Холма;
     - повторная кросс-валидация с делением по отелям (5 фолдов x 5 повторов),
       модели: только длина, 11 базовых признаков, расширенный набор, BoW (TF-IDF);
     - permutation importance на отложенных фолдах + коэффициенты логистической регрессии.
  3. Сравнивает направление эффектов у положительных и отрицательных отзывов.
  4. Проверка переноса: обучение на одной полярности, тест на другой.
  5. Все таблицы сохраняет в results/*.csv.

*добавить в requirements:
    pip install spacy catboost scikit-learn scipy pandas numpy
    python -m spacy download en_core_web_sm

Данные: data/deceptive-opinion.csv, колонки deceptive, hotel, polarity, text.

"""
import os
import hashlib
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from catboost import CatBoostClassifier

DATA_PATH = "C:/Users/snake/Desktop/Masters/Masters_en/data/deceptive-opinion.csv"
CACHE_PATH = "cache/ott_features_v2.csv"
RESULTS_DIR = "results"
N_SPLITS = 5
N_REPEATS = 5
SEED = 42

# ---------------------------------------------------------------
# 1. Лексиконы (стартовые наборы, леммы в нижнем регистре)
# ---------------------------------------------------------------
FIRST_SING = {"i", "me", "my", "mine", "myself"}
FIRST_PLUR = {"we", "us", "our", "ours", "ourselves"}
FIRST_PERSON = FIRST_SING | FIRST_PLUR
THIRD_PERSON = {"he", "she", "it", "they", "him", "her", "them", "his", "hers",
                "its", "their", "theirs"}
COG_MARKERS = {"think", "know", "believe", "consider", "realize", "understand",
               "wonder", "suppose", "guess", "feel", "decide", "remember", "expect"}
CAUSAL_MARKERS = {"because", "since", "therefore", "thus", "hence", "reason",
                  "cause", "result", "consequently", "so"}
EMO_POS = {"happy", "great", "wonderful", "amazing", "lovely", "perfect", "excellent",
           "fantastic", "beautiful", "comfortable", "friendly", "pleasant",
           "delightful", "superb", "nice", "good", "clean", "helpful", "spacious",
           "gorgeous", "fabulous"}
EMO_NEG = {"sad", "terrible", "awful", "horrible", "nasty", "disgusting", "dirty",
           "rude", "unpleasant", "disappointing", "bad", "noisy", "filthy", "poor",
           "uncomfortable", "worst"}
EMOTION = EMO_POS | EMO_NEG
NEGATION = {"not", "no", "never", "nothing", "none", "neither", "nor", "nobody",
            "nowhere", "cannot"}
HEDGE = {"maybe", "perhaps", "possibly", "probably", "might", "may", "could", "seem",
         "appear", "somewhat", "apparently", "likely", "somehow"}
EXCLUSIVE = {"but", "except", "however", "although", "though", "unless", "whereas",
             "instead", "yet", "rather", "besides"}
SENSORY = {"see", "hear", "smell", "taste", "touch", "look", "sound", "bright", "loud",
           "quiet", "soft", "warm", "cold", "view", "sight", "noise"}
SPATIAL = {"near", "nearby", "above", "below", "under", "inside", "outside", "behind",
           "between", "across", "around", "beside", "toward", "away", "floor",
           "street", "block", "downtown", "distance"}

BASE_FEATURES = [
    "Total_Words", "First_Person_%", "Third_Person_%", "Cognitive_Words_%",
    "Emotion_Words_%", "Causal_Words_%", "Tree_Depth", "Clauses",
    "Dependency_Distance", "Syntax_Density", "Lexical_TTR",
]
NEW_FEATURES = [
    "First_Sing_%", "First_Plur_%", "Emotion_Pos_%", "Emotion_Neg_%",
    "Negation_%", "Hedge_%", "Exclusive_%", "Sensory_%", "Spatial_%",
]
ALL_FEATURES = BASE_FEATURES + NEW_FEATURES
# в расширенном наборе объединённые признаки заменены их частями (без коллинеарности)
EXT_FEATURES = [f for f in ALL_FEATURES if f not in ("First_Person_%", "Emotion_Words_%")]


# ---------------------------------------------------------------
# 2. Признаки одного документа spaCy
# ---------------------------------------------------------------
def sentence_complexity(sent):
    """Глубина дерева, число клауз, средняя длина зависимости, плотность синтаксиса."""
    def walk(node, depth):
        children = list(node.children)
        if not children:
            return depth
        return max(walk(c, depth + 1) for c in children)

    roots = [t for t in sent if t.head == t]
    tree_depth = max(walk(r, 1) for r in roots) if roots else 1
    clauses = sum(1 for t in sent
                  if t.pos_ in ("VERB", "AUX") or t.dep_ in ("ccomp", "xcomp", "advcl", "relcl"))
    dists = [abs(t.i - t.head.i) for t in sent if not t.is_punct and t.head != t]
    avg_dist = float(np.mean(dists)) if dists else 0.0
    sent_len = len([t for t in sent if not t.is_punct])
    density = tree_depth / sent_len if sent_len > 0 else 0.0
    return tree_depth, clauses, avg_dist, density


def features_from_doc(doc):
    feats = {n: 0.0 for n in ALL_FEATURES}
    sentences = list(doc.sents)
    tokens = [t for t in doc if not t.is_punct]
    n = len(tokens)
    if n == 0 or not sentences:
        return feats

    counts = dict.fromkeys(
        ["fs", "fp", "third", "cog", "causal", "pos", "neg", "negation",
         "hedge", "excl", "sens", "spat"], 0)
    for t in doc:
        lemma = t.lemma_.lower()
        if t.pos_ == "PRON":
            if lemma in FIRST_SING:
                counts["fs"] += 1
            elif lemma in FIRST_PLUR:
                counts["fp"] += 1
            elif lemma in THIRD_PERSON:
                counts["third"] += 1
        if lemma in COG_MARKERS:
            counts["cog"] += 1
        if lemma in CAUSAL_MARKERS:
            counts["causal"] += 1
        if t.pos_ in ("ADJ", "ADV"):
            if lemma in EMO_POS:
                counts["pos"] += 1
            elif lemma in EMO_NEG:
                counts["neg"] += 1
        if lemma in NEGATION:
            counts["negation"] += 1
        if lemma in HEDGE:
            counts["hedge"] += 1
        if lemma in EXCLUSIVE:
            counts["excl"] += 1
        if lemma in SENSORY:
            counts["sens"] += 1
        if lemma in SPATIAL:
            counts["spat"] += 1

    pct = lambda c: c / n * 100
    cx = [sentence_complexity(s) for s in sentences]
    feats.update({
        "Total_Words": n,
        "First_Person_%": pct(counts["fs"] + counts["fp"]),
        "Third_Person_%": pct(counts["third"]),
        "Cognitive_Words_%": pct(counts["cog"]),
        "Emotion_Words_%": pct(counts["pos"] + counts["neg"]),
        "Causal_Words_%": pct(counts["causal"]),
        "Tree_Depth": np.mean([c[0] for c in cx]),
        "Clauses": np.mean([c[1] for c in cx]),
        "Dependency_Distance": np.mean([c[2] for c in cx]),
        "Syntax_Density": np.mean([c[3] for c in cx]),
        "Lexical_TTR": len({t.lemma_.lower() for t in tokens}) / n * 100,
        "First_Sing_%": pct(counts["fs"]),
        "First_Plur_%": pct(counts["fp"]),
        "Emotion_Pos_%": pct(counts["pos"]),
        "Emotion_Neg_%": pct(counts["neg"]),
        "Negation_%": pct(counts["negation"]),
        "Hedge_%": pct(counts["hedge"]),
        "Exclusive_%": pct(counts["excl"]),
        "Sensory_%": pct(counts["sens"]),
        "Spatial_%": pct(counts["spat"]),
    })
    return feats


def compute_features(texts):
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    texts = [t if isinstance(t, str) else "" for t in texts]
    rows = [features_from_doc(d) for d in nlp.pipe(texts, batch_size=64)]
    return pd.DataFrame(rows)[ALL_FEATURES]


def get_features(df):
    """Признаки для всех текстов; кэш проверяется по md5 текстов."""
    md5 = [hashlib.md5(str(t).encode("utf-8")).hexdigest() for t in df["text"]]
    if os.path.exists(CACHE_PATH):
        cached = pd.read_csv(CACHE_PATH)
        if list(cached.get("md5", [])) == md5 and all(f in cached.columns for f in ALL_FEATURES):
            return cached[ALL_FEATURES]
        print("Кэш не совпадает с данными, пересчитываем...")
    print(f"Извлекаем признаки для {len(df)} текстов (spaCy, несколько минут)...")
    feats = compute_features(df["text"].tolist())
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    out = feats.copy()
    out["md5"] = md5
    out.to_csv(CACHE_PATH, index=False)
    return feats


# ---------------------------------------------------------------
# 3. Статистика: Манн-Уитни, rank-biserial, поправка Холма
# ---------------------------------------------------------------
def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def feature_stats(X_df, y):
    """effect_rb > 0: у ложных текстов значения выше, чем у правдивых."""
    rows = []
    for f in X_df.columns:
        dec = X_df.loc[y == 1, f].values
        tru = X_df.loc[y == 0, f].values
        res = mannwhitneyu(dec, tru, alternative="two-sided")
        rows.append({
            "feature": f,
            "truthful_mean": tru.mean(),
            "deceptive_mean": dec.mean(),
            "effect_rb": 2 * res.statistic / (len(dec) * len(tru)) - 1,
            "p_value": res.pvalue,
        })
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p_value"].values)
    out["signif_holm"] = out["p_holm"] < 0.05
    return out


# ---------------------------------------------------------------
# 4. Модели и кросс-валидация
# ---------------------------------------------------------------
def make_models():
    lr = lambda seed: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    cb = lambda seed: CatBoostClassifier(iterations=300, depth=4, learning_rate=0.05,
                                         l2_leaf_reg=10, verbose=0, random_seed=seed)
    bow = lambda seed: make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        LogisticRegression(max_iter=2000, C=10))
    # cols=None означает, что модель работает по сырому тексту
    return {
        "length_only (LogReg)": (lr, ["Total_Words"]),
        "base11 (LogReg)": (lr, BASE_FEATURES),
        "ext (LogReg)": (lr, EXT_FEATURES),
        "base11 (CatBoost)": (cb, BASE_FEATURES),
        "ext (CatBoost)": (cb, EXT_FEATURES),
        "BoW TF-IDF (LogReg)": (bow, None),
    }


def model_input(cols, X_df, texts):
    return texts if cols is None else X_df[cols].values


def bootstrap_auc_ci(y, p, groups, n_boot=1000, seed=0):
    """95% доверительный интервал AUC, бутстреп по отелям (группам)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by_g = {g: np.where(groups == g)[0] for g in uniq}
    aucs = []
    for _ in range(n_boot):
        sel = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_g[g] for g in sel])
        if len(np.unique(y[idx])) == 2:
            aucs.append(roc_auc_score(y[idx], p[idx]))
    return np.percentile(aucs, [2.5, 97.5])


def cross_validate(X_df, texts, y, groups, models):
    """AUC считается по объединённым out-of-fold предсказаниям каждого повтора.
    CI: бутстреп по отелям на предсказаниях первого повтора."""
    rows = []
    for name, (factory, cols) in models.items():
        inp = model_input(cols, X_df, texts)
        aucs, accs, f1s, first_oof = [], [], [], None
        for r in range(N_REPEATS):
            seed = SEED + r
            cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
            oof = np.zeros(len(y))
            for tr, te in cv.split(np.zeros(len(y)), y, groups):
                model = factory(seed)
                model.fit(inp[tr], y[tr])
                oof[te] = model.predict_proba(inp[te])[:, 1]
            aucs.append(roc_auc_score(y, oof))
            accs.append(accuracy_score(y, oof > 0.5))
            f1s.append(f1_score(y, oof > 0.5))
            if r == 0:
                first_oof = oof
        lo, hi = bootstrap_auc_ci(y, first_oof, groups)
        rows.append({"model": name, "AUC": np.mean(aucs), "AUC_std_repeats": np.std(aucs),
                     "CI95_low": lo, "CI95_high": hi,
                     "accuracy": np.mean(accs), "F1": np.mean(f1s)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------
# 5. Важность признаков: permutation importance на отложенных фолдах
# ---------------------------------------------------------------
def permutation_importance_cv(X_df, y, groups, factory, cols, n_perm=10):
    rng = np.random.default_rng(SEED)
    Xv = X_df[cols].values
    imp = np.zeros(len(cols))
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    n_folds = 0
    for tr, te in cv.split(Xv, y, groups):
        model = factory(SEED)
        model.fit(Xv[tr], y[tr])
        X_te = Xv[te]
        base = roc_auc_score(y[te], model.predict_proba(X_te)[:, 1])
        for j in range(len(cols)):
            drops = []
            for _ in range(n_perm):
                Xp = X_te.copy()
                Xp[:, j] = rng.permutation(Xp[:, j])
                drops.append(base - roc_auc_score(y[te], model.predict_proba(Xp)[:, 1]))
            imp[j] += np.mean(drops)
        n_folds += 1
    return pd.Series(imp / n_folds, index=cols)


def logreg_coefficients(X_df, y, cols):
    """Стандартизованные коэффициенты; > 0: признак повышает вероятность 'ложь'."""
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    model.fit(X_df[cols].values, y)
    return pd.Series(model[-1].coef_[0], index=cols)


# ---------------------------------------------------------------
# 6. Перенос между полярностями
# ---------------------------------------------------------------
def transfer_eval(X_tr, tx_tr, y_tr, X_te, tx_te, y_te, models):
    rows = []
    for name, (factory, cols) in models.items():
        m = factory(SEED)
        m.fit(model_input(cols, X_tr, tx_tr), y_tr)
        p = m.predict_proba(model_input(cols, X_te, tx_te))[:, 1]
        rows.append({"model": name, "AUC": roc_auc_score(y_te, p),
                     "accuracy": accuracy_score(y_te, p > 0.5)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------
# 7. Загрузка данных и основной сценарий
# ---------------------------------------------------------------
def load_data():
    df = pd.read_csv(DATA_PATH)
    print("Колонки в файле:", list(df.columns))
    missing = {"deceptive", "hotel", "polarity", "text"} - set(df.columns)
    if missing:
        raise SystemExit(f"В CSV нет колонок {missing}. Поправьте имена в load_data().")
    df["label"] = (df["deceptive"] == "deceptive").astype(int)
    return df.reset_index(drop=True)


def main():
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = load_data()
    X_all = get_features(df)
    y_all = df["label"].values
    texts_all = df["text"].astype(str).values
    groups_all = df["hotel"].astype(str).values
    models = make_models()

    subsets, stats = {}, {}
    for pol in [p for p in ("positive", "negative") if (df["polarity"] == p).any()]:
        mask = (df["polarity"] == pol).values
        X_df = X_all[mask].reset_index(drop=True)
        y, texts, groups = y_all[mask], texts_all[mask], groups_all[mask]
        subsets[pol] = (X_df, texts, y)
        print(f"\n{'=' * 70}\nПОЛЯРНОСТЬ: {pol} (n={len(y)}, ложных={int(y.sum())})\n{'=' * 70}")

        st = feature_stats(X_df, y)
        stats[pol] = st
        st.to_csv(f"{RESULTS_DIR}/stats_{pol}.csv", index=False)
        print("\n--- Различия признаков (effect_rb > 0: выше у ложных; поправка Холма по всем признакам) ---")
        print(st.round(4).to_string(index=False))

        print(f"\n--- Кросс-валидация по отелям ({N_SPLITS} фолдов x {N_REPEATS} повторов) ---")
        cv_res = cross_validate(X_df, texts, y, groups, models)
        cv_res.to_csv(f"{RESULTS_DIR}/cv_{pol}.csv", index=False)
        print(cv_res.round(3).to_string(index=False))

        print("\n--- Важность признаков (расширенный набор) ---")
        imp = pd.DataFrame({
            "coef_logreg_std": logreg_coefficients(X_df, y, EXT_FEATURES),
            "perm_AUC_drop_logreg": permutation_importance_cv(X_df, y, groups, models["ext (LogReg)"][0], EXT_FEATURES),
            "perm_AUC_drop_catboost": permutation_importance_cv(X_df, y, groups, models["ext (CatBoost)"][0], EXT_FEATURES),
        }).sort_values("perm_AUC_drop_logreg", ascending=False)
        imp.to_csv(f"{RESULTS_DIR}/importance_{pol}.csv")
        print(imp.round(4).to_string())

    if len(subsets) == 2:
        print(f"\n{'=' * 70}\nСРАВНЕНИЕ НАПРАВЛЕНИЯ ЭФФЕКТОВ: positive vs negative\n{'=' * 70}")
        a = stats["positive"].set_index("feature")
        b = stats["negative"].set_index("feature")
        cmp_df = pd.DataFrame({
            "rb_pos": a["effect_rb"], "rb_neg": b["effect_rb"],
            "signif_pos": a["signif_holm"], "signif_neg": b["signif_holm"],
        })
        cmp_df["same_sign"] = np.sign(cmp_df["rb_pos"]) == np.sign(cmp_df["rb_neg"])
        cmp_df.to_csv(f"{RESULTS_DIR}/direction_comparison.csv")
        print(cmp_df.round(3).to_string())

        print(f"\n{'=' * 70}\nПЕРЕНОС МЕЖДУ ПОЛЯРНОСТЯМИ\n{'=' * 70}")
        all_tr = []
        for src, dst in (("positive", "negative"), ("negative", "positive")):
            Xs, ts, ys = subsets[src]
            Xd, td, yd = subsets[dst]
            res = transfer_eval(Xs, ts, ys, Xd, td, yd, models)
            res.insert(0, "direction", f"{src} -> {dst}")
            all_tr.append(res)
            print(f"\n{src} -> {dst}")
            print(res.drop(columns="direction").round(3).to_string(index=False))
        pd.concat(all_tr).to_csv(f"{RESULTS_DIR}/transfer.csv", index=False)

    print(f"\nГотово. Таблицы сохранены в {RESULTS_DIR}/")


if __name__ == "__main__":
    main()