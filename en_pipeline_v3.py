"""
ott_pipeline_v3.py: надстройка над ott_pipeline_v2.py 


Признаки берутся из кэша v2 (cache/ott_features_v2.csv)
Результаты: results/*.csv
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score

from en_pipeline_v2 import (EXT_FEATURES, N_SPLITS, N_REPEATS, SEED, load_data,
                             get_features, feature_stats, bootstrap_auc_ci, make_models)

RESULTS_DIR = "results"
CHARTS_DIR = "charts"
N_BOOT = 1000

# Группы признаков (только признаки расширенного набора, без пересечений)
GROUPS = {
    "Местоимения (Хэнкок)": ["First_Sing_%", "First_Plur_%", "Third_Person_%"],
    "Аффект и когниция (Хэнкок)": ["Cognitive_Words_%", "Causal_Words_%", "Emotion_Pos_%", "Emotion_Neg_%"],
    "Синтаксис (Де Санто)": ["Tree_Depth", "Clauses", "Dependency_Distance", "Syntax_Density"],
    "LIWC-дополнения": ["Negation_%", "Hedge_%", "Exclusive_%", "Sensory_%", "Spatial_%"],
    "Длина и TTR": ["Total_Words", "Lexical_TTR"],
}
assert sorted(sum(GROUPS.values(), [])) == sorted(EXT_FEATURES), "GROUPS должны покрывать EXT_FEATURES"

PAIRS = [
    ("ext (LogReg)", "base11 (LogReg)"),
    ("ext (CatBoost)", "base11 (CatBoost)"),
    ("BoW TF-IDF (LogReg)", "ext (LogReg)"),
    ("ext+BoW stack (LogReg)", "BoW TF-IDF (LogReg)"),
]


# ---------------------------------------------------------------
# Стекинг: признаки + logit вероятности BoW
# ---------------------------------------------------------------
class BowStackLR:
    """Вход: массив (n, k+2): k признаков, затем группа (отель), затем текст."""

    def __init__(self, seed=SEED):
        self.seed = seed

    @staticmethod
    def _split(inp):
        return inp[:, :-2].astype(float), inp[:, -2], inp[:, -1]

    @staticmethod
    def _bow():
        return make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                             LogisticRegression(max_iter=2000, C=10))

    @staticmethod
    def _logit(p):
        p = np.clip(p, 1e-4, 1 - 1e-4)
        return np.log(p / (1 - p))

    def fit(self, inp, y):
        F, g, t = self._split(inp)
        inner = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=self.seed)
        oof = np.zeros(len(y))
        for tr, te in inner.split(np.zeros(len(y)), y, g):
            oof[te] = self._bow().fit(t[tr], y[tr]).predict_proba(t[te])[:, 1]
        self.bow_ = self._bow().fit(t, y)
        self.meta_ = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        self.meta_.fit(np.column_stack([F, self._logit(oof)]), y)
        return self

    def predict_proba(self, inp):
        F, _, t = self._split(inp)
        z = self._logit(self.bow_.predict_proba(t)[:, 1])
        return self.meta_.predict_proba(np.column_stack([F, z]))


def models_v3():
    models = make_models()
    models["ext+BoW stack (LogReg)"] = (lambda seed: BowStackLR(seed), "STACK")
    return models


def build_input(cols, X_df, texts, groups):
    if cols is None:
        return texts
    if isinstance(cols, str) and cols == "STACK":
        return np.column_stack([X_df[EXT_FEATURES].values.astype(object),
                                groups.astype(object), texts.astype(object)])
    return X_df[cols].values


# ---------------------------------------------------------------
# Кросс-валидация и перенос (с сохранением предсказаний)
# ---------------------------------------------------------------
def oof_predict(factory, inp, y, groups, seed):
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    for tr, te in cv.split(np.zeros(len(y)), y, groups):
        model = factory(seed)
        model.fit(inp[tr], y[tr])
        oof[te] = model.predict_proba(inp[te])[:, 1]
    return oof


def cv_models(X_df, texts, y, groups, models):
    rows, oof_first = [], {}
    for name, (factory, cols) in models.items():
        print(f"   {name} ...", flush=True)
        inp = build_input(cols, X_df, texts, groups)
        aucs = []
        for r in range(N_REPEATS):
            oof = oof_predict(factory, inp, y, groups, SEED + r)
            aucs.append(roc_auc_score(y, oof))
            if r == 0:
                oof_first[name] = oof
        lo, hi = bootstrap_auc_ci(y, oof_first[name], groups)
        rows.append({"model": name, "AUC": np.mean(aucs), "AUC_std_repeats": np.std(aucs),
                     "CI95_low": lo, "CI95_high": hi})
    return pd.DataFrame(rows), oof_first


def transfer_models(src, dst, models):
    Xs, ts, ys, gs = src
    Xd, td, yd, gd = dst
    rows, preds = [], {}
    for name, (factory, cols) in models.items():
        model = factory(SEED)
        model.fit(build_input(cols, Xs, ts, gs), ys)
        p = model.predict_proba(build_input(cols, Xd, td, gd))[:, 1]
        lo, hi = bootstrap_auc_ci(yd, p, gd)
        rows.append({"model": name, "AUC": roc_auc_score(yd, p), "CI95_low": lo, "CI95_high": hi})
        preds[name] = p
    return pd.DataFrame(rows), preds


# ---------------------------------------------------------------
# Парный бутстреп разности AUC
# ---------------------------------------------------------------
def paired_bootstrap(y, pa, pb, groups, n_boot=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by_g = {g: np.where(groups == g)[0] for g in uniq}
    diffs = []
    for _ in range(n_boot):
        idx = np.concatenate([idx_by_g[g] for g in rng.choice(uniq, size=len(uniq), replace=True)])
        if len(np.unique(y[idx])) == 2:
            diffs.append(roc_auc_score(y[idx], pa[idx]) - roc_auc_score(y[idx], pb[idx]))
    diffs = np.array(diffs)
    obs = roc_auc_score(y, pa) - roc_auc_score(y, pb)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
    return obs, lo, hi, max(p, 1.0 / len(diffs))


def paired_table(settings):
    """settings: {название: (y, groups, {модель: предсказания})}"""
    rows = []
    for setting, (y, groups, preds) in settings.items():
        for a, b in PAIRS:
            obs, lo, hi, p = paired_bootstrap(y, preds[a], preds[b], groups)
            rows.append({"setting": setting, "A": a, "B": b, "AUC_diff_A_minus_B": obs,
                         "CI95_low": lo, "CI95_high": hi, "p_boot": p})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------
# Абляция по группам признаков
# ---------------------------------------------------------------
def ablation(subsets):
    lr = lambda seed: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    configs = [("все признаки (ext)", EXT_FEATURES)]
    for g, cols in GROUPS.items():
        configs.append((f"только: {g}", cols))
        configs.append((f"без: {g}", [c for c in EXT_FEATURES if c not in cols]))

    rows = []
    for label, cols in configs:
        row = {"config": label, "n_features": len(cols)}
        for pol, (X_df, _, y, groups) in subsets.items():
            inp = X_df[cols].values
            row[f"AUC_{pol}"] = np.mean([roc_auc_score(y, oof_predict(lr, inp, y, groups, SEED + r))
                                         for r in range(N_REPEATS)])
        for src, dst in (("positive", "negative"), ("negative", "positive")):
            Xs, _, ys, _ = subsets[src]
            Xd, _, yd, _ = subsets[dst]
            m = lr(SEED).fit(Xs[cols].values, ys)
            row[f"AUC_{src[:3]}2{dst[:3]}"] = roc_auc_score(yd, m.predict_proba(Xd[cols].values)[:, 1])
        rows.append(row)

    out = pd.DataFrame(rows)
    auc_cols = [c for c in out.columns if c.startswith("AUC_")]
    for c in auc_cols:
        out["d_" + c[4:]] = out[c] - out.loc[0, c]
    return out


# ---------------------------------------------------------------
# Графики
# ---------------------------------------------------------------
def forest_plot(st_pos, st_neg, path):
    a = st_pos.set_index("feature").loc[EXT_FEATURES]
    b = st_neg.set_index("feature").loc[EXT_FEATURES]
    order = list(((a["effect_rb"] + b["effect_rb"]) / 2).sort_values().index)
    fig, ax = plt.subplots(figsize=(9, 8))
    for i, f in enumerate(order):
        for off, st, marker, color in ((-0.17, a, "o", "#1f77b4"), (0.17, b, "s", "#d62728")):
            filled = bool(st.loc[f, "signif_holm"])
            ax.scatter(st.loc[f, "effect_rb"], i + off, marker=marker, s=70, linewidth=1.6,
                       edgecolor=color, facecolor=color if filled else "white", zorder=3)
    ax.axvline(0, color="gray", lw=1)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.set_xlabel("Размер эффекта (rank-biserial); > 0: выше у ложных текстов")
    ax.set_title("Различия признаков между правдивыми и ложными отзывами")
    handles = [
        Line2D([], [], marker="o", color="#1f77b4", linestyle="", label="положительные"),
        Line2D([], [], marker="s", color="#d62728", linestyle="", label="отрицательные"),
        Line2D([], [], marker="o", color="black", markerfacecolor="black", linestyle="", label="значимо (Холм, 0.05)"),
        Line2D([], [], marker="o", color="black", markerfacecolor="white", linestyle="", label="не значимо"),
    ]
    ax.legend(handles=handles, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def auc_chart(cv_pos, cv_neg, tr_p2n, tr_n2p, path):
    names = list(cv_pos["model"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    y = np.arange(len(names))
    h = 0.38
    panels = [
        ("Внутри полярности (CV по отелям)", (cv_pos, "positive"), (cv_neg, "negative")),
        ("Перенос между полярностями", (tr_p2n, "positive → negative"), (tr_n2p, "negative → positive")),
    ]
    for ax, (title, first, second) in zip(axes, panels):
        for k, (d, label, color) in enumerate((first + ("#1f77b4",), second + ("#d62728",))):
            d = d.set_index("model").loc[names]
            err = [d["AUC"] - d["CI95_low"], d["CI95_high"] - d["AUC"]]
            ax.barh(y + (k - 0.5) * h, d["AUC"], height=h, xerr=err, capsize=2, color=color, label=label)
        ax.axvline(0.5, color="gray", lw=1, linestyle="--")
        ax.set_xlim(0.4, 1.0)
        ax.set_xlabel("ROC-AUC (95% ДИ, бутстреп по отелям)")
        ax.set_title(title)
        ax.legend(loc="lower right")
        ax.grid(axis="x", linestyle="--", alpha=0.4)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(names)
    axes[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------
# Основной сценарий
# ---------------------------------------------------------------
def main():
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CHARTS_DIR, exist_ok=True)

    df = load_data()
    X_all = get_features(df)
    subsets = {}
    for pol in ("positive", "negative"):
        mask = (df["polarity"] == pol).values
        if mask.any():
            subsets[pol] = (X_all[mask].reset_index(drop=True),
                            df.loc[mask, "text"].astype(str).values,
                            df.loc[mask, "label"].values,
                            df.loc[mask, "hotel"].astype(str).values)
    if len(subsets) < 2:
        raise SystemExit("Для v3 нужны и положительные, и отрицательные отзывы.")
    pos, neg = subsets["positive"], subsets["negative"]
    models = models_v3()

    print("\n[1/5] Размеры эффекта и forest plot")
    st_pos, st_neg = feature_stats(pos[0], pos[2]), feature_stats(neg[0], neg[2])
    forest_plot(st_pos, st_neg, f"{CHARTS_DIR}/forest_effects.png")

    print("\n[2/5] Кросс-валидация моделей (включая стекинг с BoW)")
    cv_res, oof = {}, {}
    for pol, (X_df, texts, y, groups) in subsets.items():
        print(f"  {pol}:")
        cv_res[pol], oof[pol] = cv_models(X_df, texts, y, groups, models)
        cv_res[pol].to_csv(f"{RESULTS_DIR}/v3_cv_{pol}.csv", index=False)
        print(cv_res[pol].round(3).to_string(index=False))

    print("\n[3/5] Перенос между полярностями")
    tr_p2n, pred_p2n = transfer_models(pos, neg, models)
    tr_n2p, pred_n2p = transfer_models(neg, pos, models)
    pd.concat([tr_p2n.assign(direction="positive->negative"),
               tr_n2p.assign(direction="negative->positive")]).to_csv(f"{RESULTS_DIR}/v3_transfer.csv", index=False)
    print("positive -> negative")
    print(tr_p2n.round(3).to_string(index=False))
    print("negative -> positive")
    print(tr_n2p.round(3).to_string(index=False))
    auc_chart(cv_res["positive"], cv_res["negative"], tr_p2n, tr_n2p, f"{CHARTS_DIR}/auc_models.png")

    print("\n[4/5] Парные сравнения AUC (бутстреп по отелям)")
    paired = paired_table({
        "CV positive": (pos[2], pos[3], oof["positive"]),
        "CV negative": (neg[2], neg[3], oof["negative"]),
        "positive->negative": (neg[2], neg[3], pred_p2n),
        "negative->positive": (pos[2], pos[3], pred_n2p),
    })
    paired.to_csv(f"{RESULTS_DIR}/v3_paired_tests.csv", index=False)
    print(paired.round(3).to_string(index=False))

    print("\n[5/5] Абляция по группам признаков (логистическая регрессия)")
    abl = ablation(subsets)
    abl.to_csv(f"{RESULTS_DIR}/v3_ablation_groups.csv", index=False)
    print(abl.round(3).to_string(index=False))

    print(f"\nГотово. Таблицы: {RESULTS_DIR}/v3_*.csv, графики: {CHARTS_DIR}/forest_effects.png, "
          f"{CHARTS_DIR}/auc_models.png")


if __name__ == "__main__":
    main()