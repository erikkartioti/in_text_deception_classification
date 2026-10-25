"""
train_demo_model.py: обучает модели для демо-приложения и сохраняет их в models/demo_models.joblib.

Данные и признаки берутся из ott_pipeline_v2.py (корпус Ott, кэш признаков cache/ott_features_v2.csv).
Модели: логистическая регрессия на стандартизованных признаках (коэффициенты читаются напрямую).
Качество оценивается кросс-валидацией по отелям (3 повтора) и сохраняется вместе с моделью.

Запуск:  python train_demo_model.py
"""
import os
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ott_pipeline_v2 import load_data, get_features, EXT_FEATURES

MODEL_PATH = "models/demo_models.joblib"

# «Устойчивое ядро» выбрано по результатам абляции на Ott (v3): это гипотеза для проверки
# на новых данных, а не доказанный набор.
CORE_FEATURES = ["First_Sing_%", "First_Plur_%", "Third_Person_%",
                 "Tree_Depth", "Clauses", "Dependency_Distance", "Syntax_Density"]

SPECS = {
    "ext": {"title": "Расширенный набор (18 признаков)", "features": EXT_FEATURES},
    "core": {"title": "«Устойчивое ядро»: местоимения + синтаксис (7 признаков)", "features": CORE_FEATURES},
}


def new_model():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))


def main(n_repeats=3, seed=42):
    df = load_data()
    X_all = get_features(df)
    y = df["label"].values
    groups = df["hotel"].astype(str).values

    bundle = {}
    for key, spec in SPECS.items():
        cols = spec["features"]
        X = X_all[cols].values
        aucs = []
        for r in range(n_repeats):
            cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed + r)
            oof = np.zeros(len(y))
            for tr, te in cv.split(X, y, groups):
                oof[te] = new_model().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
            aucs.append(roc_auc_score(y, oof))

        bundle[key] = {
            "title": spec["title"],
            "features": cols,
            "pipeline": new_model().fit(X, y),
            "class_means": {c: [float(X_all.loc[y == 0, c].mean()), float(X_all.loc[y == 1, c].mean())]
                            for c in cols},
            "cv_auc": float(np.mean(aucs)),
            "n_train": int(len(y)),
        }
        print(f"{spec['title']}: AUC (CV по отелям) = {bundle[key]['cv_auc']:.3f}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    print(f"Сохранено: {MODEL_PATH}")


if __name__ == "__main__":
    main()
