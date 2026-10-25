"""
veracity_explain.py: логика приложения без Streamlit (чтобы её можно было тестировать).

- explain(): вероятность класса «ложь» и вклад каждого признака (коэффициент x z-оценка);
- highlight_html(): подсветка слов-маркеров в тексте;
- verdict_band(), plain_summary(), cyrillic_share(): вспомогательные функции.

Лексиконы и правила подсчёта берутся из ott_pipeline_v2.py, чтобы приложение и
обучение модели считали признаки одинаково.
"""
import html
import numpy as np
import pandas as pd

from ott_pipeline_v2 import (FIRST_SING, FIRST_PLUR, THIRD_PERSON, COG_MARKERS, CAUSAL_MARKERS,
                             EMO_POS, EMO_NEG, NEGATION, HEDGE, EXCLUSIVE, SENSORY, SPATIAL)

# название признака на русском и единица измерения
FEATURE_LABELS = {
    "Total_Words": ("Длина текста (слов)", ""),
    "First_Person_%": ("Местоимения 1-го лица (I, we…)", "%"),
    "First_Sing_%": ("Местоимения 1-го лица, ед. ч. (I, me, my)", "%"),
    "First_Plur_%": ("Местоимения 1-го лица, мн. ч. (we, our)", "%"),
    "Third_Person_%": ("Местоимения 3-го лица (he, they, it)", "%"),
    "Cognitive_Words_%": ("Когнитивные слова (think, know)", "%"),
    "Causal_Words_%": ("Причинные слова (because, so)", "%"),
    "Emotion_Words_%": ("Эмоциональная лексика (всего)", "%"),
    "Emotion_Pos_%": ("Положительная эмоциональная лексика", "%"),
    "Emotion_Neg_%": ("Отрицательная эмоциональная лексика", "%"),
    "Negation_%": ("Отрицания (not, never)", "%"),
    "Hedge_%": ("Смягчения (maybe, might)", "%"),
    "Exclusive_%": ("Исключающие слова (but, however)", "%"),
    "Sensory_%": ("Сенсорные слова (see, hear)", "%"),
    "Spatial_%": ("Пространственные слова (near, inside)", "%"),
    "Tree_Depth": ("Глубина синтаксического дерева (среднее)", ""),
    "Clauses": ("Клаузы в предложении (среднее)", ""),
    "Dependency_Distance": ("Дистанция зависимостей (среднее)", ""),
    "Syntax_Density": ("Плотность синтаксиса (глубина / длина)", ""),
    "Lexical_TTR": ("Лексическое разнообразие (TTR)", "%"),
}

# категории слов для подсветки: ключ -> (название, цвет)
CATEGORIES = {
    "first_sing": ("Я (I, me, my)", "#FFD166"),
    "first_plur": ("Мы (we, our)", "#F4A261"),
    "third": ("Он / она / они", "#CDB4DB"),
    "negation": ("Отрицания", "#EF9A9A"),
    "hedge": ("Смягчения", "#A5D8FF"),
    "exclusive": ("Исключающие слова", "#B2F2BB"),
    "cognitive": ("Когнитивные слова", "#FFF3BF"),
    "causal": ("Причинные слова", "#D0BFFF"),
    "emo_pos": ("Положительные эмоции", "#C3FAE8"),
    "emo_neg": ("Отрицательные эмоции", "#FFC9C9"),
    "sensory": ("Сенсорные слова", "#99E9F2"),
    "spatial": ("Пространственные слова", "#FFEC99"),
}
DEFAULT_HIGHLIGHT = ["first_sing", "first_plur", "negation", "hedge", "exclusive"]


def token_category(token):
    """Категория слова; правила совпадают с подсчётом признаков в features_from_doc()."""
    lemma = token.lemma_.lower()
    if token.pos_ == "PRON":
        if lemma in FIRST_SING:
            return "first_sing"
        if lemma in FIRST_PLUR:
            return "first_plur"
        if lemma in THIRD_PERSON:
            return "third"
    if lemma in NEGATION:
        return "negation"
    if lemma in HEDGE:
        return "hedge"
    if lemma in EXCLUSIVE:
        return "exclusive"
    if lemma in COG_MARKERS:
        return "cognitive"
    if lemma in CAUSAL_MARKERS:
        return "causal"
    if token.pos_ in ("ADJ", "ADV"):
        if lemma in EMO_POS:
            return "emo_pos"
        if lemma in EMO_NEG:
            return "emo_neg"
    if lemma in SENSORY:
        return "sensory"
    if lemma in SPATIAL:
        return "spatial"
    return None


def highlight_html(doc, selected):
    """HTML текста с подсветкой выбранных категорий (весь пользовательский текст экранируется)."""
    selected = set(selected)
    parts = []
    for tok in doc:
        text = html.escape(tok.text)
        space = html.escape(tok.whitespace_)
        cat = token_category(tok)
        if cat in selected:
            title, color = CATEGORIES[cat]
            parts.append(f'<mark title="{html.escape(title)}" style="background:{color};color:#111;'
                         f'padding:0 2px;border-radius:3px">{text}</mark>{space}')
        else:
            parts.append(text + space)
    return '<div style="white-space:pre-wrap;line-height:1.8">' + "".join(parts) + "</div>"


def explain(entry, feats):
    """entry: запись из models/demo_models.joblib; feats: словарь признаков текста.
    Возвращает вероятность класса «ложь», базовую вероятность и таблицу вкладов."""
    cols = entry["features"]
    x = np.array([[feats[c] for c in cols]], dtype=float)
    scaler, clf = entry["pipeline"][0], entry["pipeline"][-1]
    z = scaler.transform(x)[0]
    contrib = clf.coef_[0] * z
    logit = clf.intercept_[0] + contrib.sum()
    sigmoid = lambda v: 1.0 / (1.0 + np.exp(-v))

    rows = []
    for i, c in enumerate(cols):
        label, unit = FEATURE_LABELS[c]
        truthful_mean, deceptive_mean = entry["class_means"][c]
        rows.append({"feature": c, "label": label, "unit": unit, "value": x[0, i],
                     "truthful_mean": truthful_mean, "deceptive_mean": deceptive_mean,
                     "z": z[i], "contribution": contrib[i]})
    table = pd.DataFrame(rows).sort_values("contribution", key=lambda s: s.abs(),
                                           ascending=False).reset_index(drop=True)
    return {"p": float(sigmoid(logit)), "base": float(sigmoid(clf.intercept_[0])), "table": table}


def verdict_band(p):
    """(текст вердикта, тип сообщения Streamlit)"""
    if p < 0.4:
        return "Скорее правдивый", "success"
    if p <= 0.6:
        return "Неопределённо", "warning"
    return "Скорее ложный", "error"


def plain_summary(table, k=3):
    lines = []
    for _, r in table.head(k).iterrows():
        direction = "повышает" if r["contribution"] > 0 else "снижает"
        u = r["unit"]
        lines.append(
            f"**{r['label']}**: {r['value']:.2f}{u} в вашем тексте (в среднем у правдивых отзывов "
            f"{r['truthful_mean']:.2f}{u}, у ложных {r['deceptive_mean']:.2f}{u}), "
            f"это {direction} оценку «ложь».")
    return lines


def cyrillic_share(text):
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    return sum("\u0400" <= ch <= "\u04ff" for ch in letters) / len(letters)
