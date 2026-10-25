"""
app.py: Streamlit-интерфейс демо «Лингвистические маркеры достоверности текста».

Запуск (из папки проекта):
    python train_demo_model.py      # один раз, создаёт models/demo_models.joblib
    streamlit run app.py
"""
import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from ott_pipeline_v2 import features_from_doc
from veracity_explain import (CATEGORIES, DEFAULT_HIGHLIGHT, cyrillic_share, explain,
                              highlight_html, plain_summary, verdict_band)

MODEL_PATH = "models/demo_models.joblib"
MIN_WORDS = 40

EXAMPLES = {
    "A": ("We stayed three nights in room 412 in early March. The room was on the quiet side of the "
          "building, about ten minutes on foot from the train station. Breakfast cost 14 dollars and was "
          "served until ten, but the coffee was weak. The bed was firm, the shower pressure was low on the "
          "second morning, and the front desk fixed it within an hour. The elevator is slow, so we used "
          "the stairs most of the time."),
    "B": ("I absolutely loved this hotel and I would recommend it to everyone. My husband and I had the "
          "most wonderful time. I could not believe how amazing the staff were, they made me feel so "
          "special. I think this is the best hotel I have ever stayed at and I know I will come back "
          "again. I really felt that every single detail was perfect for our family vacation."),
}

st.set_page_config(page_title="Лингвистические маркеры достоверности", layout="wide")


@st.cache_resource
def load_nlp():
    import spacy
    return spacy.load("en_core_web_sm", disable=["ner"])


@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH)


@st.cache_resource(max_entries=32)
def analyze_text(text):
    doc = load_nlp()(text)
    return doc, features_from_doc(doc)


def set_example(name):
    st.session_state["text"] = EXAMPLES[name]


# ---------------------------------------------------------------
# Модель и боковая панель
# ---------------------------------------------------------------
if not os.path.exists(MODEL_PATH):
    st.error("Модель не найдена. Сначала выполните в терминале: `python train_demo_model.py`")
    st.stop()
bundle = load_bundle()

with st.sidebar:
    st.header("Модель")
    key = st.selectbox("Набор признаков", list(bundle), format_func=lambda k: bundle[k]["title"])
    entry = bundle[key]
    st.metric("AUC (кросс-валидация по отелям)", f"{entry['cv_auc']:.2f}")
    st.caption(f"Обучена на {entry['n_train']} отзывах корпуса Ott (английский, отели). "
               "AUC 0.5 означает угадывание, 1.0 идеальное разделение.")
    st.divider()
    st.caption("Исследовательский прототип магистерской работы. "
               "Не используйте для решений о реальных людях.")

# ---------------------------------------------------------------
# Ввод текста
# ---------------------------------------------------------------
st.title("Лингвистические маркеры достоверности текста")
st.write("Вставьте текст на английском и получите оценку модели и объяснение, "
         "какие лингвистические признаки повлияли на неё.")

st.session_state.setdefault("text", "")
b1, b2, _ = st.columns([1, 1, 6])
b1.button("Пример A", on_click=set_example, args=("A",))
b2.button("Пример B", on_click=set_example, args=("B",))
st.caption("Примеры написаны для демонстрации, они не взяты из корпуса и не проверяют качество модели.")

text = st.text_area("Текст для анализа (на английском)", key="text", height=220)
if st.button("Проанализировать", type="primary"):
    st.session_state["analyzed"] = text.strip()

analyzed = st.session_state.get("analyzed", "")
if not analyzed:
    st.info("Вставьте текст и нажмите «Проанализировать».")
    st.stop()

if cyrillic_share(analyzed) > 0.3:
    st.error("Похоже, текст не на английском. Модель обучена на английских отзывах и для русского текста "
             "даст бессмысленный результат. Русская версия появится после подключения русскоязычного корпуса.")
    st.stop()

doc, feats = analyze_text(analyzed)
n_words = int(feats["Total_Words"])
if n_words < MIN_WORDS:
    st.warning(f"В тексте {n_words} слов. Модель обучена на текстах в среднем из 120–180 слов, "
               "на коротких текстах оценка очень ненадёжна.")

res = explain(entry, feats)
p, table = res["p"], res["table"]

# ---------------------------------------------------------------
# Вердикт
# ---------------------------------------------------------------
st.subheader("Вердикт модели")
m1, m2, m3 = st.columns(3)
label, kind = verdict_band(p)
m1.metric("Вероятность класса «ложь»", f"{p:.0%}")
m2.metric("Слов в тексте", n_words)
m3.metric("Предложений", len(list(doc.sents)))
getattr(st, kind)(f"**{label}.** Это статистическая оценка по лингвистическим признакам, "
                  "а не доказательство лжи.")
st.progress(min(max(p, 0.0), 1.0))
st.caption(f"Для «среднего» текста обучающей выборки модель дала бы {res['base']:.0%}. "
           "Диапазон 40–60% считается неопределённым.")

# ---------------------------------------------------------------
# Объяснение
# ---------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["Почему такой вердикт", "Слова-маркеры в тексте", "О модели и ограничениях"])

with tab1:
    st.markdown("**Главные факторы:**")
    for line in plain_summary(table, k=3):
        st.markdown("- " + line)

    d = table.sort_values("contribution")
    fig, ax = plt.subplots(figsize=(10, 0.45 * len(d) + 1.4))
    ax.barh(d["label"], d["contribution"],
            color=["#d62728" if v > 0 else "#1f77b4" for v in d["contribution"]])
    ax.axvline(0, color="gray", lw=1)
    ax.set_xlabel("Вклад в логарифм шансов: вправо к «лжи», влево к «правде»")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    show = table[["label", "value", "truthful_mean", "deceptive_mean", "z", "contribution"]].copy()
    show["direction"] = np.where(table["contribution"] > 0, "→ ложь", "→ правда")
    show.columns = ["Признак", "В вашем тексте", "Среднее: правдивые", "Среднее: ложные",
                    "z-оценка", "Вклад", "Направление"]
    st.dataframe(show.round(2), hide_index=True)
    st.caption("«z-оценка» показывает, насколько значение в вашем тексте отличается от среднего по обучающей "
               "выборке (в стандартных отклонениях). «Вклад» равен коэффициенту модели, умноженному на z-оценку: "
               "положительный сдвигает оценку к «лжи», отрицательный к «правде». "
               "Проценты считаются от числа слов текста.")

with tab2:
    selected = st.multiselect("Что подсвечивать", options=list(CATEGORIES), default=DEFAULT_HIGHLIGHT,
                              format_func=lambda k: CATEGORIES[k][0])
    st.markdown(highlight_html(doc, selected), unsafe_allow_html=True)
    st.caption("Подсвечены слова из словарей признаков. Модель учитывает их долю в тексте, а не отдельные слова. "
               "Выбранный набор признаков может использовать не все категории.")

with tab3:
    st.markdown(f"""
**Как устроена модель.** Логистическая регрессия на стандартизованных лингвистических признаках
(местоимения, лексика по категориям типа LIWC, синтаксическая сложность по Де Санто, лексическое разнообразие).
Выход модели: вероятность класса «ложь».

**Данные.** Корпус Ott et al.: отзывы об отелях на английском, правдивые (с TripAdvisor) и ложные
(написаны краудворкерами по заданию), положительные и отрицательные.

**Качество.** AUC около {entry['cv_auc']:.2f} при проверке на отелях, которых модель не видела.
Это умеренный сигнал. Модель по словам (bag-of-words) на этом корпусе заметно точнее, но не объясняет свой выбор.

**Ограничения.**
- Только английский язык и жанр отзывов об отелях; на других жанрах и языках результат не определён.
- Часть признаков меняет смысл в зависимости от контекста (например, у положительных и отрицательных отзывов).
- Ложные тексты в корпусе написаны на заказ, реальный обман может выглядеть иначе.
- Оценка не доказывает ложь и не должна использоваться для решений о людях.
""")
