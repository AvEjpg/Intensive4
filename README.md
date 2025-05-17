# Intensive4

<details>
  <summary style="display: flex; align-items: center; gap: 8px;">
    <h2 style="display: inline; margin: 0; font-size: 1.5em;">
      Многоклассовая классификация пользовательских комментариев на русском языке
    </h2>
  </summary>

Проект направлен на построение и обучение модели для мультиразметочной классификации пользовательских комментариев с использованием предобученной модели [blanchefort/rubert-base-cased-sentiment](https://huggingface.co/blanchefort/rubert-base-cased-sentiment).

---

## Оглавление

* [Описание проекта](#описание-проекта)
* [Установка зависимостей](#установка-зависимостей)
* [Обработка и анализ данных](#обработка-и-анализ-данных)
* [Визуализация данных](#визуализация-данных)
* [Предобработка текста](#предобработка-текста)
* [Модель и обучение](#модель-и-обучение)
* [Метрики и оценка качества](#метрики-и-оценка-качества)
* [Результаты](#результаты)
* [Авторы](#авторы)

---

## Описание проекта

Цель: классифицировать комментарии по нескольким категориям одновременно (*multi-label классификация*).

Категории:

* Вопрос решен
* Нравится качество выполнения заявки
* Нравится качество работы сотрудников
* Нравится скорость отработки заявок
* Понравилось выполнение заявки
* Другое

Каждому комментарию может соответствовать одна или несколько категорий одновременно.

---

## Установка зависимостей

bash
pip install spacy nltk scikit-learn transformers datasets iterative-stratification wordcloud evaluate
python -m spacy download ru_core_news_md


---

## Обработка и анализ данных

* Загрузка и объединение источников разметки
* Статистический анализ (распределения, частоты, оценки)
* Проверка ID и чистка пустых строк

---

## Визуализация данных

Для лучшего понимания структуры данных были построены:

* *Облако слов* по корпусу комментариев
* *Гистограмма* по частоте категорий
* *Гистограмма* распределения оценок пользователей


---

## Предобработка текста

* Очистка символов, приведение к нижнему регистру
* Лемматизация с помощью ru_core_news_md
* Токенизация с использованием AutoTokenizer
* Стратифицированное разбиение с MultilabelStratifiedKFold
* Преобразование категорий в мульти-бинарный формат

---

## Модель и обучение

*Используемая модель:* blanchefort/rubert-base-cased-sentiment

Особенности:

* Формат задачи: multi_label_classification
* Потери: Focal Loss с весами классов
* Токенизация: AutoTokenizer
* Кастомный Trainer с собственной функцией потерь

*Гиперпараметры:*

* Эпохи: 25
* Batch size: 32
* Learning rate: 2e-5
* Warmup: 10%
* Weight decay: 0.01

---

## Метрики и оценка качества

Для оценки модели используются:

* ROC-AUC (macro)
* F1-мера (macro)
* Accuracy

python
def compute_metrics(p):
    binary_preds = (p.predictions > 0.5).astype(int)
    return {
        "accuracy": accuracy_score(p.label_ids, binary_preds),
        "f1_macro": f1_score(p.label_ids, binary_preds, average='macro'),
        "roc_auc_macro": roc_auc_score(p.label_ids, p.predictions, average='macro'),
    }


---

## Результаты

Модель стабильно демонстрирует высокие показатели по всем ключевым метрикам, обеспечивая корректную мультиклассовую разметку комментариев.

---

</details>

---

[main.ipynb](https://github.com/AvEjpg/Intensive4/blob/main/main.ipynb) - <ins>**Конечный результат нашей работы**</ins>

[markup.csv](https://github.com/AvEjpg/Intensive4/blob/main/markup.csv) - **Итоговая разметка**

[building (1).png](https://github.com/AvEjpg/Intensive4/blob/main/building%20(1).png) - Маска для облака слов

[разметка комментариев 2.csv](https://github.com/AvEjpg/Intensive4/blob/main/разметка%20комментариев%202.csv) - Пазметка, нужна для сравнения

[?](-) - Презентация к работе (пока нет)

---
<details>
<summary><a href="https://github.com/AvEjpg/Intensive4/tree/main/experiments">experiments</a> - папка с нашими наработками и эксперементами</summary>

  * Модель_классификации_комментариев_по_работе_управляющей(инт4) .ipynb - старая модель
  * Модель_классификации_комментариев_по_работе_управляющей(инт4).ipynb - старая модель
  * разметка ч1.csv - старая разметка
  * разметка.ч1.csv - старая разметка
  * readmeold.md - старый readme файл

</details>

<details>
<summary><a href="https://github.com/AvEjpg/Intensive4/tree/main/Kuznetsov">Kuznetsov</a> - обработка данных и бейслайн Кузнецова Арсения</summary>

  * baseline2.ipynb - бейслайн модель
  * 321.csv - разметка

</details>


## Авторы

Работу выполняют:

[Флейшгауэр Александр](https://github.com/Glorc12) - ИСП-22

[Кузнецов Арсений](https://github.com/AvEjpg) - ИСП-22

[Серобян Кима](https://github.com/Kimaaaaaaaaaaaaaaaa) - ИСП-22
