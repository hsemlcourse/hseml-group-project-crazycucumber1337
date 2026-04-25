[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/kOqwghv0)
# ML Project — Предсказание уровня эмоционального выгорания студентов

**Студент:** Дыбнова Ирина Сергеевна

**Группа:** БИВ234


## Оглавление

1. [Описание задачи](#описание-задачи)
2. [Структура репозитория](#структура-репозитория)
3. [Запуски](#быстрый-старт)
4. [Данные](#данные)
5. [Результаты](#результаты)
7. [Отчёт](#отчёт)


## Описание задачи

**Задача:** Классификация

**Датасет:** Student Mental Health and Burnout Dataset (https://www.kaggle.com/datasets/sehaj1104/student-mental-health-and-burnout-dataset/data)

**Целевая метрика:** Weighted F1-score


## Структура репозитория
```
.
├── data
│   ├── processed               # Очищенные и обработанные данные
│   └── raw                     # Исходные файлы
├── models                      # Сохранённые модели 
├── notebooks
│   ├── 01_eda.ipynb            # EDA
│   ├── 02_baseline.ipynb       # Baseline-модель
│   └── 03_experiments.ipynb    # Эксперименты и ablation study
├── presentation                # Презентация для защиты
├── report
│   ├── images                  # Изображения для отчёта
│   └── report.md               # Финальный отчёт
├── src
│   ├── preprocessing.py        # Предобработка данных
│   └── modeling.py             # Обучение и оценка моделей
├── tests
│   └── test.py                 # Тесты пайплайна
├── requirements.txt
└── README.md
```

## Запуск

```bash
# 1. Клонировать репозиторий
git clone hseml-group-project-crazycucumber1337
cd hseml-group-project-crazycucumber1337

# 2. Создать виртуальное окружение
python -m venv .venv
# source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate    # Windows

# 3. Установить зависимости
pip install -r requirements.txt
```

## Данные
- `data/raw/` — исходные файлы
- `data/processed/` — предобработанные данные


## Результаты
| model_name            | hypothesis           | split | accuracy | weighted_f1 | macro_f1 | roc_auc_ovr | cv_f1            |
|-----------------------|----------------------|-------|----------|-------------|----------|-------------|------------------|
| Logistic Regression   | Линейное разделение  | val   | 0.3342   | 0.3325      | 0.3324   | 0.4976      | 0.3333 ± 0.0022  |
| Decision Tree         | Нелинейные правила   | val   | 0.3318   | 0.3318      | 0.3318   | 0.4989      | 0.3336 ± 0.0012  |
| KNN (k=11)            | Похожие студенты     | val   | 0.3287   | 0.3255      | 0.3256   | 0.4965      | 0.3307 ± 0.0028  |
| Dummy (most_frequent) | Нижняя граница       | val   | 0.3351   | 0.1682      | 0.1673   | 0.5000      | NaN              |


## Отчёт

Финальный отчёт: [`report/report.md`](report/report.md)
