[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/kOqwghv0)
# ML Project — Предсказание уровня эмоционального выгорания студентов

**Студент:** Дыбнова Ирина Сергеевна

**Группа:** БИВ234


## Оглавление

1. [Описание задачи](#описание-задачи)
2. [Структура репозитория](#структура-репозитория)
3. [Быстрый старт](#быстрый-старт)
4. [Запуск через Docker](#запуск-через-docker)
5. [Линтеры и качество кода](#линтеры-и-качество-кода)
6. [Данные](#данные)
7. [Результаты](#результаты)
8. [Отчёт](#отчёт)


## Описание задачи

**Задача:** Классификация

**Датасет:** Student Mental Health and Burnout Dataset (https://www.kaggle.com/datasets/sehaj1104/student-mental-health-and-burnout-dataset/data)

**Целевая метрика:** Weighted F1-score

**Целевая переменная:** `burnout_level` — уровень эмоционального выгорания студента (`Low` / `Medium` / `High`)


## Структура репозитория
```
.
├── data
│   ├── processed               # Очищенные и обработанные данные (train/val/test.csv)
│   └── raw                     # Исходные файлы (положить student_burnout.csv сюда)
├── models                      # Сохранённые модели (.pkl)
├── notebooks
│   ├── 00_data_loading.ipynb   # Загрузка данных через Kaggle API (REST + kaggle lib)
│   ├── 01_eda.ipynb            # EDA: загрузка, очистка, визуализация, сплит
│   ├── 02_baseline.ipynb       # Baseline-модели (LogReg, KNN, DT, Dummy)
│   └── 03_experiments.ipynb    # Эксперименты: RF, XGBoost, LightGBM, PCA, ансамбли,
│                               #   гиперпараметрический поиск (RandomizedSearchCV, GridSearchCV),
│                               #   выводы и обоснование финальной модели
├── presentation                # Презентация для защиты
├── report
│   ├── images                  # Изображения для отчёта
│   ├── baseline_results.csv    # Таблица результатов baseline
│   └── report.md               # Финальный отчёт
├── src
│   ├── __init__.py
│   ├── data_loader.py          # Загрузка данных через Kaggle API
│   ├── preprocessing.py        # Предобработка данных, feature engineering
│   └── modeling.py             # Обучение и оценка моделей
├── tests
│   └── test.py                 # Тесты пайплайна
├── Dockerfile                  # Docker-образ для воспроизводимости
├── docker-compose.yml          # Запуск Jupyter через Docker
├── Makefile                    # Команды для установки, линтинга, тестов, запуска
├── requirements.txt            # Зафиксированные версии зависимостей
└── README.md
```

## Быстрый старт

```bash
# 1. Клонировать репозиторий
git clone https://github.com/hsemlcourse/hseml-group-project-CrazyCucumber1337
cd hseml-group-project-CrazyCucumber1337

# 2. Создать виртуальное окружение
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# 3. Установить зависимости
make install
# или: pip install -r requirements.txt

# 4. Скачать датасет через Kaggle API (автоматически через ноутбук 00_data_loading.ipynb)
kaggle datasets download sehaj1104/student-mental-health-and-burnout-dataset -p data/raw/ --unzip
mv data/raw/student_mental_health_and_burnout_dataset.csv

# 5. Запустить все ноутбуки по порядку:
make run-all  # включает 00_data_loading, 01_eda, 02_baseline, 03_experiments
# или через Jupyter Lab вручную:
jupyter lab
```


## Запуск через Docker

Docker обеспечивает полностью воспроизводимое окружение без ручной установки зависимостей.

```bash
# Собрать Docker-образ
make docker-build
# или: docker-compose build

# Запустить Jupyter Lab
make docker-up
# или: docker-compose up -d jupyter
# Jupyter доступен по адресу: http://localhost:8888

# Запустить тесты внутри контейнера
make docker-test
# или: docker-compose --profile testing run --rm tests

# Остановить контейнеры
make docker-down
```

Директории `data/` и `models/` монтируются как volumes — данные и модели сохраняются между перезапусками.


## Линтеры и качество кода

Проект использует **ruff** (линтер + форматтер) и **flake8** для проверки стиля кода.

```bash
# Проверить и автоматически исправить проблемы со стилем:
make lint

# Только форматирование:
make format

# Только проверка без изменений (режим CI):
make lint-check

# Запустить напрямую:
ruff check src/ tests/ --fix
ruff format src/ tests/
flake8 src/ tests/ --max-line-length=120
```

Линтеры автоматически запускаются в GitHub Actions CI при каждом push (`.github/workflows/ci.yml`).


## Данные
- `data/raw/` — исходный файл `student_mental_health_burnout.csv` (скачивается с Kaggle, 150 000 строк, 19 колонок)
- `data/processed/` — предобработанные файлы: `train.csv`, `val.csv`, `test.csv`
- Сплит: **70% train / 15% val / 15% test**, стратификация по `burnout_level`


## Результаты
### Baseline-модели
| model_name            | hypothesis           | split | accuracy | weighted_f1 | macro_f1 | roc_auc_ovr | cv_f1            |
|-----------------------|----------------------|-------|----------|-------------|----------|-------------|------------------|
| Logistic Regression   | Линейное разделение  | val   | 0.3342   | 0.3325      | 0.3324   | 0.4976      | 0.3333 ± 0.0022  |
| Decision Tree         | Нелинейные правила   | val   | 0.3318   | 0.3318      | 0.3318   | 0.4989      | 0.3336 ± 0.0012  |
| KNN (k=11)            | Похожие студенты     | val   | 0.3287   | 0.3255      | 0.3256   | 0.4965      | 0.3307 ± 0.0028  |
| Dummy (most_frequent) | Нижняя граница       | val   | 0.3351   | 0.1682      | 0.1673   | 0.5000      | NaN              |

### Результаты экспериментов

| Model                             | Hypothesis                                | Weighted F1 | Macro F1 | Accuracy |
|-----------------------------------|-------------------------------------------|-------------|----------|----------|
| Voting (RF+XGB+LGBM)              | Soft voting                               | 0.3366      | 0.3366   | 0.3370   |
| LightGBM (GridSearchCV)           | Точный перебор num_leaves и learning_rate | 0.3337      | 0.3337   | 0.3340   |
| RandomForest (RandomizedSearchCV) | Случайный поиск гиперпараметров           | 0.3330      | 0.3330   | 0.3333   |
| RandomForest (tuned)              | Ограничение глубины                       | 0.3326      | 0.3326   | 0.3330   |
| RandomForest (default)            | Ансамбль > одиночное дерево               | 0.3315      | 0.3315   | 0.3317   |
| LightGBM + PCA(95%)              | Удаление шума через PCA                   | 0.3274      | 0.3274   | 0.3277   |
| XGBoost                           | Бустинг лучше RF                          | 0.2671      | 0.2666   | 0.3330   |
| LightGBM                          | Leaf-wise быстрее XGBoost                 | 0.2205      | 0.2197   | 0.3348   |

**Финальная модель:** Voting Ensemble (RF + XGBoost + LightGBM, soft voting) — Weighted F1 = 0.337


## Отчёт

Финальный отчёт: [`report/report.md`](report/report.md)


## Предотвращение утечки данных

Пайплайн предобработки был обновлен для исключения утечки данных:

- Сырые данные разделяются на train/validation/test до предобработки.
- Статистика пропущенных значений вычисляется только на обучающей части.
- Пороги выбросов определяются только на обучающей части.
- Масштабирование признаков настраивается только на обучающей части.
- Пороги инженерии признаков повторно используются последовательно для наборов валидации и тестирования.

Это гарантирует, что никакая информация из данных валидации или тестирования не попадает в обучение модели.
