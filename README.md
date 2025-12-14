# **Проект: Рекомендательные системы на MovieLens 1m**  
  
**Коротко:** набор ноутбуков Colab для поэтапной разработки и сравнения классических и нейросетевых рекомендательных алгоритмов на датасете MovieLens (1m / ml-latest).  
**Цель** - собрать воспроизводимый пайплайн от EDA и baseline'ов до гибридных two-tower моделей с чёткой метрикой и выводами.  
  
---
  
## **Структура репозитория** 
docs/ 
- theory.md  - лайтовая теория рексистем

data/ - сохранённые предобработанные датасеты (train/test, метаданные...) или в GoogleDrive

notebooks/

- **01_data_and_eda.ipynb - загрузка, EDA, предобработка, хронологическое разбиение, popularity baseline (готов)**

- **02_knn_like.ipynb - ItemKNN, UserKNN, подбор гиперпараметров - Optuna, сравнение метрик(готов)**

- **03_matrix_factorization - ALS  и SVD**
- - **03_01_matrix_factorization_SVD.ipynb(готов)**
  - **03_02_matrix_factorization_ALS_CLEAN.ipynb(готов)**  ( с очисткой метадаты)

- **04_linear_models_ease_slim.ipynb - EASE (closed form) и  SLIM**
- - **04_01_linear_models_EASE_CLEAN.ipynb(готов  с очисткой метадаты)**
  - **04_02_linear_models_SLIM.ipynb(готов)**

- 05_implicit_evaluation_and_splits.ipynb - implicit-метрики, negative sampling, split’ы, cold-start

- 06_ncf_pytorch.ipynb -  Neural Collaborative Filtering на PyTorch, BPR/BCE

- 07_two_tower_hybrid_pytorch.ipynb - two-tower гибрид с простым text encoder

- 08_results_and_conclusions.ipynb - сводные метрики, графики, выводы

- src/ - вспомогательные модули (метрики, даталоадеры, модели, utils)

pyproject.toml для poetry

README.md - этот файл  

---

## **Быстрый старт (Colab)**   
Клонировать репозиторий в Colab: в Colab выберите "GitHub" и вставьте URL репо.  
  
Откройте ноутбук 01_data_and_eda.ipynb и выполните ячейки последовательно.  
  
В 01 ноутбуке скачивается MovieLens (ml-1m / ml-latest), выполняется очистка и сохраняются готовые train/test split
в папку /content/drive/MyDrive/Colab Notebooks/data/processed/.  
  
Для последующих ноутбуков просто подключитесь к тому же репозиторию и загрузите  Google Drive, где сохранены артефакты из 01.  

---  

## **Основные технические решения**
Датасет: MovieLens 1m . Предобработка в 01 включает: удаление аномалий, приведение timestamp к datetime,
хронологическое разбиение по времени (train - все события до timelimit, test - последующие).

Метрики: precision@K, recall@K, MAP@K, NDCG@K и hit-rate.

Pipeline Top-N: два шага - генерация кандидатов (popularity / KNN / MF / EASE / NCF) → ранжирование (полный скоринг топ-кандидатов).

Reproducibility: фиксируем seed, сохраняем параметры split'ов.  

---  


## Лицензия   
Лицензия: **MIT**
