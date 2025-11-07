# импорты
from typing import Tuple, Dict
import pandas as pd
from scipy.sparse import csr_matrix


def prepare_knn_matrix(
    df: pd.DataFrame,
    threshold: float = 0.0
) -> Tuple[csr_matrix, Dict[int, int], Dict[int, int]]:
    """
    Преобразует df в sparse матрицу взаимодействий для KNN-моделей на основе implicit feedback.

    Args:
        df (pd.DataFrame): Данные с колонками ['user_id', 'item_id', 'rating']
        threshold (float): Минимальный рейтинг для учета взаимодействия (по умолчанию 0.0)

    Returns:
        Tuple[csr_matrix, Dict[int, int], Dict[int, int]]:
            - interaction_matrix: csr_matrix (users × items)
            - user_to_index: словарь user_id → индекс
            - item_to_index: словарь item_id → индекс
    """
    # Фильтрация по порогу
    implicit_df = df[df['rating'] > threshold].copy()

    # Построение словарей индексации
    user_to_index = {user_id: idx for idx, user_id in enumerate(implicit_df['user_id'].unique())}
    item_to_index = {item_id: idx for idx, item_id in enumerate(implicit_df['item_id'].unique())}

    # ID в индексы
    row_indices = implicit_df['user_id'].map(user_to_index).values
    col_indices = implicit_df['item_id'].map(item_to_index).values

    # Все взаимодействия — единицы
    data = [1] * len(implicit_df)

    # sparse матрица
    interaction_matrix = csr_matrix(
        (data, (row_indices, col_indices)),
        shape=(len(user_to_index), len(item_to_index))
    )

    return interaction_matrix, user_to_index, item_to_index
