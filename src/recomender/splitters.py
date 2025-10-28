import pandas as pd
from typing import List, Tuple

def df_time_split(df: pd.DataFrame,
                  time_column: str,
                  columns_to_save: List[str],
                  user_id: str = 'user_id',
                  item_id: str = 'item_id',
                  min_n_reitings: int = 5,
                  n: int = 15,
                  k: int = 10,
                  quantile: float = 0.85
                  ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Time-split  df -> (v_train, val)

    Args:
        df: входной датафрейм содержит  time_column, user_id, item_id.
        time_column: время для сплита.
        columns_to_save:колонки для выходных df.
        user_id, item_id: user_id, item_id.
        min_n_reitings: минимальное число рейтингов для "тёплого" юзера.
        n: минимальное число рейтингов тест_пользователя в трэйн.
        k: минимальное число рейтингов тест_пользователя в тест.
        quantile: порог деления.

    Returns:
        train, test  (все users и items из test присутствуют в train
            все users и items в train - разогретые.)
    """
    if time_column not in df.columns:
        raise ValueError(f"time_column {time_column} not found in df.columns")
    if user_id not in df.columns:
        raise ValueError(f"user_column {user_id} not found in df.columns")
    if item_id not in df.columns:
        raise ValueError(f"item_column {item_id} not found in df.columns")

    # limit
    time_treshold = df[time_column].quantile(q=quantile, interpolation='nearest')
    print(f"Порог разбиения по времени {time_treshold}")

    train_df = df[df[time_column] <= time_treshold][columns_to_save]
    test_df = df[df[time_column] > time_treshold][columns_to_save]
    print(f"Размеры после разбиения: train {train_df.shape[0]} test {test_df.shape[0]}")
    if train_df[time_column].max()>= test_df[time_column].min():
        raise ValueError("!!! Achtung!!! Train test данные пересекаются по времени.")

    # сколько у каждого юзера оценок
    user_counts = train_df[user_id].value_counts()

    # список "теплых" из train
    warm_users = user_counts[user_counts >= min_n_reitings].index

    # выбрасываем холодных
    warm_train_df = train_df[train_df[user_id].isin(warm_users)]

    print(f"\nОсталось пользователей в train : {len(warm_users)}")
    print(f"Новый размер train: {warm_train_df.shape}")

    # оставляем в test только item'ы, которые есть в warm_train
    train_items = set(warm_train_df[item_id].unique())
    good_test_df = test_df[test_df[item_id].isin(train_items)]

    # оставляем в test только тех у кого есть не менее 10 событий в test и
    # не менее 15 в трайн
    train_n_rait = warm_train_df[user_id].value_counts()
    test_n_rait = good_test_df[user_id].value_counts()

    good_users = (set(train_n_rait[train_n_rait >= n].index)
                  & set(test_n_rait[test_n_rait >= k].index))

    good_test_df = good_test_df[good_test_df[user_id].isin(good_users)]


    print(f"Осталось пользователей в test: {len(good_users)}")
    print(f"Новый размер test: {good_test_df.shape}")

    if len(good_users) == 0:
        raise ValueError("ACHTUNG!!! Отфильтровали, так уж отфильтровали...\
                         В test не осталось юзеров...")

    return warm_train_df.reset_index(drop=True),  good_test_df.reset_index(drop=True)