# импорты
import math
from typing import Dict, List, Set
import pandas as pd

def hitrate_at_k(rec_items: Dict[int, List[int]],
                 test_items: Dict[int, Set[int]],
                 k: int = 10) -> float:
    '''
    HitRate@K
    rec_items - топ-K items
    test_items - item-ы поюзерно из test
    '''
    if rec_items.keys() != test_items.keys():
        raise ValueError("Achtung!!!\nРазный набор юзеров в rec_items и test_items")
    hr = sum(1 for us, recs in rec_items.items() if set(recs[:k]) & test_items[us])
    return hr / len(test_items)

def coverage_at_k(rec_items: Dict[int, List[int]],
                  all_items: Set[int],
                  k: int = 10) -> float:
    '''
    Coverage@K
    rec_items: словарь user_id -> list of item_id (top-K)
    '''
    rec_items_set = {item for recs in rec_items.values() for item in recs[:k]}
    return len(rec_items_set) / len(all_items)

def precision_at_k(rec_items: Dict[int, List[int]],
                   test_items: Dict[int, Set[int]],
                   k: int = 10) -> float:
    '''
    Precision@K
    rec_items - словарь user_id -> list of item_id (top-K)
    test_items - item-ы поюзерно из test
    '''
    if rec_items.keys() != test_items.keys():
        raise ValueError("Achtung!!! precision_at_k\nРазный набор юзеров")
    total = sum(len(set(recs[:k]) & test_items[u]) for u, recs in rec_items.items())
    return total / (k * len(test_items))

def recall_at_k(rec_items: Dict[int, List[int]],
                test_items: Dict[int, Set[int]],
                k: int = 10) -> float:
    '''
    Recall@K
    '''
    if rec_items.keys() != test_items.keys():
        raise ValueError("Achtung!!! recall_at_k\nРазный набор юзеров")
    total = 0.0
    for u, recs in rec_items.items():
        R_u = len(test_items[u])
        # можно: считать как 0 или исключать из среднего;
        # здесь исключаем из суммы и учёта
        # но так как у нас R_u всегда >=10, то это лишнее
        if R_u == 0:
            print('Achtung!!! recall_at_k\nВ test есть юзеры с 0 items !!!')
            continue
        topk = recs[:k]
        hits = len(set(topk) & test_items[u])
        total += hits / R_u
    # нормируем на число пользователей с R_u>0
    num_with_pos = sum(1 for s in test_items.values() if len(s) > 0)
    return total / num_with_pos if num_with_pos > 0 else 0.0

def ndcg_at_k(rec_items: Dict[int, List[int]],
              test_items: Dict[int, Set[int]],
              k: int = 10) -> float:
    '''
    NDCG@K с бинарным ранжированием
    '''
    if rec_items.keys() != test_items.keys():
        raise ValueError("Achtung!!! ndcg_at_k\nРазный набор юзеров")

    def dcg(rels: List[int]) -> float:
        return sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(rels))

    total_ndcg, num_users = 0.0, 0
    for u, recs in rec_items.items():
        R_u = len(test_items[u])
        if R_u == 0:
            print('Achtung!!! ndcg_at_k\nВ test есть юзеры с 0 items !!!')
            continue
        rels = [1 if item in test_items[u] else 0 for item in recs[:k]]
        user_dcg = dcg(rels)
        user_idcg = dcg([1] * min(R_u, k))
        total_ndcg += user_dcg / user_idcg if user_idcg > 0 else 0.0
        num_users += 1
    return total_ndcg / num_users if num_users > 0 else 0.0

def map_at_k(rec_items: Dict[int, List[int]],
             test_items: Dict[int, Set[int]],
             k: int = 10) -> float:
    '''
    MAP@K (Mean Average Precision @ K) с бинарной релевантностью.
    AP@K(u) = 1/min(R_u,K) * sum_{i=1..K} Precision@i(u) * rel_i
    '''
    if rec_items.keys() != test_items.keys():
        raise ValueError("Achtung!!! map_at_k\nРазный набор юзеров")

    total_ap, num_users = 0.0, 0
    for u, recs in rec_items.items():
        R_u = len(test_items[u])
        if R_u == 0:
            print('Achtung!!! map_at_k\nВ test есть юзеры с 0 items !!!')
            continue
        hits, sum_prec = 0, 0.0
        for i, item in enumerate(recs[:k], start=1):
            if item in test_items[u]:
                hits += 1
                sum_prec += hits / i
        denom = min(R_u, k)
        total_ap += sum_prec / denom if denom > 0 else 0.0
        num_users += 1
    return total_ap / num_users if num_users > 0 else 0.0


def model_evaluation(rec_items: Dict[int, List[int]],           # рекомендации поюзерно
                              test_items: Dict[int, Set[int]],  # item-ы поюзерно из test
                              all_items: Set[int],
                              k: int = 10,
                              model_name: str = "model") -> pd.DataFrame:
    """
    Возвращает результат в pandas.DataFrame
    """
    hr = hitrate_at_k(rec_items, test_items, k=k)
    pr_k = precision_at_k(rec_items, test_items, k=k)
    rec_k = recall_at_k(rec_items, test_items, k=k)
    ndcg = ndcg_at_k(rec_items, test_items, k=k)
    map_k = map_at_k(rec_items, test_items, k=k)
    cov = coverage_at_k(rec_items, all_items, k=k)
    df = pd.DataFrame(
        [[hr, pr_k, rec_k, ndcg, map_k, cov]],
        index=[model_name],
        columns=[f"hit_rate@{k}", f"precision@{k}", f"recall@{k}",
                 f"ndcg@{k}", f"map@{k}", f"coverage@{k}"]
    )
    return df