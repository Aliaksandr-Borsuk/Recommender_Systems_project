import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple
from IPython.display import display

def save_experiment_results(
        result: pd.DataFrame,
        model_name: str,
        meta: Dict[str, Any],
        results_dir: Path,
        verbosity: bool = True
    )-> Tuple[Dict[str, Any], Path, Path]:

    # Сохранение результатов оценки модели
    # Создаем директорию для результатов
    results_dir.mkdir(exist_ok=True)
    # Генерируем уникальное имя файла с временной меткой
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Подготавливаем результаты для сохранения
    results_data = {
        'model_name': model_name,
        'evaluation_date': datetime.now().isoformat(),
        'metrics': {
            'hit_rate@10': result.iloc[0]['hit_rate@10'],
            'precision@10': result.iloc[0]['precision@10'],
            'recall@10': result.iloc[0]['recall@10'],
            'ndcg@10': result.iloc[0]['ndcg@10'],
            'map@10': result.iloc[0]['map@10'],
            'coverage@10': result.iloc[0]['coverage@10']
        },
        'parameters': {
            'k': meta.get('k'),
            'min_train_interactions': meta['min_train_interactions'],
            'min_test_interactions': meta['min_test_interactions']
        },
        'dataset_info': {
            'n_train_users': meta['n_train_users'],
            'n_test_users': meta['n_test_users'],
            'n_items': meta['n_items'],
            'train_size': meta['train_shape'][0],
            'test_size': meta['test_shape'][0]
        }
    }

    # Сохраняем результаты в JSON с уникальным именем
    results_json_file = results_dir / f'{model_name}_{timestamp}.json'
    with open(results_json_file, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False, default=str)

    # Для CSV создаем/обновляем сводную таблицу всех экспериментов
    results_csv_file = results_dir / 'all_experiments_results.csv'
    # Создаем DataFrame для текущего результата с временной меткой и именем модели
    current_result = result.reset_index().rename(columns={'index': 'model_name'})
    current_result['timestamp'] = timestamp
    current_result['evaluation_date'] = results_data['evaluation_date']

    if results_csv_file.exists():
        # Если файл существует, читаем и добавляем новую строку
        existing_results = pd.read_csv(results_csv_file)
        updated_results = pd.concat([existing_results, current_result], ignore_index=True)
        updated_results.to_csv(results_csv_file, index=False)
        print("Результат добавлен в существующий CSV файл")
    else:
        # Если файла нет, создаем новый
        current_result.to_csv(results_csv_file, index=False)
        print("Создан новый CSV файл с результатами")

    print(f"JSON результат сохранен как: {results_json_file.name}")
    print(f"CSV со всеми экспериментами: {results_csv_file.name}")
    print(f"Все результаты в: {results_dir}")

    if verbosity:
        # Выводим сводку по эксперименту
        print("\n" + "="*50)
        print("СВОДКА ЭКСПЕРИМЕНТА")
        print("="*50)
        print(f"Модель: {results_data['model_name']}")
        print(f"Метка времени: {timestamp}")
        print(f"Дата оценки: {results_data['evaluation_date'][:19]}")
        print(f"Размер train: {results_data['dataset_info']['train_size']:,}")
        print(f"Размер test: {results_data['dataset_info']['test_size']:,}")
        print(f"Пользователей в test: {results_data['dataset_info']['n_test_users']}")
        print(f"Уникальных предметов: {results_data['dataset_info']['n_items']}")
        print(f"HitRate@10: {results_data['metrics']['hit_rate@10']:.1%}")
        print(f"precision@10: {results_data['metrics']['precision@10']:.2%}")
        print(f"recall@10: {results_data['metrics']['recall@10']:.2%}")
        print(f"ndcg@10: {results_data['metrics']['ndcg@10']:.2%}")
        print(f"map@10: {results_data['metrics']['map@10']:.2%}")
        print(f"Coverage@10: {results_data['metrics']['coverage@10']:.2%}")
        print("="*50)

        # Дополнительно: показываем последние 5 экспериментов если CSV существует
        if results_csv_file.exists():
            all_results = pd.read_csv(results_csv_file)
            print(f"\nПоследние эксперименты ({len(all_results)} всего):")
            display(all_results.tail(5))
    return results_data, results_json_file, results_csv_file
