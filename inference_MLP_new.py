import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import mlflow
import pickle
import json
import os
import warnings

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path

# ==============================================================================
# 1. КОНФИГУРАЦИЯ
# ==============================================================================
# --- Укажите ID запуска (run_id) модели, которую хотите протестировать ---
RUN_ID_TO_TEST = "7a004f7cffd14fe396c485a320d11f18"

# --- Пути к данным (аналогично обучающему скрипту) ---
CONTRACTS_PARQUET_PATH = Path('ML test task v3/test_task.parquet')
CONTEXT_CSV_PATH = Path('ML test task v3/context_df.csv')

# --- Системные константы ---
MLFLOW_TRACKING_URI = f"file:{os.path.join(os.getcwd(), 'mlruns')}"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Игнорируем технические предупреждения
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning)


# ==============================================================================
# 2. ОПРЕДЕЛЕНИЕ АРХИТЕКТУРЫ МОДЕЛИ (для загрузки)
# ==============================================================================
# Класс модели должен быть определен, чтобы PyTorch мог корректно загрузить веса
class Classifier(nn.Module):
    def __init__(self, num_features, num_classes):
        super(Classifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )
    def forward(self, x):
        return self.network(x)

# ==============================================================================
# 3. ФУНКЦИЯ ПОЛНОЙ ПОДГОТОВКИ ДАННЫХ
# ==============================================================================

def load_and_prepare_data_nn():
    """
    Загружает и подготавливает данные, полностью повторяя логику обучающего скрипта для нейросети.
    """
    print("\n[Шаг 1] Загрузка и подготовка данных...")
    macro_df = pd.read_csv(CONTEXT_CSV_PATH)
    contracts_df = pd.read_parquet(CONTRACTS_PARQUET_PATH)

    # Предобработка макроэкономической таблицы
    macro_df.columns = macro_df.columns.str.lower().str.replace(' ', '_')
    macro_df['context_data_from'] = pd.to_datetime(macro_df['context_data_from'])
    percent_cols = ['inflation', 'key_rate', 'deposit_1', 'deposit_3', 'deposit_6', 'deposit_12', 'fa_delta', 'usd_delta', 'imoex_delta', 'rgbi_delta']
    for col in percent_cols:
        macro_df[col] = pd.to_numeric(macro_df[col].str.replace('%', ''), errors='coerce')
    macro_df.ffill(inplace=True)
    macro_df.bfill(inplace=True)

    # Предобработка таблицы контрактов
    contracts_df.rename(columns={'Договор Дата Заключения': 'contract_date'}, inplace=True)
    contracts_df['cus_class'] = contracts_df['cus_class'].astype(int)

    # Объединение таблиц и создание признаков
    merged_df = pd.merge_asof(
        contracts_df.sort_values('contract_date'),
        macro_df.sort_values('context_data_from'),
        left_on='contract_date',
        right_on='context_data_from',
        direction='backward'
    ).dropna(subset=macro_df.columns)
    merged_df['day_of_year'] = merged_df['contract_date'].dt.dayofyear
    merged_df['day_of_week'] = merged_df['contract_date'].dt.dayofweek
    merged_df['month'] = merged_df['contract_date'].dt.month

    # Агрегация классов
    def aggregate_cus_class(c):
        if c in [1, 5, 8, 10, 4]: return 0  # Группа 'Base'
        if c in [101, 102, 103, 104, 105, 106, 107, 108, 109]: return 1  # Группа 'Premium'
        return 2  # Группа 'Rare'
    merged_df['cus_class_agg'] = merged_df['cus_class'].apply(aggregate_cus_class)

    features = [
        'quarter', 'inflation', 'key_rate', 'deposit_1', 'deposit_3', 'deposit_6',
        'deposit_12', 'fa_delta', 'usd_delta', 'imoex_delta', 'rgbi_delta',
        'day_of_year', 'day_of_week', 'month'
    ]
    X = merged_df[features]
    y = merged_df['cus_class_agg']
    
    print("   Данные успешно подготовлены.")
    return X, y

# ==============================================================================
# 4. ФУНКЦИЯ ОЦЕНКИ МОДЕЛИ
# ==============================================================================

def evaluate_model(model, data_loader, class_names):
    """Оценивает модель, выводит метрики и показывает матрицу ошибок."""
    print(f"\n[Шаг 4] Оценка модели на устройстве {DEVICE}...")
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            all_preds.extend(torch.max(outputs, 1)[1].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    print("\n--- Результаты тестирования ---")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Матрица ошибок - (run_id: {RUN_ID_TO_TEST[:8]})')
    plt.xlabel('Предсказанный класс')
    plt.ylabel('Истинный класс')
    plt.show()

# ==============================================================================
# 5. ОСНОВНОЙ СКРИПТ
# ==============================================================================

if __name__ == "__main__":
    if RUN_ID_TO_TEST == "ВАШ_RUN_ID_ЗДЕСЬ":
        print("ОШИБКА: Пожалуйста, укажите реальный RUN_ID в переменной RUN_ID_TO_TEST.")
    else:
        try:
            # Подготовка данных и разделение на выборки
            X, y = load_and_prepare_data_nn()
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # Загрузка артефактов из MLflow
            print(f"\n[Шаг 2] Загрузка артефактов для run_id: {RUN_ID_TO_TEST}")
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            
            local_scaler_path = mlflow.artifacts.download_artifacts(run_id=RUN_ID_TO_TEST, artifact_path="scaler.pkl")
            with open(local_scaler_path, "rb") as f: scaler = pickle.load(f)
                
            local_mapping_path = mlflow.artifacts.download_artifacts(run_id=RUN_ID_TO_TEST, artifact_path="class_mapping.json")
            with open(local_mapping_path, "r") as f: class_mapping = json.load(f)
            class_names = list(class_mapping.values())

            model_uri = f"runs:/{RUN_ID_TO_TEST}/model_pytorch"
            model = mlflow.pytorch.load_model(model_uri, map_location=DEVICE)
            print("   Модель и артефакты успешно загружены.")

            # Подготовка тестовой выборки
            print("\n[Шаг 3] Подготовка тестовой выборки...")
            X_test_scaled = scaler.transform(X_test)
            X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
            y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)
            
            test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
            test_loader = DataLoader(test_dataset, batch_size=64)

            # Оценка модели
            evaluate_model(model, test_loader, class_names)
            
            print("\nТестирование завершено.")

        except Exception as e:
            print(f"\n[КРИТИЧЕСКАЯ ОШИБКА]: {e}")