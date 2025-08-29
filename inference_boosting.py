import pandas as pd
import joblib
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix
)

# ==============================================================================
# 1. КОНФИГУРАЦИЯ
# ==============================================================================
# --- Укажите ID запуска (run_id) модели, которую хотите протестировать ---
RUN_ID_TO_TEST = "813f9a094f354b629526e091f11f56f9"

# --- Пути к данным (аналогично обучающему скрипту) ---
CONTRACTS_PARQUET_PATH = 'ML test task v3/test_task.parquet'
CONTEXT_CSV_PATH = 'ML test task v3/context_df.csv'

# --- Системные константы ---
MLFLOW_TRACKING_URI = f"file:{os.path.join(os.getcwd(), 'mlruns')}"

# Игнорируем технические предупреждения для чистоты вывода
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning)


# ==============================================================================
# 2. ФУНКЦИЯ ПОЛНОЙ ПОДГОТОВКИ ДАННЫХ
# ==============================================================================

def load_and_prepare_data():
    """
    Загружает и подготавливает данные, полностью повторяя логику обучающего скрипта.
    Возвращает подготовленные X, y и LabelEncoder.
    """
    print("\n[Шаг 1] Загрузка и подготовка данных...")
    macro_df = pd.read_csv(CONTEXT_CSV_PATH)
    contracts_df = pd.read_parquet(CONTRACTS_PARQUET_PATH)

    # --- Обработка макроэкономических данных ---
    macro_df['context_data_from'] = pd.to_datetime(macro_df['context_data_from'])
    
    def clean_and_convert_numeric(series):
        return pd.to_numeric(
            series.astype(str).str.replace('%', '', regex=False).str.replace(',', '.', regex=False),
            errors='coerce'
        )

    numeric_macro_features = [
        'inflation', 'key_rate', 'deposit_1', 'deposit_3', 'deposit_6', 'deposit_12',
        'fa_delta', 'usd_delta', 'IMOEX_delta', 'RGBI_delta'
    ]
    for col in numeric_macro_features:
        macro_df[col] = clean_and_convert_numeric(macro_df[col])
    macro_df.dropna(subset=numeric_macro_features, inplace=True)

    # --- Создание лагов и дельт ---
    macro_df_for_merge = macro_df.rename(columns={'context_data_from': 'quarter_start_date'}).sort_values('quarter_start_date')
    engineered_macro_features = numeric_macro_features.copy()
    for feature in numeric_macro_features:
        for lag in [1, 2]:
            lag_feature_name = f'{feature}_lag{lag}'
            macro_df_for_merge[lag_feature_name] = macro_df_for_merge[feature].shift(lag)
            engineered_macro_features.append(lag_feature_name)
        delta_feature_name = f'{feature}_delta1'
        macro_df_for_merge[delta_feature_name] = macro_df_for_merge[feature] - macro_df_for_merge[f'{feature}_lag1']
        engineered_macro_features.append(delta_feature_name)

    # --- Обработка данных о контрактах ---
    date_col = 'Договор Дата Заключения'
    contracts_df[date_col] = pd.to_datetime(contracts_df[date_col])
    contracts_df['quarter_start_date'] = contracts_df[date_col].dt.to_period('Q').dt.start_time

    # --- Создание временных признаков ---
    contracts_df['contract_year'] = contracts_df[date_col].dt.year
    contracts_df['contract_quarter'] = contracts_df[date_col].dt.quarter
    contracts_df['contract_month'] = contracts_df[date_col].dt.month
    contracts_df['contract_dayofweek'] = contracts_df[date_col].dt.dayofweek
    categorical_date_features = ['contract_year', 'contract_quarter', 'contract_month', 'contract_dayofweek']

    # --- Объединение и очистка ---
    merged_df = pd.merge(contracts_df, macro_df_for_merge, on='quarter_start_date', how='left')
    all_features_for_model = engineered_macro_features + categorical_date_features
    merged_df.dropna(subset=all_features_for_model + ['cus_class'], inplace=True)

    # --- Кодирование целевой переменной ---
    label_encoder = LabelEncoder()
    merged_df['cus_class_encoded'] = label_encoder.fit_transform(merged_df['cus_class'])
    
    X = merged_df[all_features_for_model]
    y = merged_df['cus_class_encoded']
    
    print("   Данные успешно подготовлены.")
    return X, y, label_encoder


# ==============================================================================
# 3. ФУНКЦИЯ ОЦЕНКИ МОДЕЛИ
# ==============================================================================

def evaluate_model(model, X_test, y_test, model_name, class_names):
    """Оценивает модель, выводит метрики и показывает матрицу ошибок."""
    print(f"\n[Шаг 4] Оценка модели '{model_name}' на тестовой выборке...")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    balanced_acc = balanced_accuracy_score(y_test, y_pred)

    print("\n--- Результаты ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-score (Macro): {f1_macro:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

    # Создание и отображение матрицы ошибок
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Матрица ошибок - Модель {model_name} (run_id: {RUN_ID_TO_TEST[:8]})')
    plt.xlabel('Предсказанный класс')
    plt.ylabel('Истинный класс')
    plt.show()


# ==============================================================================
# 4. ОСНОВНОЙ СКРИПТ
# ==============================================================================

if __name__ == "__main__":
    if RUN_ID_TO_TEST == "ВАШ_RUN_ID_ЗДЕСЬ":
        print("ОШИБКА: Пожалуйста, укажите реальный RUN_ID в переменной RUN_ID_TO_TEST.")
    else:
        try:
            # --- Подготовка данных и разделение на выборки ---
            X, y, le = load_and_prepare_data()
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # --- Загрузка модели из MLflow ---
            print(f"\n[Шаг 2] Загрузка артефактов для run_id: {RUN_ID_TO_TEST}")
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            
            # Определяем тип модели из параметров запуска
            client = mlflow.tracking.MlflowClient()
            run_info = client.get_run(RUN_ID_TO_TEST)
            model_type = run_info.data.params.get("model_type", "unknown").lower()
            
            model_uri = f"runs:/{RUN_ID_TO_TEST}/model"
            if model_type == 'lightgbm':
                model = mlflow.lightgbm.load_model(model_uri)
            elif model_type == 'xgboost':
                model = mlflow.xgboost.load_model(model_uri)
            elif model_type == 'catboost':
                model = mlflow.catboost.load_model(model_uri)
            else:
                raise ValueError(f"Неподдерживаемый тип модели: '{model_type}'.")
            
            print(f"   Модель '{model_type}' успешно загружена.")

            # --- Особенность для XGBoost ---
            if model_type == 'xgboost':
                print("\n[Шаг 3] Преобразование признаков для XGBoost...")
                categorical_features = ['contract_year', 'contract_quarter', 'contract_month', 'contract_dayofweek']
                for col in categorical_features:
                    if col in X_test.columns:
                        X_test[col] = X_test[col].astype("category")

            # --- Оценка модели ---
            evaluate_model(model, X_test, y_test, model_type.upper(), le.classes_.astype(str))
            
            print("\nТестирование завершено.")

        except Exception as e:
            print(f"\n[КРИТИЧЕСКАЯ ОШИБКА]: {e}")