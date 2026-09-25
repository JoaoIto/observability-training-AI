"""
Módulo de Ingestão e Pré-processamento de Dados
Atividade Prática Avaliativa de Inteligência Artificial - UNITINS
Aluno: João Victor Ito | Professor: Marco Antonio Firmino de Sousa

Responsabilidades:
1. Ingestão automatizada do dataset via kagglehub com fallback local.
2. Mapeamento da variável alvo ('Dropout': 0, 'Enrolled': 1, 'Graduate': 2).
3. Conversão de variáveis preditivas para float32 e validação de schema.
4. Particionamento em 3 vias estritas (60% Treino, 20% Validação, 20% Teste Reservado) com SEED=42 e estratificação.
5. Normalização com StandardScaler (fit apenas no treino, transform em validação e teste).
6. Persistência de artefatos locais (scaler.pkl e resumo_pre_processamento.json) em artifacts/.
"""

import os
import json
import logging
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATASET_HANDLE = "thedevastator/higher-education-predictors-of-student-retention"
TARGET_MAPPING = {"Dropout": 0, "Enrolled": 1, "Graduate": 2}
TARGET_NAMES = ["Dropout", "Enrolled", "Graduate"]


def ingest_raw_data(data_path: str = None) -> pd.DataFrame:
    """
    Ingere o dataset oficial utilizando kagglehub ou busca arquivo local dataset.csv.
    """
    # 1. Verifica se foi passado um caminho explícito ou se dataset.csv existe localmente
    candidates = [data_path, "dataset.csv", os.path.join("artifacts", "dataset.csv")]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            logger.info(f"Carregando dataset a partir do caminho local: {candidate}")
            return pd.read_csv(candidate)

    # 2. Ingestão automatizada via kagglehub
    logger.info(f"Iniciando download automatizado do dataset via kagglehub: {DATASET_HANDLE}")
    try:
        import kagglehub
        download_dir = kagglehub.dataset_download(DATASET_HANDLE)
        csv_file = os.path.join(download_dir, "dataset.csv")
        if not os.path.exists(csv_file):
            # Procura qualquer arquivo CSV no diretório baixado
            for f in os.listdir(download_dir):
                if f.endswith(".csv"):
                    csv_file = os.path.join(download_dir, f)
                    break

        logger.info(f"Dataset obtido com sucesso em: {csv_file}")
        return pd.read_csv(csv_file)
    except Exception as e:
        logger.error(f"Falha na ingestão via kagglehub: {e}")
        raise RuntimeError(
            f"Não foi possível baixar o dataset via kagglehub e nenhum arquivo local foi encontrado: {e}"
        )


def load_and_preprocess_data(
    data_path: str = None,
    artifacts_dir: str = "artifacts",
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, Dict[str, Any]]:
    """
    Executa o pipeline completo de pré-processamento, split em 3 vias e salvamento de artefatos.

    Retorna:
        X_train, y_train, X_val, y_val, X_test, y_test, scaler, metadata
    """
    os.makedirs(artifacts_dir, exist_ok=True)

    # 1. Ingestão
    df = ingest_raw_data(data_path)
    total_samples_raw = len(df)
    logger.info(f"Dataset bruto carregado com {total_samples_raw} linhas e {df.shape[1]} colunas.")

    if "Target" not in df.columns:
        raise ValueError("A coluna obrigatória 'Target' não foi encontrada no dataset.")

    # 2. Tratamento do Target
    # Se já não for numérico, realiza o mapeamento
    if df["Target"].dtype == object or isinstance(df["Target"].iloc[0], str):
        df["Target"] = df["Target"].map(TARGET_MAPPING)

    if df["Target"].isnull().any():
        raise ValueError("Valores nulos identificados na coluna 'Target' após mapeamento.")

    # 3. Separação de Atributos e Target
    X_df = df.drop(columns=["Target"])
    y_series = df["Target"]

    feature_names = list(X_df.columns)
    num_features = len(feature_names)

    # Garantir que todos os atributos sejam float32
    X = X_df.astype(np.float32).values
    y = y_series.values.astype(np.int64)

    # 4. Divisão em 3 vias estritas (60% Treino, 20% Validação, 20% Teste Reservado)
    # Primeiro split: 80% (Treino + Validação) e 20% Teste Reservado
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=random_state,
        stratify=y,
    )

    # Segundo split: Dos 80%, separar 75% para Treino (60% do total) e 25% para Validação (20% do total)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.25,
        random_state=random_state,
        stratify=y_train_val,
    )

    # 5. Normalização com StandardScaler (Fit estritamente no Treino)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_val_scaled = scaler.transform(X_val).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    # 6. Geração de Artefatos Locais
    scaler_path = os.path.join(artifacts_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    logger.info(f"Scaler salvo com sucesso em: {scaler_path}")

    # Distribuição de classes
    def get_class_dist(labels: np.ndarray) -> Dict[str, int]:
        counts = np.bincount(labels, minlength=3)
        return {TARGET_NAMES[i]: int(counts[i]) for i in range(3)}

    metadata = {
        "dataset_name": "Predict Students' Dropout and Academic Success",
        "total_instances": total_samples_raw,
        "num_features": num_features,
        "feature_names": feature_names,
        "target_classes": TARGET_MAPPING,
        "target_class_names": TARGET_NAMES,
        "partition_sizes": {
            "train": int(len(X_train)),
            "validation": int(len(X_val)),
            "test_reserved": int(len(X_test)),
        },
        "partition_percentages": {
            "train": "60%",
            "validation": "20%",
            "test_reserved": "20%",
        },
        "class_distribution_overall": get_class_dist(y),
        "class_distribution_train": get_class_dist(y_train),
        "class_distribution_val": get_class_dist(y_val),
        "class_distribution_test": get_class_dist(y_test),
        "random_state": random_state,
        "scaling_algorithm": "StandardScaler",
    }

    resumo_path = os.path.join(artifacts_dir, "resumo_pre_processamento.json")
    with open(resumo_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
    logger.info(f"Resumo do pré-processamento salvo em: {resumo_path}")

    logger.info(
        f"Partições prontas: Treino={X_train_scaled.shape}, Val={X_val_scaled.shape}, Teste={X_test_scaled.shape}"
    )

    return X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test, scaler, metadata


def get_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 32,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Empacota arrays numpy em TensorDatasets e DataLoaders do PyTorch.
    """
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    print("--- Teste Unitário de dataset.py ---")
    X_tr, y_tr, X_v, y_v, X_te, y_te, sc, meta = load_and_preprocess_data()
    tr_loader, v_loader, te_loader = get_dataloaders(X_tr, y_tr, X_v, y_v, X_te, y_te)
    print("Sucesso! Batch de treino:", next(iter(tr_loader))[0].shape)
