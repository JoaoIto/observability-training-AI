"""
Pipeline de Treinamento, Observabilidade e MLOps com PyTorch e MLflow
Atividade Prática Avaliativa de Inteligência Artificial - UNITINS
Aluno: João Victor Ito | Professor: Marco Antonio Firmino de Sousa

Destaques da Implementação:
1. Instrumentação de Tracing com Spans explícitos via MLflow Tracing:
   - Span 1: "Span_Preparar_Dados" (prepara batches, shapes e metadados).
   - Span 2: "Span_Treinamento" (60 épocas de treino, backpropagation e validação incremental).
   - Span 3: "Span_Validacao_Final" (avaliação final, persistência de checkpoints e artefatos).
   - Span 4: "Span_Avaliacao_Teste_Reservado" (avaliação estrita da Run Campeã em dados não vistos).
2. As 3 Configurações Experimentais Obrigatórias (Slide 10 da UNITINS):
   - Run A (Referência): lr = 0.01, weight_decay = 0.0
   - Run B (Taxa Menor): lr = 0.001, weight_decay = 0.0
   - Run C (Com Regularização L2): lr = 0.01, weight_decay = 1e-5
   - Todas sob semente fixa (SEED = 42), 60 épocas e batch_size = 32.
3. Regra de Ouro (Slide 12 da UNITINS):
   - Comparação estrita baseada nas curvas e métricas de Validação.
   - Seleção da Run Campeã.
   - Avaliação única e definitiva no conjunto de Teste Reservado.
"""

import os
import sys
import json
import random
import logging
import urllib.request
from typing import Dict, Any, Tuple, List

# Garantir codificação UTF-8 no Windows para evitar falhas com emojis do MLflow
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Suprimir mensagens de sugestão de agentes do MLflow
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

# Garantir que a raiz do projeto esteja no sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch

from src.dataset import load_and_preprocess_data, TARGET_NAMES
from src.model import build_model, StudentRetentionMLP

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("TrainPipeline")

EXPERIMENT_NAME = "UNITINS_IA_Student_Retention"
SEED = 42
EPOCHS = 60
BATCH_SIZE = 32
DROPOUT_RATE = 0.2
ARTIFACTS_DIR = "artifacts"


def set_seed(seed: int = 42):
    """
    Fixa as sementes aleatórias para garantir reprodutibilidade estrita.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_tracking_uri() -> str:
    """
    Determina o URI de rastreamento do MLflow priorizando:
    1. Variável de ambiente MLFLOW_TRACKING_URI, se definida.
    2. Servidor HTTP local (http://127.0.0.1:5000), caso esteja ativo.
    3. Banco SQLite local (sqlite:///mlflow.db).
    """
    if "MLFLOW_TRACKING_URI" in os.environ:
        return os.environ["MLFLOW_TRACKING_URI"]

    try:
        req = urllib.request.Request("http://127.0.0.1:5000", method="HEAD")
        with urllib.request.urlopen(req, timeout=1):
            return "http://127.0.0.1:5000"
    except Exception:
        pass

    db_path = os.path.abspath("mlflow.db")
    return f"sqlite:///{db_path}"


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Avalia o modelo sobre um DataLoader retornando loss média, acurácia,
    labels reais e previsões.
    """
    model.eval()
    total_loss = 0.0
    all_targets: List[int] = []
    all_preds: List[int] = []

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            total_loss += loss.item() * batch_x.size(0)

            preds = torch.argmax(logits, dim=1)
            all_targets.extend(batch_y.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

    avg_loss = total_loss / len(dataloader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    return avg_loss, acc, np.array(all_targets), np.array(all_preds)


def plot_and_save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    output_path: str,
    title: str = "Matriz de Confusão - Teste Reservado",
):
    """
    Gera e salva o gráfico da Matriz de Confusão com anotações e mapa de cores.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.8)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                x=j,
                y=i,
                s=str(cm[i, j]),
                va="center",
                ha="center",
                size="large",
                fontweight="bold",
                color="black" if cm[i, j] < cm.max() / 2 else "white",
            )

    fig.colorbar(cax)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=20, ha="left")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Classe Prevista", labelpad=10, fontweight="bold")
    ax.set_ylabel("Classe Real", labelpad=10, fontweight="bold")
    ax.set_title(title, pad=20, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


# -----------------------------------------------------------------------------
# SPANS DE RASTREAMENTO EXPLÍCITOS (MLflow Tracing)
# -----------------------------------------------------------------------------

@mlflow.trace(name="Span_Preparar_Dados")
def span_preparar_dados(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int,
    metadata: Dict[str, Any],
) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
    """
    Span 1: Mede o tempo de preparação de dados, instanciação dos tensores
    e particionamento dos DataLoaders para a run.
    """
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    summary = {
        "dataset_name": metadata["dataset_name"],
        "num_features": metadata["num_features"],
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "batch_size": batch_size,
        "batches_per_epoch": len(train_loader),
        "scaler_applied": metadata["scaling_algorithm"],
    }
    return train_loader, val_loader, summary


@mlflow.trace(name="Span_Treinamento")
def span_treinamento(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: torch.device,
    checkpoint_path: str,
    run_name: str,
) -> Dict[str, Any]:
    """
    Span 2: Mede as 60 épocas de treinamento, retropropagação e validação incremental.
    """
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_x.size(0)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_val_loss, epoch_val_acc, _, _ = evaluate_model(model, val_loader, criterion, device)

        # Log de Métricas por Época com step=epoch
        mlflow.log_metric("train_loss", epoch_train_loss, step=epoch)
        mlflow.log_metric("val_loss", epoch_val_loss, step=epoch)
        mlflow.log_metric("val_accuracy", epoch_val_acc, step=epoch)

        # Rastreia o melhor modelo segundo a menor val_loss
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_val_acc = epoch_val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint_path)

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            logger.info(
                f"[{run_name}] Época {epoch:02d}/{epochs:02d} | "
                f"Train Loss: {epoch_train_loss:.4f} | "
                f"Val Loss: {epoch_val_loss:.4f} | "
                f"Val Acc: {epoch_val_acc:.4f}"
            )

    return {
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 4),
        "best_val_acc": round(best_val_acc, 4),
        "final_train_loss": round(epoch_train_loss, 4),
        "total_epochs": epochs,
    }


@mlflow.trace(name="Span_Validacao_Final")
def span_validacao_final(
    input_dim: int,
    checkpoint_path: str,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    best_epoch: int,
    run_name: str,
) -> Dict[str, Any]:
    """
    Span 3: Mede a validação final carregando o melhor checkpoint e persistindo artefatos.
    """
    best_model = build_model(input_dim=input_dim, num_classes=3, dropout_rate=DROPOUT_RATE).to(device)
    best_model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    best_model.eval()

    final_val_loss, final_val_acc, y_val_true, y_val_preds = evaluate_model(
        best_model, val_loader, criterion, device
    )
    val_f1_weighted = f1_score(y_val_true, y_val_preds, average="weighted")
    val_f1_macro = f1_score(y_val_true, y_val_preds, average="macro")

    # Métricas consolidadas finais
    mlflow.log_metric("best_epoch", best_epoch)
    mlflow.log_metric("final_best_val_loss", final_val_loss)
    mlflow.log_metric("final_best_val_accuracy", final_val_acc)
    mlflow.log_metric("final_best_val_f1_weighted", val_f1_weighted)
    mlflow.log_metric("final_best_val_f1_macro", val_f1_macro)

    # Registro de Artefatos
    scaler_path = os.path.join(ARTIFACTS_DIR, "scaler.pkl")
    resumo_path = os.path.join(ARTIFACTS_DIR, "resumo_pre_processamento.json")
    standard_model_path = os.path.join(ARTIFACTS_DIR, "best_model.pt")
    torch.save(best_model.state_dict(), standard_model_path)

    mlflow.log_artifact(scaler_path, artifact_path="preprocessing")
    mlflow.log_artifact(resumo_path, artifact_path="preprocessing")
    mlflow.log_artifact(checkpoint_path, artifact_path="checkpoints")
    mlflow.log_artifact(standard_model_path, artifact_path="checkpoints")

    # Registro e Empacotamento do Modelo PyTorch
    mlflow.pytorch.log_model(
        pytorch_model=best_model,
        name="model",
        serialization_format="pickle",
    )

    return {
        "run_name": run_name,
        "best_epoch": best_epoch,
        "final_best_val_loss": round(final_val_loss, 4),
        "final_best_val_acc": round(final_val_acc, 4),
        "final_best_val_f1_weighted": round(val_f1_weighted, 4),
        "checkpoint": os.path.basename(checkpoint_path),
        "artifacts_logged": ["scaler.pkl", "resumo_pre_processamento.json", "checkpoints", "model"],
    }


@mlflow.trace(name="Span_Avaliacao_Teste_Reservado")
def span_avaliacao_teste_reservado(
    champion_run: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    input_dim: int,
    class_names: List[str] = TARGET_NAMES,
) -> Dict[str, Any]:
    """
    Span 4: Mede a avaliação única e definitiva no conjunto de Teste Reservado.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_model = build_model(input_dim=input_dim, num_classes=3, dropout_rate=DROPOUT_RATE).to(device)
    best_model.load_state_dict(torch.load(champion_run["checkpoint_path"], weights_only=True))
    best_model.eval()

    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    criterion = nn.CrossEntropyLoss()

    test_loss, test_acc, y_true, y_pred = evaluate_model(best_model, test_loader, criterion, device)
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    f1_macro = f1_score(y_true, y_pred, average="macro")

    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    report_text = classification_report(y_true, y_pred, target_names=class_names, digits=4)

    cm_path = os.path.join(ARTIFACTS_DIR, "matriz_confusao_teste.png")
    plot_and_save_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        output_path=cm_path,
        title=f"Matriz de Confusão ({champion_run['run_name']}) - Teste Reservado",
    )

    test_summary = {
        "champion_run": champion_run["run_name"],
        "champion_run_id": champion_run["run_id"],
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "test_f1_weighted": float(f1_weighted),
        "test_f1_macro": float(f1_macro),
        "classification_report": report_dict,
    }
    test_summary_path = os.path.join(ARTIFACTS_DIR, "relatorio_teste_campeao.json")
    with open(test_summary_path, "w", encoding="utf-8") as f:
        json.dump(test_summary, f, indent=4, ensure_ascii=False)

    return {
        "test_summary": test_summary,
        "report_text": report_text,
        "cm_path": cm_path,
        "test_summary_path": test_summary_path,
    }


def train_run(
    run_name: str,
    learning_rate: float,
    weight_decay: float,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    metadata: Dict[str, Any],
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    seed: int = SEED,
) -> Dict[str, Any]:
    """
    Executa uma run de treinamento completa com rastreamento e os 3 spans explícitos.
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = metadata["num_features"]

    logger.info(f"\n=======================================================")
    logger.info(f"Iniciando {run_name} | lr={learning_rate} | L2={weight_decay}")
    logger.info(f"=======================================================")

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id

        # SPAN 1: Span_Preparar_Dados
        train_loader, val_loader, data_summary = span_preparar_dados(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            batch_size=batch_size,
            metadata=metadata,
        )

        # Log dos parâmetros no MLflow
        params = {
            "run_name": run_name,
            "learning_rate": learning_rate,
            "weight_decay_l2": weight_decay,
            "epochs": epochs,
            "batch_size": batch_size,
            "dropout_rate": DROPOUT_RATE,
            "seed": seed,
            "optimizer": "Adam",
            "loss_function": "CrossEntropyLoss",
            "architecture": "MLP(in->64->ReLU->Dropout(0.2)->32->ReLU->3)",
            "input_dim": input_dim,
            "output_classes": 3,
            "split_ratio": "60/20/20",
            "device": str(device),
        }
        mlflow.log_params(params)

        # Inicializa o modelo para a run
        model = build_model(input_dim=input_dim, num_classes=3, dropout_rate=DROPOUT_RATE).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        checkpoint_filename = f"best_model_{run_name}.pt"
        checkpoint_path = os.path.join(ARTIFACTS_DIR, checkpoint_filename)

        # SPAN 2: Span_Treinamento
        training_res = span_treinamento(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            epochs=epochs,
            device=device,
            checkpoint_path=checkpoint_path,
            run_name=run_name,
        )

        # SPAN 3: Span_Validacao_Final
        val_res = span_validacao_final(
            input_dim=input_dim,
            checkpoint_path=checkpoint_path,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            best_epoch=training_res["best_epoch"],
            run_name=run_name,
        )

    # Forçar descarga dos traces assíncronos da run
    try:
        mlflow.flush_trace_async_logging()
    except Exception:
        pass

    return {
        "run_name": run_name,
        "run_id": run_id,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "best_epoch": training_res["best_epoch"],
        "best_val_loss": val_res["final_best_val_loss"],
        "best_val_accuracy": val_res["final_best_val_acc"],
        "best_val_f1_weighted": val_res["final_best_val_f1_weighted"],
        "checkpoint_path": checkpoint_path,
    }


def evaluate_champion_on_test_set(
    champion_run: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    metadata: Dict[str, Any],
    class_names: List[str] = TARGET_NAMES,
) -> Dict[str, Any]:
    """
    REGRA DE OURO DA UNITINS (Slide 12):
    O modelo vencedor é avaliado estritamente UMA ÚNICA VEZ no conjunto de
    Teste Reservado (dados completamente não vistos durante o treinamento e seleção).
    """
    logger.info("\n" + "=" * 70)
    logger.info(f"AVALIAÇÃO DA RUN CAMPEÃ NO TESTE RESERVADO: {champion_run['run_name']}")
    logger.info("=" * 70)

    eval_out = span_avaliacao_teste_reservado(
        champion_run=champion_run,
        X_test=X_test,
        y_test=y_test,
        input_dim=metadata["num_features"],
        class_names=class_names,
    )

    test_summary = eval_out["test_summary"]
    cm_path = eval_out["cm_path"]
    test_summary_path = eval_out["test_summary_path"]
    report_text = eval_out["report_text"]

    # Registra métricas e artefatos de teste na Run Campeã no MLflow
    client = mlflow.tracking.MlflowClient()
    client.log_metric(champion_run["run_id"], "test_loss", test_summary["test_loss"])
    client.log_metric(champion_run["run_id"], "test_accuracy", test_summary["test_accuracy"])
    client.log_metric(champion_run["run_id"], "test_f1_weighted", test_summary["test_f1_weighted"])
    client.log_metric(champion_run["run_id"], "test_f1_macro", test_summary["test_f1_macro"])
    client.log_artifact(champion_run["run_id"], cm_path, artifact_path="test_evaluation")
    client.log_artifact(champion_run["run_id"], test_summary_path, artifact_path="test_evaluation")

    # Exibição no Terminal
    print("\n" + "=" * 70)
    print(f"DESEMPENHO DO MODELO CAMPEÃO ({champion_run['run_name']}) NO TESTE RESERVADO")
    print("=" * 70)
    print(f"Loss no Teste:           {test_summary['test_loss']:.4f}")
    print(f"Acurácia no Teste:       {test_summary['test_accuracy']:.4f} ({test_summary['test_accuracy'] * 100:.2f}%)")
    print(f"F1-Score Ponderado:      {test_summary['test_f1_weighted']:.4f}")
    print(f"F1-Score Macro:          {test_summary['test_f1_macro']:.4f}")
    print("\nRelatório de Classificação Detalhado:")
    print(report_text)
    print(f"Matriz de Confusão salva em: {cm_path}")
    print(f"Relatório JSON salvo em:     {test_summary_path}")
    print("=" * 70 + "\n")

    return test_summary


def run_pipeline():
    """
    Orquestrador Principal:
    1. Ingestão e Pré-processamento com particionamento 60/20/20.
    2. Execução sequencial das 3 Runs obrigatórias (A, B e C).
    3. Comparação estrita de validação e eleição da Campeã.
    4. Avaliação única no Teste Reservado.
    """
    logger.info("Iniciando Pipeline de MLOps - UNITINS (Prof. Marco Antonio Firmino de Sousa)...")

    # Configuração do MLflow
    tracking_uri = get_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    logger.info(f"MLflow Tracking URI configurado: {tracking_uri}")

    client = mlflow.tracking.MlflowClient()
    existing_exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if existing_exp and existing_exp.lifecycle_stage == "deleted":
        client.restore_experiment(existing_exp.experiment_id)

    mlflow.set_experiment(EXPERIMENT_NAME)
    logger.info(f"Experimento MLflow ativo: {EXPERIMENT_NAME}")

    # 1. Carregamento dos dados
    X_train, y_train, X_val, y_val, X_test, y_test, scaler, metadata = load_and_preprocess_data(
        artifacts_dir=ARTIFACTS_DIR, random_state=SEED
    )

    # 2. Definição das Três Configurações Obrigatórias (Slide 10)
    experiments_config = [
        {
            "name": "Run_A_Referencia",
            "lr": 0.01,
            "weight_decay": 0.0,
            "desc": "Taxa de aprendizado padrão (0.01) sem regularização L2",
        },
        {
            "name": "Run_B_Taxa_Menor",
            "lr": 0.001,
            "weight_decay": 0.0,
            "desc": "Taxa de aprendizado menor (0.001) para conferir estabilidade na convergência",
        },
        {
            "name": "Run_C_Com_L2",
            "lr": 0.01,
            "weight_decay": 1e-5,
            "desc": "Taxa 0.01 combinada com penalização L2 (1e-5) para mitigar sobreajuste",
        },
    ]

    runs_results: List[Dict[str, Any]] = []

    for cfg in experiments_config:
        res = train_run(
            run_name=cfg["name"],
            learning_rate=cfg["lr"],
            weight_decay=cfg["weight_decay"],
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            metadata=metadata,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            seed=SEED,
        )
        res["description"] = cfg["desc"]
        runs_results.append(res)

    # 3. Comparação Estrita no Conjunto de Validação (Slide 12 da UNITINS)
    print("\n" + "=" * 80)
    print("QUADRO COMPARATIVO DAS 3 CONFIGURAÇÕES (MÉTRICAS DE VALIDAÇÃO)")
    print("=" * 80)
    print(f"{'Run':<20} | {'LR':<8} | {'L2':<8} | {'Melhor Época':<12} | {'Val Loss':<10} | {'Val Acc':<10} | {'Val F1-Pond':<10}")
    print("-" * 80)
    for r in runs_results:
        print(
            f"{r['run_name']:<20} | {r['learning_rate']:<8} | {r['weight_decay']:<8} | "
            f"{r['best_epoch']:<12} | {r['best_val_loss']:<10.4f} | {r['best_val_accuracy']:<10.4f} | "
            f"{r['best_val_f1_weighted']:<10.4f}"
        )
    print("=" * 80)

    # Seleção da Campeã: menor val_loss (com desempate por val_accuracy)
    champion_run = min(runs_results, key=lambda x: (x["best_val_loss"], -x["best_val_accuracy"]))
    logger.info(
        f"\n>>> RUN CAMPEÃ ELEITA: {champion_run['run_name']} "
        f"(Val Loss={champion_run['best_val_loss']:.4f}, Val Acc={champion_run['best_val_accuracy']:.4f}) <<<\n"
    )

    # 4. Avaliação ÚNICA e Definitiva no Teste Reservado (Regra de Ouro)
    test_results = evaluate_champion_on_test_set(
        champion_run=champion_run,
        X_test=X_test,
        y_test=y_test,
        metadata=metadata,
        class_names=TARGET_NAMES,
    )

    try:
        mlflow.flush_trace_async_logging()
        logger.info("Traces do MLflow sincronizados com sucesso.")
    except Exception as e:
        logger.warning(f"Aviso ao sincronizar traces: {e}")

    logger.info("Pipeline concluído com sucesso!")


if __name__ == "__main__":
    run_pipeline()
