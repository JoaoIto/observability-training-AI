"""
Módulo de Arquitetura de Rede Neural (PyTorch Puro)
Atividade Prática Avaliativa de Inteligência Artificial - UNITINS
Aluno: João Victor Ito | Professor: Marco Antonio Firmino de Sousa

Modelo: Multilayer Perceptron (MLP) Supervisionado para Classificação Multiclasse.
Arquitetura Estrita:
- Entrada: input_dim (compatível com os atributos preditivos normalizados, default=34)
- Camada Oculta 1: Linear(input_dim, 64) -> ReLU() -> Dropout(0.2)
- Camada Oculta 2: Linear(64, 32) -> ReLU()
- Camada de Saída: Linear(32, 3) -> 3 Logits brutos compatíveis com nn.CrossEntropyLoss
"""

import torch
import torch.nn as nn


class StudentRetentionMLP(nn.Module):
    """
    Classificador Multiclasse baseado em Multilayer Perceptron (MLP) em PyTorch puro.
    Prevê o desfecho acadêmico do estudante:
    - 0: Dropout (Evasão)
    - 1: Enrolled (Matriculado)
    - 2: Graduate (Graduado/Formado)
    """

    def __init__(self, input_dim: int = 34, num_classes: int = 3, dropout_rate: float = 0.2):
        super(StudentRetentionMLP, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        self.network = nn.Sequential(
            # Camada Oculta 1
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            # Camada Oculta 2
            nn.Linear(64, 32),
            nn.ReLU(),
            # Camada de Saída (Logits)
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Executa a passagem para frente (forward pass).
        Retorna logits brutos (sem Softmax), adequados para nn.CrossEntropyLoss.
        """
        return self.network(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calcula as probabilidades normalizadas via Softmax.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Retorna a classe prevista (argmax dos logits).
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=1)


def build_model(input_dim: int = 34, num_classes: int = 3, dropout_rate: float = 0.2) -> StudentRetentionMLP:
    """
    Função fábrica para instanciação do modelo StudentRetentionMLP.
    """
    return StudentRetentionMLP(input_dim=input_dim, num_classes=num_classes, dropout_rate=dropout_rate)


if __name__ == "__main__":
    print("--- Teste de Arquitetura do Modelo StudentRetentionMLP ---")
    model = build_model(input_dim=34, num_classes=3, dropout_rate=0.2)
    print(model)

    dummy_input = torch.randn(4, 34)
    logits = model(dummy_input)
    probas = model.predict_proba(dummy_input)
    preds = model.predict(dummy_input)

    print("\nVerificação de Tensores:")
    print(f"Logits shape: {logits.shape} (Esperado: [4, 3])")
    print(f"Probabilidades shape: {probas.shape} (Soma por linha: {probas.sum(dim=1).numpy()})")
    print(f"Predições shape: {preds.shape}, valores: {preds.tolist()}")
