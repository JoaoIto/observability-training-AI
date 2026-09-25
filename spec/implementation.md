# Engenharia de Software, Pipeline e Observabilidade de MLOps
## Atividade Prática Avaliativa de Inteligência Artificial — UNITINS
**Curso:** Bacharelado em Sistemas de Informação / Engenharia de Software  
**Disciplina:** Inteligência Artificial  
**Docente:** Prof. Me. Marco Antonio Firmino de Sousa  
**Discente:** João Victor Ito  

---

## 1. Arquitetura do Repositório e Modularidade de Software

A estrutura do projeto foi concebida sob rigorosos princípios de **Engenharia de Software e MLOps**, evitando o padrão antiético de "scripts monolíticos de notebook" que mesclam regras de negócio, transformações de dados, definições de arquitetura e loops de treinamento em um único bloco opaco.

```
IA-MlFlow/
├── .gitignore                          # Exclusão de artefatos binários, bases de dados e caches
├── requirements.txt                    # Pinagem estrita de bibliotecas e versões
├── README.md                           # Vitrine executiva e roteiro detalhado de apresentação
├── artifacts/                          # Camada de persistência local antes do upload no MLflow
│   ├── scaler.pkl                      # Normalizador StandardScaler persistido via joblib
│   ├── resumo_pre_processamento.json   # Metadados e distribuição de partições e classes
│   ├── best_model_Run_A_Referencia.pt  # Pesos ótimos da Run A (época 7)
│   ├── best_model_Run_B_Taxa_Menor.pt  # Pesos ótimos da Run B - Campeã (época 16)
│   ├── best_model_Run_C_Com_L2.pt      # Pesos ótimos da Run C (época 7)
│   ├── best_model.pt                   # Checkpoint canônico do modelo vencedor
│   ├── matriz_confusao_teste.png       # Plot de alta resolução da Matriz de Confusão
│   └── relatorio_teste_campeao.json    # Métricas consolidadas e classification report
├── spec/                               # Especificações técnicas exaustivas
│   ├── architecture.md                 # Modelagem matemática e topologia da rede
│   ├── implementation.md               # Engenharia de software e instrumentação MLOps
│   └── results.md                      # Caderno experimental e diagnóstico de incidentes
└── src/                                # Código-fonte modular e testável
    ├── __init__.py                     # Declaração do pacote Python
    ├── dataset.py                      # Ingestão, pré-processamento, split 3 vias e DataLoaders
    ├── model.py                        # Definição pura da arquitetura em torch.nn.Module
    └── train_pipeline.py               # Orquestrador com MLflow Tracing, 3 Runs e Avaliação
```

### 1.1. Justificativa da Separação Modular
- **[`src/dataset.py`](file:///c:/Users/itofr/Documents/Testes/IA-MlFlow/src/dataset.py):** Encapsula o ciclo de vida dos dados (ETL). Isola o download automatizado via `kagglehub`, o particionamento estratificado, a padronização e o envelopamento em tensores do PyTorch (`TensorDataset` e `DataLoader`). Permite testes unitários desacoplados da GPU ou da modelagem.
- **[`src/model.py`](file:///c:/Users/itofr/Documents/Testes/IA-MlFlow/src/model.py):** Contém unicamente a definição declarativa da rede neural [`StudentRetentionMLP`](file:///c:/Users/itofr/Documents/Testes/IA-MlFlow/src/model.py#L18). Não possui dependências do pipeline de treinamento nem do MLflow, garantindo portabilidade para exportação em formatos de borda (ex: TorchScript, ONNX ou TensorRT).
- **[`src/train_pipeline.py`](file:///c:/Users/itofr/Documents/Testes/IA-MlFlow/src/train_pipeline.py):** Atua como o maestro do sistema. Gerencia o contexto de rastreamento do MLflow, orquestra os spans de tracing distribuído, executa sequencialmente as 3 configurações experimentais, calcula as métricas por época com *step*, salva os artefatos nos diretórios corretos e aplica a **Regra de Ouro** no teste reservado.
- **`artifacts/` e `mlartifacts/`:** Estabelecem uma clara separação entre a persistência local transitória e o repositório centralizado de artefatos gerenciado pelo servidor do MLflow.

---

## 2. Ingestão e Pré-processamento dos Dados

### 2.1. Ingestão Automatizada e Resiliência (Fallback Local)
A aquisição dos dados é implementada na função [`ingest_raw_data()`](file:///c:/Users/itofr/Documents/Testes/IA-MlFlow/src/dataset.py#L32), aplicando um padrão de resiliência:
1. **Verificação de Cache Local:** Primeiro inspeciona se um arquivo `dataset.csv` já está presente na raiz ou na pasta `artifacts/`.
2. **Download Programático via `kagglehub`:** Caso ausente, invoca `kagglehub.dataset_download("thedevastator/higher-education-predictors-of-student-retention")`, que autentica e gerencia o cache de downloads em `~/.cache/kagglehub/datasets/...`.
3. **Resolução de Path Automática:** Localiza dinamicamente o arquivo CSV descompactado no diretório de destino, blindando o pipeline contra alterações na estrutura de pastas da biblioteca.

### 2.2. Tratamento da Variável Alvo e Tipagem
- **Mapeamento Categórico da Variável Alvo (`Target`):**
  A coluna `Target` é convertida de representação nominal em texto para inteiros de classe compatíveis com tensores `torch.long` (int64) requeridos por `nn.CrossEntropyLoss`:
  $$\text{Target} \longrightarrow \begin{cases} \text{"Dropout"} & \mapsto 0 \\ \text{"Enrolled"} & \mapsto 1 \\ \text{"Graduate"} & \mapsto 2 \end{cases}$$
- **Tratamento de Atributos Preditivos:**
  Todos os 34 atributos preditivos são validados e convertidos para ponto flutuante de precisão simples (`np.float32`), eliminando incompatibilidades de tipos entre representações inteiras do Pandas e operadores matriciais do PyTorch.

### 2.3. Particionamento em 3 Vias Estritas (A Regra de Ouro)
A separação em três partições independentes com estratificação de classe é um dos pilares de governança exigidos no **Slide 12 da UNITINS**:

```
TOTAL DO DATASET: 4.424 Registros (100%)
│
├── [1º Split: train_test_split(test_size=0.20, stratify=y, random_state=42)]
│   │
│   ├── TESTE RESERVADO: 885 Amostras (20,0%) — [LACRADO ATÉ A ESCOLHA FINAL]
│   │
│   └── CONJUNTO DE DESENVOLVIMENTO: 3.539 Amostras (80,0%)
│       │
│       └── [2º Split: train_test_split(test_size=0.25, stratify=y_dev, random_state=42)]
│           │
│           ├── TREINO: 2.654 Amostras (60,0% do total) — [Ajuste de Pesos via SGD/Adam]
│           │
│           └── VALIDAÇÃO: 885 Amostras (20,0% do total) — [Seleção de Hiperparâmetros]
```

#### Demonstração da Proporcionalidade Estratificada Exata
A estratificação preserva com precisão matemática a proporção natural das classes em todas as partições:

| Partição | Dropout (0) | Enrolled (1) | Graduate (2) | Total de Instâncias | Proporção Relativa |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Treino** | $853$ ($32,14\%$) | $476$ ($17,94\%$) | $1.325$ ($49,92\%$) | **$2.654$** | **$60,0\%$** |
| **Validação** | $284$ ($32,09\%$) | $159$ ($17,97\%$) | $442$ ($49,94\%$) | **$885$** | **$20,0\%$** |
| **Teste Reservado** | $284$ ($32,09\%$) | $159$ ($17,97\%$) | $442$ ($49,94\%$) | **$885$** | **$20,0\%$** |
| **TOTAL GERAL** | **$1.421$** ($32,12\%$) | **$794$** ($17,95\%$) | **$2.209$** ($49,93\%$) | **$4.424$** | **$100,0\%$** |

### 2.4. Prevenção Rigorosa de Vazamento de Dados (Data Leakage)
O vazamento de dados ocorre quando informações estatísticas de validação ou teste contaminam o processo de aprendizado da rede. Para neutralizar esse risco:
1. O normalizador [`StandardScaler`](file:///c:/Users/itofr/Documents/Testes/IA-MlFlow/src/dataset.py#L129) calcula a média $\mu_{\text{train}}$ e o desvio padrão $\sigma_{\text{train}}$ **exclusivamente sobre os dados de Treino**:
   $$\mu_{\text{train}} = \frac{1}{N_{\text{train}}} \sum_{i=1}^{N_{\text{train}}} \mathbf{x}_i^{(\text{train})}, \quad \sigma_{\text{train}} = \sqrt{\frac{1}{N_{\text{train}}} \sum_{i=1}^{N_{\text{train}}} (\mathbf{x}_i^{(\text{train})} - \mu_{\text{train}})^2}$$
   $$\mathbf{X}_{\text{train}}^{\text{scaled}} = \frac{\mathbf{X}_{\text{train}} - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
2. As partições de **Validação** e **Teste** são transformadas de maneira cega utilizando unicamente $\mu_{\text{train}}$ e $\sigma_{\text{train}}$:
   $$\mathbf{X}_{\text{val}}^{\text{scaled}} = \frac{\mathbf{X}_{\text{val}} - \mu_{\text{train}}}{\sigma_{\text{train}}}, \quad \mathbf{X}_{\text{test}}^{\text{scaled}} = \frac{\mathbf{X}_{\text{test}} - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
3. O objeto ajustado `scaler` é serializado via `joblib` no arquivo [`artifacts/scaler.pkl`](file:///c:/Users/itofr/Documents/Testes/IA-MlFlow/artifacts/scaler.pkl) e posteriormente anexado como artefato da run no MLflow, garantindo que futuras inferências em produção utilizem os mesmos parâmetros de escala.

---

## 3. Arquitetura de Observabilidade com MLflow

### 3.1. Conexão de Rastreamento e Backend Store
O orquestrador [`src/train_pipeline.py`](file:///c:/Users/itofr/Documents/Testes/IA-MlFlow/src/train_pipeline.py) foi configurado para suportar ambientes locais e de servidor por meio da função [`get_tracking_uri()`](file:///c:/Users/itofr/Documents/Testes/IA-MlFlow/src/train_pipeline.py#L76):
- Se a variável de ambiente `MLFLOW_TRACKING_URI` estiver definida, ela é imediatamente respeitada.
- Caso contrário, o pipeline testa se o servidor web HTTP local do MLflow está ativo na porta padrão (`http://127.0.0.1:5000`). Estando ativo, conecta-se a ele para que as execuções apareçam em tempo real no navegador.
- Caso o servidor não esteja ativo, o sistema opera de forma autônoma sobre a base SQLite em arquivo `sqlite:///mlflow.db` e diretório de artefatos `mlartifacts/`.

O experimento unificado é registrado como:
```python
EXPERIMENT_NAME = "UNITINS_IA_Student_Retention"
```

### 3.2. MLflow Tracing: Monitoramento por Spans Explícitos
Em aderência ao Critério 1 de Avaliação (25%), o código foi instrumentado com a funcionalidade de **MLflow Tracing**, decompondo o pipeline em **4 Spans instrumentados** com a anotação `@mlflow.trace`:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        MLFLOW TRACE DISTRIBUÍDO DO PIPELINE                           │
└───────────────────────────────────┬────────────────────────────────────────────────────┘
                                    │
       ┌────────────────────────────┼────────────────────────────┐
       ▼                            ▼                            ▼
┌──────────────────────┐   ┌──────────────────────┐   ┌──────────────────────┐
│ Span_Preparar_Dados  │   │   Span_Treinamento   │   │ Span_Validacao_Final │
├──────────────────────┤   ├──────────────────────┤   ├──────────────────────┤
│ • Batch Size (32)    │   │ • 60 Épocas          │   │ • Load Checkpoint    │
│ • Shapes Tensores    │   │ • Perdas por Época   │   │ • Val Loss & Acc     │
│ • Amostras (2654/885)│   │ • Menor Loss e Ponto │   │ • Val F1 Ponderado   │
│ • Scaler Utilizado   │   │ • Backprop e Grafo   │   │ • Log de Artefatos   │
└──────────────────────┘   └──────────────────────┘   └──────────┬───────────┘
                                                                 │
                                                    (Apenas para a Run Campeã)
                                                                 ▼
                                                   ┌───────────────────────────┐
                                                   │ Span_Avaliacao_Teste_     │
                                                   │ Reservado                 │
                                                   ├───────────────────────────┤
                                                   │ • Avaliação Cega (N=885)  │
                                                   │ • Test Accuracy & Loss    │
                                                   │ • Matriz de Confusão PNG  │
                                                   │ • Classification Report   │
                                                   └───────────────────────────┘
```

#### Detalhamento dos Spans:
1. **[`Span_Preparar_Dados`](file:///c:/Users/itofr/Documents/Testes/IA-MlFlow/src/train_pipeline.py#L173):**
   - *Entradas:* Metadados do dataset, tensores brutos, semente (`SEED=42`), tamanho do lote (`batch_size=32`).
   - *Saídas:* Total de lotes de treino ($83$ lotes de tamanho $32$ por época), lotes de validação ($28$ lotes), confirmação do algoritmo `StandardScaler`.
2. **[`Span_Treinamento`](file:///c:/Users/itofr/Documents/Testes/IA-MlFlow/src/train_pipeline.py#L215):**
   - *Entradas:* Taxa de aprendizado, decaimento de peso $L_2$, número de épocas ($60$), tipo de otimizador (`Adam`).
   - *Saídas:* Época ótima de convergência (`best_epoch`), menor perda de validação (`best_val_loss`), perda final de treino (`final_train_loss`).
3. **[`Span_Validacao_Final`](file:///c:/Users/itofr/Documents/Testes/IA-MlFlow/src/train_pipeline.py#L282):**
   - *Entradas:* Nome do arquivo de checkpoint persistido em disco (`best_model_Run_X.pt`).
   - *Saídas:* Acurácia consolidada de validação (`final_best_val_acc`), F1-score ponderado e macro, relação de artefatos enviados ao MLflow.
4. **[`Span_Avaliacao_Teste_Reservado`](file:///c:/Users/itofr/Documents/Testes/IA-MlFlow/src/train_pipeline.py#L346):**
   - *Executado unicamente para a Run Campeã (Run B).*
   - *Entradas:* ID da Run vencedora, conjunto de dados de teste (885 instâncias blindadas).
   - *Saídas:* Acurácia final no teste ($77,74\%$), F1-Score ponderado ($0,7742$), caminhos do relatório JSON e do plot gráfico da matriz de confusão.

---

## 4. Os 4 Retratos de Observabilidade por Run

Em cada uma das 3 runs experimentais executadas pelo pipeline, o MLflow captura quatro dimensões essenciais de auditoria e governança:

### 4.1. Retrato de Parâmetros (`mlflow.log_params`)
Registra imutavelmente toda a configuração inicial da hipótese:
- `run_name`: Nome mnemônico da run (`Run_A_Referencia`, `Run_B_Taxa_Menor`, `Run_C_Com_L2`).
- `learning_rate`: Taxa de aprendizado inicial ($0.01$ ou $0.001$).
- `weight_decay_l2`: Coeficiente de penalização de norma $L_2$ ($0.0$ ou $1\times 10^{-5}$).
- `epochs`: Número fixo de iterações de treino ($60$).
- `batch_size`: Tamanho de lote estocástico ($32$).
- `dropout_rate`: Probabilidade de desconexão neuronal ($0.2$).
- `seed`: Semente determinística universal ($42$).
- `optimizer`: Algoritmo de otimização (`"Adam"`).
- `loss_function`: Função objetivo (`"CrossEntropyLoss"`).
- `architecture`: Especificação da topologia (`"MLP(in->64->ReLU->Dropout(0.2)->32->ReLU->3)"`).
- `split_ratio`: Proporção de partição (`"60/20/20"`).

### 4.2. Retrato Temporal: Métricas Época a Época com `step=epoch`
A cada época $k \in \{1, \dots, 60\}$, o orquestrador invoca `mlflow.log_metric()` associando explicitamente o argumento `step=epoch`:
- `train_loss`: Média da perda de entropia cruzada sobre os $83$ batches de treino da época.
- `val_loss`: Perda da rede calculada sobre o conjunto de validação estático.
- `val_accuracy`: Proporção de acertos na validação.

Ao passar `step=epoch`, o MLflow gera os gráficos bidimensionais contínuos de aprendizado no painel web, permitindo diagnosticar visualmente se a rede está sofrendo sobreajuste, estagnação ou instabilidade.

### 4.3. Retrato de Pesos: Checkpoint Mínimo em Disco
Diferente de abordagens ingênuas que salvam os pesos da última época ($k=60$), o pipeline implementa a **política de checkpoint pelo ponto ótimo de generalização**:
```python
if epoch_val_loss < best_val_loss:
    best_val_loss = epoch_val_loss
    best_val_acc = epoch_val_acc
    best_epoch = epoch
    torch.save(model.state_dict(), checkpoint_path)
```
Se a rede começar a sofrer sobreajuste a partir da época $8$, os pesos salvos para inferência permanecerão sendo aqueles obtidos no ponto mínimo da época $7$, preservando a capacidade de generalização do modelo em produção.

### 4.4. Retrato de Artefatos e Model Registry
Ao término da execução de cada run, múltiplos artefatos são persistidos:
1. `preprocessing/scaler.pkl`: Objeto `StandardScaler` serializado.
2. `preprocessing/resumo_pre_processamento.json`: Sumário das variáveis, mapeamentos e partições.
3. `checkpoints/best_model.pt` e `best_model_Run_X.pt`: Pesos dos tensores (`state_dict`).
4. `model/`: Empacotamento canônico do modelo via `mlflow.pytorch.log_model()`, incluindo o arquivo `MLmodel`, dependências `requirements.txt` e ambiente `conda.yaml` para servir o modelo via REST API com `mlflow models serve`.
5. *(Apenas na Run Campeã)* `test_evaluation/matriz_confusao_teste.png` e `relatorio_teste_campeao.json`.
