# Especificação Técnica Unificada e Guia de Auditoria — MLOps UNITINS
## Predição de Evasão Universitária com PyTorch e Rastreamento Experimental no MLflow

```
========================================================================================================
UNIVERSIDADE ESTADUAL DO TOCANTINS (UNITINS)
Câmpus Universitário de Palmas | Colegiado de Sistemas de Informação
Disciplina: Inteligência Artificial (Período Letivo 2026/1)
Docente: Prof. Me. Marco Antonio Firmino de Sousa
Discente: João Victor Ito
Documento: Especificação Central Unificada e Guia de Auditoria do Projeto (spec/README.md)
========================================================================================================
```

---

## 1. Visão Geral e Contexto Institucional

### 1.1. O Problema de Negócio em Sistemas de Informação Acadêmicos
A evasão no ensino superior representa um dos mais severos desafios institucionais e financeiros enfrentados por universidades no Brasil e no mundo. Nos Sistemas Integrados de Gestão Universitária (ERPs Acadêmicos), o monitoramento discente tradicional opera de maneira passiva (*ex-post facto*), registrando a perda do aluno somente após o abandono formal ou trancamento de matrícula.

Neste projeto acadêmico-corporativo desenvolvido para a **UNITINS**, construiu-se uma solução baseada em **Engenharia de Machine Learning e MLOps**, concebida para atuar como o núcleo preditivo de um **Sistema de Alerta Precoce (*Early Warning System*)**. O objetivo é prever a trajetória do estudante em três desfechos categóricos mutuamente exclusivos:
- **Classe 0 — Dropout (Evasão):** Discentes em risco iminente de desistência ou cancelamento de matrícula, público-alvo para ações de retenção institucional (apoio pedagógico, renegociação de débitos e bolsas).
- **Classe 1 — Enrolled (Matriculado / Retido):** Discentes em curso regular ou enfrentando retenções parciais em disciplinas.
- **Classe 2 — Graduate (Graduado / Concluinte):** Discentes com trajetória curricular consistente e alta probabilidade de conclusão do curso.

### 1.2. O Paradigma MLOps: Da Fragilidade Manual à Engenharia Auditável
Historicamente, equipes de desenvolvimento realizam experimentos com modelos de Inteligência Artificial de forma artesanal: executando scripts ad-hoc no terminal, inspecionando saídas efêmeras via `print()`, sem versionamento de hiperparâmetros, sem garantia de sementes determinísticas e sem salvamento atômico de artefatos.

A abordagem de **MLOps** adotada neste repositório substitui essa fragilidade por uma infraestrutura orientada a código (*Infrastructure as Code* e *Data-as-Code*):
1. **Automação e Reprodutibilidade:** Ingestão orientada por API (`kagglehub`), particionamento estratificado rigoroso sob semente fixa universal (`SEED = 42`) e normalização livre de vazamento estatístico (*Data Leakage*).
2. **Observabilidade Granular:** Registro em tempo real de parâmetros, métricas calculadas época a época associadas a *step* temporal, checkpoints automáticos condicionados à menor perda de validação e rastreamento distribuído via **MLflow Tracing**.
3. **Governança Estrita (A Regra de Ouro da UNITINS):** Avaliação cega e isolamento absoluto da partição de teste até a definição conclusiva do modelo campeão.

---

## 2. O Dataset e o Pipeline de Engenharia de Dados

### 2.1. Metadados da Base Oficial
O conjunto de dados adotado é o consagrado *"Predict Students' Dropout and Academic Success"* (The Devastator / UCI Machine Learning Repository), ingerido de forma automatizada pelo [`src/dataset.py`](file:///c:/Users/itofr/Documents/Testes/IA-MlFlow/src/dataset.py) através da biblioteca `kagglehub`:
- **Total de Amostras Brutas:** **4.424 instâncias discentes**.
- **Espaço Dimensional de Entrada:** **34 atributos preditivos** após higienização e validação tabular.
- **Dimensionalidade da Variável Alvo (`Target`):** 3 classes nominais mapeadas para valores ordinais inteiros (`int64`):
  $$\text{Target} \longrightarrow \begin{cases} \text{"Dropout"} & \mapsto 0 \quad (1.421 \text{ instâncias, } 32,12\%) \\ \text{"Enrolled"} & \mapsto 1 \quad (794 \text{ instâncias, } 17,95\%) \\ \text{"Graduate"} & \mapsto 2 \quad (2.209 \text{ instâncias, } 49,93\%) \end{cases}$$

### 2.2. A Regra de Ouro da Partição (Slide 12 da UNITINS)
Para assegurar comparabilidade científica estrita e neutralizar o viés de seleção (*data snooping*), o dataset foi dividido em três vias independentes com estratificação de classe (`stratify = y`) e semente determinística (`random_state = 42`):

```
========================================================================================================
DIVISÃO ESTRATIFICADA EM 3 VIAS — MESMO SPLIT PARA TODAS AS RUNS (SEED = 42)
========================================================================================================
Partição         | Proporção | Total Amostras | Dropout (0)    | Enrolled (1)   | Graduate (2)
--------------------------------------------------------------------------------------------------------
Treino (Train)   | 60,0%     | 2.654          | 853 (32,14%)   | 476 (17,94%)   | 1.325 (49,92%)
Validação (Val)  | 20,0%     | 885            | 284 (32,09%)   | 159 (17,97%)   | 442 (49,94%)
Teste Reservado  | 20,0%     | 885            | 284 (32,09%)   | 159 (17,97%)   | 442 (49,94%)
--------------------------------------------------------------------------------------------------------
TOTAL CONSOLIDADO| 100,0%    | 4.424          | 1.421 (32,12%) | 794 (17,95%)   | 2.209 (49,93%)
========================================================================================================
```

- **Treino (60%):** Utilizado unicamente pelo otimizador estocástico para atualização matricial dos pesos e bias via retropropagação (*backpropagation*).
- **Validação (20%):** Utilizado como conjunto de sintonia fina (*tuning*) para monitorar curvas de aprendizado, prevenir sobreajuste (*early stopping/checkpointing*) e eleger a Run Campeã.
- **Teste Reservado (20%):** Mantido lacrado e blindado de qualquer intervenção, sendo avaliado **estritamente uma única vez** pela Run vencedora.

### 2.3. Prevenção Rigorosa de Vazamento de Dados (Data Leakage)
O normalizador estatístico [`StandardScaler`](file:///c:/Users/itofr/Documents/Testes/IA-MlFlow/src/dataset.py#L129) transforma os atributos para média nula ($\mu=0$) e variância unitária ($\sigma^2=1$). Para blindar a avaliação contra contaminação informacional:
1. O cálculo de $\mu_{\text{train}}$ e $\sigma_{\text{train}}$ é executado **exclusivamente no conjunto de Treino**:
   $$\mathbf{X}_{\text{train}}^{\text{scaled}} = \text{scaler.fit\_transform}(\mathbf{X}_{\text{train}})$$
2. As partições de Validação e Teste são transformadas de maneira cega utilizando os parâmetros congelados do treino:
   $$\mathbf{X}_{\text{val}}^{\text{scaled}} = \text{scaler.transform}(\mathbf{X}_{\text{val}}), \quad \mathbf{X}_{\text{test}}^{\text{scaled}} = \text{scaler.transform}(\mathbf{X}_{\text{test}})$$
3. O scaler ajustado é serializado via `joblib` no artefato [`artifacts/scaler.pkl`](file:///c:/Users/itofr/Documents/Testes/IA-MlFlow/artifacts/scaler.pkl) e anexado ao experimento no MLflow.

### 2.4. Diagrama ASCII do Fluxo de Dados Ponta a Ponta
```
┌────────────────────────────────────────────────────────┐
│  Dataset Oficial (KaggleHub: thedevastator/...retention)│
└───────────────────────────┬────────────────────────────┘
                            │ Download Automatizado / Cache Local
                            ▼
┌────────────────────────────────────────────────────────┐
│  DataFrame Pandas (4.424 Linhas x 35 Colunas Brutas)   │
└───────────────────────────┬────────────────────────────┘
                            │ 1. Mapeamento do Target {'Dropout':0, 'Enrolled':1, 'Graduate':2}
                            │ 2. Conversão dos 34 Atributos Preditivos para float32
                            ▼
┌────────────────────────────────────────────────────────┐
│  Split 1: 80% Desenvolvimento (3.539) vs 20% Teste (885)│
└───────────────────────────┬────────────────────────────┘
                            │ Split 2: 75% Treino (2.654) vs 25% Validação (885)
                            ▼
┌───────────────────────────────────────────────────────────────────────────┐
│     StandardScaler: fit_transform(X_train) ──> artifacts/scaler.pkl       │
│                     transform(X_val)                                      │
│                     transform(X_test)                                     │
└───────────────────────────┬───────────────────────────────────────────────┘
                            │ Empacotamento em Tensores PyTorch (batch_size=32)
                            ▼
┌──────────────────────┐ ┌──────────────────────┐ ┌──────────────────────┐
│  train_loader (83 b) │ │   val_loader (28 b)  │ │   test_loader (28 b) │
│  2.654 Amostras (60%)│ │   885 Amostras (20%) │ │   885 Amostras (20%) │
└──────────────────────┘ └──────────────────────┘ └──────────────────────┘
```

---

## 3. Arquitetura Neural e Decomposição Matemática dos Parâmetros

### 3.1. Topologia da Rede Neural (StudentRetentionMLP)
A arquitetura conexionista supervisionada foi desenvolvida estritamente em **PyTorch puro** ([`src/model.py`](file:///c:/Users/itofr/Documents/Testes/IA-MlFlow/src/model.py)), herdando de `torch.nn.Module`:

```
========================================================================================================
ESQUEMA TOPOLÓGICO DA REDE NEURAL MULTILAYER PERCEPTRON (PyTorch Puro)
========================================================================================================

    [Camada de Entrada]       X ∈ ℝ³²ˣ³⁴ (Mini-lote de 32 amostras x 34 features)
                                  │
                                  ▼
    [Camada Oculta 1]         Linear(34, 64) ──> W¹ ∈ ℝ⁶⁴ˣ³⁴, b¹ ∈ ℝ⁶⁴  ==> 2.240 parâmetros
                                  │
                                  ▼
    [Ativação Não Linear]     ReLU()         ──> f(u) = max(0, u)
                                  │
                                  ▼
    [Regularização Estocástica]Dropout(p=0.2) ──> Máscara Bernoulli Invertida (Treino)
                                  │
                                  ▼
    [Camada Oculta 2]         Linear(64, 32) ──> W² ∈ ℝ³²ˣ⁶⁴, b² ∈ ℝ³²  ==> 2.080 parâmetros
                                  │
                                  ▼
    [Ativação Não Linear]     ReLU()         ──> f(u) = max(0, u)
                                  │
                                  ▼
    [Camada de Saída]         Linear(32, 3)  ──> W³ ∈ ℝ³ˣ³², b³ ∈ ℝ³    ==>    99 parâmetros
                                  │
                                  ▼
    [Vetor de Logits]         z ∈ ℝ³²ˣ³      ──> Emite 3 logits brutos sem ativação final
                                  │
                                  ▼
    [Função de Custo]         nn.CrossEntropyLoss() (Calcula Softmax + NLLLoss internamente)
```

### 3.2. Dedução Analítica dos 4.419 Parâmetros Treináveis
Para qualquer camada densa conexa com dimensão de entrada $N_{\text{in}}$ e dimensionalidade de saída $N_{\text{out}}$, o número exato de parâmetros livres corresponde à soma dos pesos sinápticos com os vieses:

$$\Theta_{\text{camada}} = (N_{\text{in}} \times N_{\text{out}}) + N_{\text{out}} = (N_{\text{in}} + 1) \times N_{\text{out}}$$

Aplicando a dedução analítica às três camadas lineares de [`StudentRetentionMLP`](file:///c:/Users/itofr/Documents/Testes/IA-MlFlow/src/model.py#L18):
1. **Camada Linear 1 (`network.0`):**
   $$\Theta_1 = (34 \times 64) + 64 = 2.176 + 64 = \mathbf{2.240 \text{ parâmetros}}$$
2. **Camada Linear 2 (`network.3`):**
   $$\Theta_2 = (64 \times 32) + 32 = 2.048 + 32 = \mathbf{2.080 \text{ parâmetros}}$$
3. **Camada Linear 3 (`network.5`):**
   $$\Theta_3 = (32 \times 3) + 3 = 96 + 3 = \mathbf{99 \text{ parâmetros}}$$

$$\Theta_{\text{total}} = 2.240 + 2.080 + 99 = \mathbf{4.419 \text{ parâmetros treináveis}}$$

*Nota: As camadas de ativação ReLU e Dropout são operadores livres de parâmetros de peso ($\Theta = 0$).*

### 3.3. Fundamentação Matemática das Operações Conexionistas
- **Imunidade à Saturação com Ativação ReLU ($f(u) = \max(0, u)$):**
  Diferente da função Sigmoide ($\sigma'(u) \le 0.25$) ou Tangente Hiperbólica ($\tanh'(u) \le 1.0$), cuja derivada decai exponencialmente para zero em magnitudes elevadas de ativação gerando o colapso dos gradientes (*Vanishing Gradient*), a ReLU possui derivada local estritamente constante $f'(u) = 1$ para todo regime positivo ($u > 0$). Isso permite retropropagar o sinal de erro através das camadas com estabilidade numérica.
- **Regularização com Dropout Invertido ($p=0.2$):**
  Durante a passagem direta em modo de treino, cada ativação $a_j$ da camada oculta 1 é multiplicada por uma variável aleatória de Bernoulli $r_j \sim \text{Bernoulli}(0.8)$ e reescalada por $\frac{1}{1-p} = \frac{1}{0.8} = 1.25$:
  $$\widetilde{a}_j = \frac{r_j}{1 - p} a_j$$
  Esse mecanismo impede a co-adaptação espúria entre neurônios vizinhos e atua matematicamente como o treinamento implícito de um conjunto exponencial de $2^{64}$ sub-redes esparsas com pesos compartilhados (Srivastava et al., 2014).
- **Entropia Cruzada Multiclasse com LogSumExp:**
  A perda empírica por amostra combina o Softmax normalizador com a Log-Verossimilhança Negativa (NLLLoss):
  $$\mathcal{L}_{\text{CE}}(\mathbf{z}, y) = - \log\left(\frac{e^{z_y}}{\sum_{k=0}^{2} e^{z_k}}\right) = -z_y + \log\left(\sum_{k=0}^{2} e^{z_k}\right)$$
  O PyTorch implementa internamente o truque aritmético do **LogSumExp (LSE)**:
  $$\log\left(\sum_{k=0}^{2} e^{z_k}\right) = M + \log\left(\sum_{k=0}^{2} e^{z_k - M}\right), \quad M = \max_k(z_k)$$
  Isso impede o estouro de precisão (*overflow*) de expoentes em ponto flutuante FP32, garantindo que o maior argumento do exponencial seja $e^0 = 1$.

### 3.4. Conexão com a Literatura Clássica (Russell & Norvig, Cap. 18 e 21)
- **Teorema da Aproximação Universal (Cybenko, 1989; Hornik, 1991):** Redes neurais de camada única linear são incapazes de modelar fronteiras não lineares (como a separação entre matriculados e evasores). Camadas ocultas com funções de ativação não lineares contínuas formam uma base funcional densa, capaz de aproximar com precisão arbitrária qualquer superfície de decisão contínua.
- **O Princípio da Parcimônia (Navalha de Ockham):** O dimensionamento contido de 4.419 parâmetros para um universo de 2.654 amostras de treino ($\approx 1,66$ parâmetros por amostra) restringe a capacidade da hipótese ($\mathcal{H}$), equilibrando o compromisso entre viés e variância (*Bias-Variance Tradeoff*) e prevenindo a memorização trivial de ruídos tabulares.

---

## 4. Arquitetura de MLOps e Observabilidade com MLflow

### 4.1. Persistência de Metadados e Backend Store
O orquestrador [`src/train_pipeline.py`](file:///c:/Users/itofr/Documents/Testes/IA-MlFlow/src/train_pipeline.py) foi implementado com mecanismo de detecção dinâmica do ambiente:
1. Se a variável de ambiente `MLFLOW_TRACKING_URI` estiver presente, prioriza-a.
2. Caso contrário, verifica se o servidor HTTP local do MLflow está ativo em `http://127.0.0.1:5000` e conecta-se a ele para atualização ao vivo da interface gráfica.
3. Como mecanismo resiliente de base, conecta-se diretamente ao banco de dados relacional [`mlflow.db`](file:///c:/Users/itofr/Documents/Testes/IA-MlFlow/mlflow.db) via `sqlite:///mlflow.db` e armazena os binários de modelos e gráficos na pasta centralizada `mlartifacts/`.

O experimento unificado é registrado como:
```python
EXPERIMENT_NAME = "UNITINS_IA_Student_Retention"
```

### 4.2. Tracing Distribuído e Instrumentação por Spans Explícitos
Em estrito cumprimento do **Critério 1 de Avaliação (25%)**, o ciclo de vida do pipeline foi instrumentado através da funcionalidade moderna de **MLflow Tracing**, decompondo a execução em **4 Spans instrumentados** com a anotação `@mlflow.trace`:

| Span de Rastreamento | Função Python | Metadados de Entrada Rastreados | Metadados de Saída Rastreados | Papel na Observabilidade |
| :--- | :--- | :--- | :--- | :--- |
| **`Span_Preparar_Dados`** | [`span_preparar_dados`](file:///c:/Users/itofr/Documents/Testes/IA-MlFlow/src/train_pipeline.py#L173) | Nome do dataset, amostras brutas, semente (42), batch size (32). | Contagem de batches ($83$ treino, $28$ val), shapes de tensores, confirmação do scaler. | Monitora tempo de I/O, partição em memória e ausência de overhead na carga de dados. |
| **`Span_Treinamento`** | [`span_treinamento`](file:///c:/Users/itofr/Documents/Testes/IA-MlFlow/src/train_pipeline.py#L215) | Taxa de aprendizado, penalização $L_2$, total de épocas ($60$), otimizador (`Adam`). | Época do ponto ótimo (`best_epoch`), menor perda de validação, perda final de treino. | Rastreia a latência das 60 épocas de forward/backward pass e convergência. |
| **`Span_Validacao_Final`** | [`span_validacao_final`](file:///c:/Users/itofr/Documents/Testes/IA-MlFlow/src/train_pipeline.py#L282) | Caminho do melhor checkpoint de pesos (`best_model_Run_X.pt`). | Perda final de validação, acurácia consolidada, F1 ponderado, lista de artefatos. | Valida a integridade do modelo ótimo carregado e serialização de checkpoints. |
| **`Span_Avaliacao_Teste_Reservado`** | [`span_avaliacao_teste_reservado`](file:///c:/Users/itofr/Documents/Testes/IA-MlFlow/src/train_pipeline.py#L346) | ID da Run Campeã, partição de teste reservado ($N=885$). | Acurácia de teste ($77,74\%$), F1 ponderado ($0.7742$), matriz de confusão salva. | **Executado estritamente para a Campeã**, auditando a avaliação em dados não vistos. |

### 4.3. Os Quatro Retratos de Observabilidade por Run
Para cada execução experimental, o MLflow captura quatro retratos imutáveis de governança:
1. **Retrato de Parâmetros (`mlflow.log_params`):** Congelamento estático de `learning_rate`, `weight_decay_l2`, `epochs=60`, `batch_size=32`, `seed=42`, `optimizer="Adam"`, `loss_function="CrossEntropyLoss"` e `split_ratio="60/20/20"`.
2. **Retrato Temporal (`mlflow.log_metric` com `step=epoch`):** Rastreamento dinâmico época a época ($1 \le \text{step} \le 60$) das métricas contínuas `train_loss`, `val_loss` e `val_accuracy`, viabilizando a sobreposição gráfica das curvas de aprendizado na UI.
3. **Retrato de Pesos (Checkpoint por Menor Val Loss):** Política ativa de salvamento de checkpoint atômico. Se a perda de validação atingir seu mínimo na época $k$, os tensores de pesos são congelados em disco ([`artifacts/best_model_Run_X.pt`](file:///c:/Users/itofr/Documents/Testes/IA-MlFlow/artifacts)). Se a rede sofrer sobreajuste nas épocas posteriores, o modelo preservado para inferência permanece sendo o do ponto ótimo de validação.
4. **Retrato de Artefatos (`mlflow.log_artifact` e `mlflow.pytorch.log_model`):** Empacotamento de modelos em formato padronizado MLmodel com especificações de ambiente (`conda.yaml`, `requirements.txt`), além dos arquivos auxiliares [`scaler.pkl`](file:///c:/Users/itofr/Documents/Testes/IA-MlFlow/artifacts/scaler.pkl), [`resumo_pre_processamento.json`](file:///c:/Users/itofr/Documents/Testes/IA-MlFlow/artifacts/resumo_pre_processamento.json) e [`matriz_confusao_teste.png`](file:///c:/Users/itofr/Documents/Testes/IA-MlFlow/artifacts/matriz_confusao_teste.png).

---

## 5. Caderno Experimental e Comparação das 3 Runs (Slide 10)

### 5.1. O Princípio Científico de "Não Trocar o Paciente"
Na experimentação científica de Machine Learning, a comparação só possui validade quando os fatores externos são rigorosamente controlados. O princípio metodológico de "não trocar o paciente" assegura que:
- O conjunto de dados foi fatiado sob o mesmo gerador pseudoaleatório (`SEED = 42`).
- As amostras contidas em Treino, Validação e Teste foram **idênticas** nas Runs A, B e C.
- A inicialização dos pesos da rede e a sequência dos mini-lotes nos DataLoaders foram idênticas.
Dessa forma, qualquer divergência de desempenho decorre **exclusivamente da interação entre taxa de aprendizado e decaimento de peso**.

### 5.2. Tabela Comparativa de Métricas Reais Extraídas do MLflow

```
=================================================================================================================
QUADRO COMPARATIVO DAS 3 CONFIGURAÇÕES CONTROLADAS (MÉTRICAS OFICIAIS EXTRAÍDAS DO MLFLOW)
=================================================================================================================
Identificador        | Taxa (LR) | L2 (Decay) | Ponto Ótimo  | Menor Val Loss | Val Acc Máxima | Gap Gen. (ΔL)
-----------------------------------------------------------------------------------------------------------------
Run_A_Referencia     | 0.01      | 0.0        | Época 7      | 0.5875         | 76,61%         | 0.5769
Run_B_Taxa_Menor 🏆  | 0.001     | 0.0        | Época 16     | 0.5649         | 76,84%         | 0.2652 (54% menor)
Run_C_Com_L2         | 0.01      | 1e-05      | Época 7      | 0.5683         | 76,38%         | 0.6408
=================================================================================================================
```

*Detalhamento do Gap de Generalização final:* $\Delta \mathcal{L} = \mathcal{L}_{\text{val}}^{(\text{época } 60)} - \mathcal{L}_{\text{train}}^{(\text{época } 60)}$.

### 5.3. Diagnóstico do Incidente Científico (Slide 13 da UNITINS)
1. **O Sintoma na Run A (Referência — $\eta=0.01, L_2=0.0$):**
   A perda de treino decresce velozmente (de $0.6835$ na época 1 para $0.3082$ na época 60), porém a perda de validação atinge seu ponto ótimo muito cedo (época 7 com $0.5875$) e passa a sofrer uma elevação contínua e volátil, encerrando em **$0.8851$** (+50,6% de degradação). Trata-se da formação nítida da **curva clássica em "U"** de sobreajuste (*overfitting*). A acurácia de validação decai de $76,61\%$ para $72,09\%$.
2. **A Hipótese Formulada:**
   Com o otimizador Adam operando sobre variáveis previamente padronizadas pelo `StandardScaler`, a taxa de aprendizado de $0.01$ (dez vezes superior ao padrão recomendado na literatura) produz passos de atualização excessivamente amplos. Ao se aproximar de vales estreitos na superfície de erro, o vetor de gradiente salta entre as paredes da ravina em vez de descer ao ponto estacionário, forçando os neurônios a memorizarem correlações espúrias do lote de treino.
3. **O Ajuste e Validação na Run B (Taxa Menor — $\eta=0.001, L_2=0.0$):**
   Ao reduzir a taxa de aprendizado em uma ordem de magnitude, a velocidade de atualização foi harmonizada com a curvatura local da função de custo. A perda de validação descendeu de maneira monótona e estável até a **época 16**, conquistando a **menor perda global ($0.5649$)** e a **maior acurácia de validação ($76,84\%$)**, com gap de generalização de apenas $0.2652$.
4. **O Efeito Amortecedor na Run C (Com L2 — $\eta=0.01, L_2=1\times 10^{-5}$):**
   A penalização $L_2$ exerceu tração analítica sobre a magnitude dos pesos, fazendo com que a perda mínima na época 7 fosse de $0.5683$ (superior à Run A). Contudo, a taxa $\eta=0.01$ prevaleceu sobre a contração de norma nas épocas finais, demonstrando que **a regularização $L_2$ mitiga a amplitude das oscilações, mas não substitui a calibração correta da taxa de aprendizado**.

---

## 6. Avaliação Cega no Teste Reservado (A "Regra de Ouro" - Slide 12)

### 6.1. Justificativa Formal da Eleição da Run Campeã
Em estrita conformidade com o **Slide 12 da UNITINS**, a partição de Teste Reservado permaneceu **100% lacrada e intocada** durante todo o processo experimental. A escolha da Run Campeã foi pautada **exclusivamente nas métricas do conjunto de Validação**:
- Menor Perda de Validação: **Run B ($0.5649$)** vs. Run C ($0.5683$) vs. Run A ($0.5875$).
- Maior Acurácia de Validação: **Run B ($76,84\%$)** vs. Run A ($76,61\%$) vs. Run C ($76,38\%$).
- Maior F1-Score Ponderado: **Run B ($0.7633$)** vs. Run A ($0.7570$) vs. Run C ($0.7520$).

A **`Run_B_Taxa_Menor`** foi consagrada a Campeã e seus pesos ótimos salvos na época 16 ([`artifacts/best_model_Run_B_Taxa_Menor.pt`](file:///c:/Users/itofr/Documents/Testes/IA-MlFlow/artifacts/best_model_Run_B_Taxa_Menor.pt)) foram submetidos à avaliação no teste.

### 6.2. Resultados Consolidados no Teste Reservado (Dados Não Vistos — $N=885$)
Os dados do relatório oficial [`artifacts/relatorio_teste_campeao.json`](file:///c:/Users/itofr/Documents/Testes/IA-MlFlow/artifacts/relatorio_teste_campeao.json) atestam a robustez de generalização do modelo em produção:

```
========================================================================================================
DESEMPENHO DO MODELO CAMPEÃO NO TESTE RESERVADO (AVALIAÇÃO CEGA ÚNICA)
========================================================================================================
Métrica de Avaliação             | Valor Obtido      | Percentual / Interpretação
--------------------------------------------------------------------------------------------------------
Acurácia Global (Accuracy)       | 0.7774            | 77,74% (688 acertos em 885 instâncias)
Perda Médio no Teste (Loss)      | 0.5820            | Entropia cruzada perfeitamente calibrada
F1-Score Ponderado (Weighted)    | 0.7742            | Equilíbrio harmônico ponderado pelo suporte
F1-Score Macro (Não Ponderado)   | 0.7251            | Alta discriminação média entre as 3 classes
Amostras Inéditas Avaliadas      | 885 discentes     | Zero contaminação com treino e validação
========================================================================================================
```

### 6.3. Matriz de Confusão Analítica e Relatório de Classificação

```
========================================================================================================
MATRIZ DE CONFUSÃO ANALÍTICA (TESTE RESERVADO — N = 885)
========================================================================================================
                       CLASSE PREVISTA PELO MODELO
                       Dropout (0)      Enrolled (1)     Graduate (2)     Total Real Discentes
--------------------------------------------------------------------------------------------------------
CLASSE   Dropout (0)   |     214        |      31        |      39        |    284 discentes
REAL     Enrolled (1)  |      30        |      82        |      47        |    159 discentes
(TRUE)   Graduate (2)  |      17        |      33        |     392        |    442 discentes
--------------------------------------------------------------------------------------------------------
TOTAL PREVISTO         |     261        |     146        |     478        |    885 discentes
========================================================================================================
```

#### Relatório Detalhado por Classe:
```
              precision    recall  f1-score   support

     Dropout     0.8199    0.7535    0.7853       284
    Enrolled     0.5616    0.5157    0.5377       159
    Graduate     0.8201    0.8869    0.8522       442

    accuracy                         0.7774       885
   macro avg     0.7339    0.7187    0.7251       885
weighted avg     0.7736    0.7774    0.7742       885
```

### 6.4. Valor Prático e Aplicação em Sistemas de Informação Acadêmicos
- **Detecção Confiável de Evasão (Dropout):** A precisão de **$81,99\%$** significa que quando o modelo emite um sinal de alerta de evasão, em mais de 8 de cada 10 casos o estudante realmente abandonaria a instituição se nada fosse feito. Para a pró-reitoria comunitária e coordenadores de curso da UNITINS, isso permite alocar assistentes sociais, tutores e bolsas com alta assertividade, sem desperdiçar recursos financeiros em falsos alarmes. Apenas 17 concluintes verdadeiros foram confundidos com evasores ($\approx 3,8\%$).
- **Identificação Precisa de Formandos (Graduate):** A sensibilidade de **$88,69\%$** assegura que a instituição reconhece quase a totalidade dos concluintes regulares, validando o impacto do desempenho curricular nos 1º e 2º semestres.
- **A Complexidade dos Matriculados (Enrolled):** Apresentou precisão de $56,16\%$ e recall de $51,57\%$. No contexto acadêmico, o status "matriculado" representa uma condição transitória e heterogênea — alunos que trancaram matérias parciais ou acumulam dependências, assemelhando-se simultaneamente a perfis de evasão futura ou formatura tardia.

---

## 7. Índice Cruzado e Guia de Navegação Técnica

Para auditoria detalhada de cada dimensão do repositório, consulte os documentos de apoio:

```
IA-MlFlow/
├── spec/
│   ├── README.md               <-- ESTE DOCUMENTO (Especificação Mestra Unificada e Guia de Auditoria)
│   ├── architecture.md         <-- Modelagem Matemática Formal, Topologia e Russell & Norvig
│   ├── implementation.md       <-- Engenharia de Software, 3 Vias, Anti-Leakage e Spans de MLOps
│   └── results.md              <-- Caderno Experimental, Curvas em "U", Diagnóstico e Métricas
├── README.md                   <-- Vitrine Executiva do GitHub e Roteiro de Vídeo (YouTube)
└── artifacts/                  <-- Artefatos Físicos Persistidos (Pesos .pt, Scaler .pkl e Plots .png)
```

| Arquivo de Especificação | Foco Técnico Principal |
| :--- | :--- |
| [**`spec/architecture.md`**](file:///c:/Users/itofr/Documents/Testes/IA-MlFlow/spec/architecture.md) | Dedução matemática analítica dos 4.419 parâmetros treináveis, ReLU contra saturação, Inverted Dropout ($p=0.2$) com máscara Bernoulli e fundamentação teórica segundo Russell & Norvig (Teorema da Aproximação Universal e Navalha de Ockham). |
| [**`spec/implementation.md`**](file:///c:/Users/itofr/Documents/Testes/IA-MlFlow/spec/implementation.md) | Engenharia de software modular, isolamento de dados com `StandardScaler` exclusivo no treino, particionamento 60/20/20, os 4 Spans do MLflow Tracing e os 4 retratos de observabilidade por run. |
| [**`spec/results.md`**](file:///c:/Users/itofr/Documents/Testes/IA-MlFlow/spec/results.md) | Diário de bordo experimental completo, tabelas de perda época a época, gráficos conceituais das curvas no MLflow, investigação formal do incidente na Run A e interpretação da matriz de confusão. |
| [**`README.md`**](file:///c:/Users/itofr/Documents/Testes/IA-MlFlow/README.md) | Visão executiva da atividade prática da UNITINS e **Roteiro Estruturado de Gravação em Vídeo (00:00 a 10:00)** para apresentação no YouTube com câmera aberta e navegação pela UI do MLflow. |

---

## 8. Guia Rápido de Reprodução do Pipeline

### 8.1. Execução do Pipeline em Comando Único
Estando no ambiente virtual ativado (`.venv`), execute:
```powershell
python src/train_pipeline.py
```
O script realizará automaticamente:
1. Ingestão da base via `kagglehub` e validação das 34 variáveis.
2. Particionamento determinístico 60/20/20 e geração dos artefatos em `artifacts/`.
3. Execução sequencial das Runs A, B e C com MLflow Tracing.
4. Comparação automática na validação e seleção da Run Campeã (Run B).
5. Avaliação cega única no Teste Reservado com geração da matriz de confusão gráfica.

### 8.2. Acesso ao Painel Visual do MLflow
Inicie o servidor de interface gráfica:
```powershell
mlflow ui
```
Acesse no navegador: [**http://127.0.0.1:5000**](http://127.0.0.1:5000).

No painel:
- **Experiments:** Selecione `UNITINS_IA_Student_Retention`.
- **Compare:** Selecione as Runs A, B e C para visualizar a curva clássica em "U" da Run A e a estabilidade da Run B.
- **Traces:** Inspecione os 4 Spans cronometrados (`Span_Preparar_Dados`, `Span_Treinamento`, `Span_Validacao_Final`, `Span_Avaliacao_Teste_Reservado`).
- **Artifacts:** Abra a pasta `test_evaluation` na Run B para visualizar a matriz de confusão `matriz_confusao_teste.png` e os pesos empacotados.
