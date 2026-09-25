# Caderno Experimental, Análise Comparativa e Diagnóstico de Incidentes
## Atividade Prática Avaliativa de Inteligência Artificial — UNITINS
**Curso:** Bacharelado em Sistemas de Informação / Engenharia de Software  
**Disciplina:** Inteligência Artificial  
**Docente:** Prof. Me. Marco Antonio Firmino de Sousa  
**Discente:** João Victor Ito  

---

## 1. Matriz Experimental Oficial (Slide 10 da UNITINS)

Para atender estritamente aos requisitos avaliativos estabelecidos pelo Prof. Marco Antonio Firmino de Sousa, foram projetadas três configurações experimentais controladas. A metodologia científica adotada segue o princípio de **isolamento de variáveis**: a arquitetura da rede ([`StudentRetentionMLP`](file:///c:/Users/itofr/Documents/Testes/IA-MlFlow/src/model.py#L18)), o particionamento dos dados ($60\%$ treino, $20\%$ validação e $20\%$ teste), a semente determinística universal (`SEED = 42`), o número de épocas ($60$) e o tamanho de lote ($32$) mantiveram-se absolutamente **idênticos** entre as três runs.

A única variação sistemática introduzida residiu no par de hiperparâmetros de otimização: **taxa de aprendizado ($\eta$ / `learning_rate`)** e **coeficiente de penalização de decaimento de peso ($L_2$ / `weight_decay`)**:

| Identificador da Run | Taxa de Aprendizado ($\eta$) | Penalização $L_2$ (`weight_decay`) | Semente (`SEED`) | Épocas | Batch Size | Finalidade Científica da Configuração |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Run A (Referência)** | $0.01$ | $0.0$ | $42$ | $60$ | $32$ | Estabelece a linha de base com a taxa padrão de mercado sem regularização analítica. |
| **Run B (Taxa Menor)** | $0.001$ | $0.0$ | $42$ | $60$ | $32$ | Avalia o impacto da redução de uma ordem de magnitude na suavidade da descida do gradiente. |
| **Run C (Com L2)** | $0.01$ | $1\times 10^{-5}$ | $42$ | $60$ | $32$ | Testa a capacidade da contração de norma dos pesos em amortecer a taxa de aprendizado agressiva ($0.01$). |

---

## 2. Análise Comparativa dos Resultados (Dados Extraídos dos Logs do MLflow)

Os dados a seguir foram extraídos diretamente do banco de dados `mlflow.db` e do servidor de rastreamento do experimento `UNITINS_IA_Student_Retention` (Experimento ID: 3):

### 2.1. Tabela Consolidada de Métricas de Treino e Validação

| Métrica / Atributo | Run A (Referência) | Run B (Taxa Menor) — Campeã | Run C (Com L2) | Interpretação MLOps |
| :--- | :---: | :---: | :---: | :--- |
| **ID da Run no MLflow** | `339c966bfa88...` | `876c59546d4f...` | `3f4d77f70094...` | Identificadores únicos imutáveis no tracking server. |
| **Train Loss Inicial (Época 1)** | $0.6835$ | $0.8667$ | $0.6845$ | Run A e C descem mais rápido de início pela taxa $\eta=0.01$. |
| **Train Loss Final (Época 60)** | $0.3082$ | $0.3670$ | $0.2914$ | Run C atinge a menor perda de treino; Run B treina de forma controlada. |
| **Ponto Ótimo (Melhor Época)** | **Época 7** | **Época 16** | **Época 7** | Run B explora a bacia por mais tempo antes de atingir o mínimo. |
| **Menor Val Loss (Ponto Ótimo)** | $0.5875$ | **$0.5649$** | $0.5683$ | **Run B atinge a menor perda global de validação.** |
| **Val Loss Final (Época 60)** | $0.8851$ | **$0.6322$** | $0.9322$ | Run A (+50,6%) e Run C (+64,0%) degradam severamente. |
| **Melhor Val Accuracy** | $76,61\%$ | **$76,84\%$** | $76,38\%$ | Run B obtém a melhor precisão global entre as 3 runs. |
| **Val Accuracy Final (Época 60)** | $72,09\%$ | **$76,61\%$** | $74,80\%$ | Run B mantém sua acurácia estável; Run A perde 4,52 pontos percentuais. |
| **Val F1-Score Ponderado** | $0.7570$ | **$0.7633$** | $0.7520$ | Run B apresenta o melhor equilíbrio entre sensibilidade e precisão. |
| **Val F1-Score Macro** | $0.6909$ | **$0.7013$** | $0.6850$ | Run B é a única com F1-Macro superior a $0.70$ na validação. |
| **Gap de Generalização ($\Delta \mathcal{L}$)** | **$0.5769$** | **$0.2652$** | **$0.6408$** | **Run B apresenta gap 54% menor que a Run A e 58% menor que a C.** |

*Nota: Gap de Generalização calculado pela divergência absoluta final $\Delta \mathcal{L} = \mathcal{L}_{\text{val}}^{(\text{época } 60)} - \mathcal{L}_{\text{train}}^{(\text{época } 60)}$.*

---

## 3. Dinâmica das Curvas de Aprendizado no MLflow

A inspeção visual da aba **Compare** no MLflow revela comportamentos empíricos fundamentais da dinâmica de otimização de redes neurais:

```
Perda
 ^
 │  Run A: Inflexão na época 7 e subida íngreme para 0.8851 (Curva em "U")
 │  Run C: Mínimo em 0.5683 na época 7, explodindo para 0.9322 na época 60
 │  Run B: Descida suave até a época 16 (0.5649) e estabilização em 0.6322
 │
 │      \          Run A (Val Loss) ──────.─────────────> [0.8851]
 │       \        /                      /
 │        \      /   Run C (Val Loss) ───'──────────────> [0.9322]
 │         \    /
 │          \  /     Run B (Val Loss) ──────────────────> [0.6322]  (Campeã)
 │           \/
 │            \____  Run B (Train Loss) ────────────────> [0.3670]
 │                 \
 └──────────────────\───────────────────────────────────────> Épocas (1 a 60)
                     Ponto Ótimo Run B (Época 16: Loss=0.5649)
```

### 3.1. A Curva Clássica em "U" na Run A (Referência)
- **Fase de Aprendizado Rápido (Épocas 1 a 7):** A perda de treino declina de $0.6835$ para $0.5180$, e a perda de validação desce de $0.5989$ para o mínimo de $0.5875$. A rede assimila rapidamente os padrões mais proeminentes (ex: notas do 2º semestre e status de bolsista).
- **A Inflexão e Degradação (Épocas 8 a 60):** A partir da época 8, a perda de treino continua caindo até $0.3082$, mas a perda de validação começa uma escalada progressiva e errática, encerrando em $0.8851$. Configura-se a **curva em formato de "U" característica do sobreajuste (*overfitting*)**. A rede passa a memorizar ruídos e idiossincrasias das 2.654 amostras de treino, prejudicando sua calibragem de probabilidades.

### 3.2. A Convergência Estável e Controlada na Run B (Taxa Menor)
- **Fase de Descida Gradual (Épocas 1 a 16):** Com $\eta = 0.001$, os passos de atualização no espaço de parâmetros $\Theta$ são contidos. A perda de validação decai de forma contínua e monótona de $0.7057$ até registrar o **ponto ótimo global na época 16 (`val_loss = 0.5649`)**.
- **Fase de Platô Suave (Épocas 17 a 60):** Diferente da Run A, a curva de validação não sofre explosão após o ponto ótimo. Ela oscila suavemente entre $0.58$ e $0.63$, finalizando a época 60 em $0.6322$. O gap de generalização final é de apenas $0.2652$, demonstrando que a rede preservou sua capacidade preditiva em dados não vistos.

### 3.3. O Efeito Amortecedor da Regularização $L_2$ na Run C
- **Ação do Decaimento de Peso:** A adição de $\lambda = 1\times 10^{-5}$ modifica a função objetivo para $\mathcal{J}(\Theta) = \mathcal{L}_{\text{CE}} + \frac{\lambda}{2} \|\Theta\|_2^2$, subtraindo $\eta \lambda \Theta$ na regra de atualização.
- **Resultado Comparativo:** No ponto ótimo (época 7), o decaimento de peso conseguiu conter a magnitude dos tensores, gerando uma perda de validação de $0.5683$ (substancialmente melhor que os $0.5875$ da Run A). No entanto, como a taxa de aprendizado principal permaneceu elevada ($0.01$), a força do gradiente sobrepôs-se à penalização $L_2$ nas épocas finais, fazendo com que a perda terminasse em $0.9322$. Isso prova que **regularização $L_2$ isolada não substitui a calibração correta da taxa de aprendizado**.

---

## 4. O Diagnóstico do Incidente / Investigação Científica (Slide 13 da UNITINS)

Na atuação de Engenharia de MLOps, a observabilidade não se limita a coletar números, mas diagnosticar causalidades e atuar cientificamente sobre anomalias.

### 4.1. O Sintoma (A Anomalia da Run A)
- **Identificação nos Gráficos:** Estagnação prematura da validação na época 7, acompanhada por grande volatilidade e aumento da perda média por batch a partir da época 8, enquanto o erro de treino continuava colapsando linearmente.
- **Risco de Negócio:** Se o modelo da última época da Run A fosse implantado em produção, a taxa de erro de calibração (*Expected Calibration Error*) seria muito alta, emitindo previsões superconfiantes e incorretas sobre a evasão discente.

### 4.2. Hipótese Científica Formulada
- **Análise da Topologia da Superfície de Custo:** Redes neurais MLP em dados tabulares padronizados com ativação ReLU exibem superfícies de custo não convexas repletas de ravinas e mínimos locais com curvaturas acentuadas (grande autovalor na matriz Hessiana $\mathbf{H}$).
- **Causa Raiz:** O otimizador Adam com taxa $\eta = 0.01$ (dez vezes maior que a recomendação canônica de Kingma & Ba, 2014, de $\eta=0.001$) produz vetores de passo $\Delta \Theta = - \eta \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}$ com norma excessiva. Os pesos "saltam" de um lado para o outro das paredes da ravina, impedindo a descida para o fundo do vale e forçando o modelo a memorizar configurações estocásticas do lote.

### 4.3. A Intervenção Técnica e Validação
1. **Intervenção 1 (Moderação da Taxa — Run B):** Ao ajustar $\eta = 0.001$, os passos de atualização foram reduzidos à escala natural da curvatura. O gradiente conseguiu percorrer a trajetória ótima até a época 16, encontrando uma bacia de atração mais larga e estável, o que resultou na vitória incontestável em todas as métricas de validação.
2. **Intervenção 2 (Atrito Analítico de Regularização — Run C):** Ao introduzir o atrito geométrico do $L_2$, provou-se que a contração de pesos reduz a perda mínima na época 7, mas que, na ausência de amortecimento do *learning rate*, a instabilidade a longo prazo prevalece.

---

## 5. Aplicação da Regra de Ouro da UNITINS: Avaliação no Teste Reservado

Em estrito cumprimento do **Slide 12 da UNITINS**, a seleção do modelo foi realizada **única e exclusivamente através da análise das curvas e métricas da partição de Validação**.

### 5.1. Eleição da Run Campeã
- **Critério de Decisão:** Menor perda de validação no ponto ótimo (`val_loss = 0.5649`), maior acurácia de validação (`76,84%`) e maior F1-Score ponderado (`0.7633`).
- **Modelo Vencedor:** **`Run_B_Taxa_Menor`** (pesos salvos no checkpoint [`artifacts/best_model_Run_B_Taxa_Menor.pt`](file:///c:/Users/itofr/Documents/Testes/IA-MlFlow/artifacts/best_model_Run_B_Taxa_Menor.pt)).

### 5.2. Avaliação Única no Teste Reservado (Dados Cegos — $N=885$)
Os pesos ótimos da Run B foram carregados no modo de inferência (`model.eval()`) e submetidos **uma única vez** aos 885 registros da partição de Teste Reservado:

```
================================================================================
DESEMPENHO DO MODELO CAMPEÃO (Run_B_Taxa_Menor) NO TESTE RESERVADO
================================================================================
Perda no Teste (CrossEntropyLoss):  0.5820
Acurácia Global no Teste:          0.7774 (77,74%)
F1-Score Ponderado:                0.7742
F1-Score Macro:                    0.7251
Amostras Avaliadas:                885 estudantes (não vistos no treino/validação)
================================================================================
```

### 5.3. Relatório Detalhado de Classificação por Categoria
```
              precision    recall  f1-score   support

     Dropout     0.8199    0.7535    0.7853       284
    Enrolled     0.5616    0.5157    0.5377       159
    Graduate     0.8201    0.8869    0.8522       442

    accuracy                         0.7774       885
   macro avg     0.7339    0.7187    0.7251       885
weighted avg     0.7736    0.7774    0.7742       885
```

### 5.4. Interpretação Qualitativa da Matriz de Confusão

A matriz de confusão gerada e salva no artefato gráfico [`artifacts/matriz_confusao_teste.png`](file:///c:/Users/itofr/Documents/Testes/IA-MlFlow/artifacts/matriz_confusao_teste.png) sintetiza os acertos e cruzamentos preditivos:

```
                       CLASSE PREVISTA (PREDICTED)
                 Dropout (0)   Enrolled (1)   Graduate (2)    Total Real
               ┌─────────────┬──────────────┬──────────────┐
  Dropout (0)  │     214     │      31      │      39      │     284
CLASSE         ├─────────────┼──────────────┼──────────────┤
REAL   Enrolled│      30     │      82      │      47      │     159
(TRUE)         ├─────────────┼──────────────┼──────────────┤
  Graduate (2) │      17     │      33      │     392      │     442
               └─────────────┴──────────────┴──────────────┘
  Total Prev.       261            146            478            885
```

#### Diagnóstico por Classe de Negócio:
1. **Graduados / Concluintes (`Graduate` — Classe 2):**
   - *Desempenho:* $392$ acertos em $442$ amostras reais. Sensibilidade (*recall*) excelente de **$88,69\%$** e precisão de **$82,01\%$** (F1 de **$0.8522$**).
   - *Impacto:* A rede aprendeu com altíssima segurança a identificar os discentes com trajetória acadêmica regular e aprovações contínuas nos primeiros semestres.
2. **Evasores (`Dropout` — Classe 0):**
   - *Desempenho:* $214$ acertos em $284$ amostras reais. Precisão expressiva de **$81,99\%$** e sensibilidade de **$75,35\%$** (F1 de **$0.7853$**).
   - *Impacto:* Para a gestão acadêmica da universidade, a precisão de $82\%$ significa que, quando o sistema emitir um alerta de risco de evasão, em mais de $8$ de cada $10$ casos a ameaça é real, viabilizando intervenções da assistência estudantil e tutoria sem desperdício de recursos institucionais. Apenas $17$ graduados foram erroneamente classificados como evasores (baixa taxa de falso-positivo grave: $\frac{17}{442} \approx 3,8\%$).
3. **Matriculados (`Enrolled` — Classe 1):**
   - *Desempenho:* $82$ acertos em $159$ amostras reais. Precisão de **$56,16\%$** e sensibilidade de **$51,57\%$** (F1 de **$0.5377$**).
   - *Causa Técnica:* A classe "Matriculado" é intrinsecamente ruidosa e instável no domínio educacional, pois funciona como uma zona de transição e indefinição curricular. São discentes que ainda não evadiram formalmente, mas acumulam reprovações e trancamentos parciais, assemelhando-se matematicamente tanto ao perfil de evasão quanto ao de graduação tardia.
