# Especificação de Arquitetura e Modelagem Matemática
## Atividade Prática Avaliativa de Inteligência Artificial — UNITINS
**Curso:** Bacharelado em Sistemas de Informação / Engenharia de Software  
**Disciplina:** Inteligência Artificial  
**Docente:** Prof. Me. Marco Antonio Firmino de Sousa  
**Discente:** João Victor Ito  

---

## 1. Contextualização do Problema de Negócio em Sistemas de Informação

### 1.1. Evasão Universitária como Desafio em ERPs Acadêmicos
A evasão no ensino superior constitui um dos mais severos gargalos socioeconômicos e pedagógicos enfrentados por instituições de ensino públicas e privadas. Nos Sistemas Integrados de Gestão Acadêmica (Sistemas ERP Acadêmico), o registro histórico de matrículas, notas semestrais, situação socioeconômica e dados demográficos tradicionalmente atua de forma passiva — isto é, relata a perda do discente apenas após a consumação do desligamento institucional (*ex-post facto*).

A integração de modelos preditivos conexionistas supervisionados a essas plataformas permite converter o banco transacional em um sistema proativo de alerta precoce (*Early Warning System*). O problema é modelado matematicamente como uma tarefa de **classificação supervisionada multiclasse em três vias**, mapeando o vetor de atributos do estudante $\mathbf{x} \in \mathbb{R}^d$ para uma das três classes mutuamente exclusivas $y \in \{0, 1, 2\}$:
- **Classe 0 — Dropout (Evasão):** Estudantes que abandonaram ou cancelaram formalmente o curso antes de sua integralização. Representa o principal foco de intervenção pedagógica e retenção.
- **Classe 1 — Enrolled (Matriculado / Em Curso):** Estudantes com vínculo institucional ativo que ainda não completaram os créditos necessários ou encontram-se em situação de retenção curricular/atraso.
- **Classe 2 — Graduate (Graduado / Concluinte):** Estudantes que integralizaram a matriz curricular e concluíram o curso com êxito.

### 1.2. O Dataset "Predict Students' Dropout and Academic Success"
O conjunto de dados adotado provém do estudo conduzido no Instituto Politécnico de Portalegre (Portugal), disponibilizado publicamente no Kaggle sob o identificador `thedevastator/higher-education-predictors-of-student-retention`. Ele engloba **4.424 registros discentes** distribuídos em **34 variáveis preditivas** após higienização tabular (e 1 coluna alvo `Target`), contemplando três dimensões ontológicas fundamentais:

1. **Dimensão Demográfica e Familiar:** Estado civil, nacionalidade, idade no ato da matrícula, nível educacional dos genitores (mãe e pai), ocupação profissional dos genitores.
2. **Dimensão Socioeconômica e Institucional:** Regime de frequência (diurno/noturno), deslocamento domiciliar (*displaced*), necessidades educacionais especiais, condição de devedor de mensalidades (*debtor*), propinas regularizadas (*tuition fees up to date*), bolsista de estudos (*scholarship holder*), estudante internacional, indicadores macroeconômicos (taxa de desemprego, taxa de inflação e PIB nacional).
3. **Dimensão de Desempenho Acadêmico Curricular:** Número de unidades curriculares creditadas, inscritas, avaliadas e aprovadas no 1º e 2º semestres, bem como as médias ponderadas obtidas e unidades curriculares não avaliadas.

---

## 2. Topologia da Rede Neural (PyTorch Puro)

### 2.1. Definição Estrutural
O modelo preditivo foi concebido na classe [`StudentRetentionMLP`](file:///c:/Users/itofr/Documents/Testes/IA-MlFlow/src/model.py#L18) herdando de `torch.nn.Module`, configurando uma arquitetura **Multilayer Perceptron (MLP)** totalmente conexa (*fully connected feedforward network*). A rede mapeia a entrada $\mathbf{x} \in \mathbb{R}^{34}$ em um vetor de probabilidades não calibradas (logits) $\mathbf{z} \in \mathbb{R}^3$.

```
=======================================================================================================
TOPOLOGIA DETALHADA DA REDE NEURAL (StudentRetentionMLP)
=======================================================================================================

      [Vetor de Entrada Padronizado]
             x ∈ ℝ³⁴ (34 Features Normalizadas via StandardScaler)
                     │
                     ▼
      ┌─────────────────────────────┐
      │     Linear Layer 1          │  W¹ ∈ ℝ⁶⁴ˣ³⁴, b¹ ∈ ℝ⁶⁴  ==> 2.240 parâmetros
      │       (34 -> 64)            │
      └──────────────┬──────────────┘
                     ▼
      ┌─────────────────────────────┐
      │     Ativação ReLU           │  f(u) = max(0, u)
      └──────────────┬──────────────┘
                     ▼
      ┌─────────────────────────────┐
      │     Dropout (p = 0.2)       │  Máscara Bernoulli Invertida (Regularização)
      └──────────────┬──────────────┘
                     ▼
      ┌─────────────────────────────┐
      │     Linear Layer 2          │  W² ∈ ℝ³²ˣ⁶⁴, b² ∈ ℝ³²  ==> 2.080 parâmetros
      │       (64 -> 32)            │
      └──────────────┬──────────────┘
                     ▼
      ┌─────────────────────────────┐
      │     Ativação ReLU           │  f(u) = max(0, u)
      └──────────────┬──────────────┘
                     ▼
      ┌─────────────────────────────┐
      │     Linear Layer 3 (Saída)  │  W³ ∈ ℝ³ˣ³², b³ ∈ ℝ³    ==>    99 parâmetros
      │        (32 -> 3)            │
      └──────────────┬──────────────┘
                     ▼
      [Vetor de Logits Brutos]
             z ∈ ℝ³  (z₀ = Dropout, z₁ = Enrolled, z₂ = Graduate)
                     │
                     ▼
       (Acoplado a nn.CrossEntropyLoss durante o Treinamento)
```

### 2.2. Cálculo Formal e Parametrização Exata
Para uma camada densa linear de entrada $N_{\text{in}}$ e saída $N_{\text{out}}$, o total de parâmetros treináveis é dado pela soma da matriz de pesos $\mathbf{W} \in \mathbb{R}^{N_{\text{out}} \times N_{\text{in}}}$ com o vetor de bias $\mathbf{b} \in \mathbb{R}^{N_{\text{out}}}$:

$$\Theta_{\text{camada}} = (N_{\text{in}} \times N_{\text{out}}) + N_{\text{out}}$$

Aplicando aos tensores da [`StudentRetentionMLP`](file:///c:/Users/itofr/Documents/Testes/IA-MlFlow/src/model.py#L18):

| Componente | Tensor de Pesos ($\mathbf{W}$) | Dimensão $\mathbf{W}$ | Tensor de Bias ($\mathbf{b}$) | Dimensão $\mathbf{b}$ | Total de Parâmetros |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Camada Oculta 1 (`Linear(34, 64)`)** | `network.0.weight` | $[64, 34]$ ($2.176$) | `network.0.bias` | $[64]$ ($64$) | **$2.240$** |
| **Ativação ReLU 1** | — | — | — | — | $0$ |
| **Dropout ($p=0.2$)** | — | — | — | — | $0$ |
| **Camada Oculta 2 (`Linear(64, 32)`)** | `network.3.weight` | $[32, 64]$ ($2.048$) | `network.3.bias` | $[32]$ ($32$) | **$2.080$** |
| **Ativação ReLU 2** | — | — | — | — | $0$ |
| **Camada de Saída (`Linear(32, 3)`)** | `network.5.weight` | $[3, 32]$ ($96$) | `network.5.bias` | $[3]$ ($3$) | **$99$** |
| **TOTAL GERAL DE PARÂMETROS TREINÁVEIS** | | | | | **$4.419$** |

---

## 3. Fundamentação Matemática das Operações

### 3.1. Propagação Direta (Forward Pass)
Dada uma matriz de mini-lote $\mathbf{X} \in \mathbb{R}^{B \times 34}$ (onde $B=32$ é o tamanho do lote), as transformações afins e ativações ocorrem conforme o encadeamento:

1. **Transformação da Camada Oculta 1:**
   $$\mathbf{U}^{(1)} = \mathbf{X} (\mathbf{W}^{(1)})^T + \mathbf{1}_B (\mathbf{b}^{(1)})^T \quad \in \mathbb{R}^{B \times 64}$$
   $$\mathbf{A}^{(1)} = \text{ReLU}\left(\mathbf{U}^{(1)}\right) \quad \in \mathbb{R}^{B \times 64}$$
   $$\widetilde{\mathbf{A}}^{(1)} = \text{Dropout}_{0.2}\left(\mathbf{A}^{(1)}\right) \quad \in \mathbb{R}^{B \times 64}$$

2. **Transformação da Camada Oculta 2:**
   $$\mathbf{U}^{(2)} = \widetilde{\mathbf{A}}^{(1)} (\mathbf{W}^{(2)})^T + \mathbf{1}_B (\mathbf{b}^{(2)})^T \quad \in \mathbb{R}^{B \times 32}$$
   $$\mathbf{A}^{(2)} = \text{ReLU}\left(\mathbf{U}^{(2)}\right) \quad \in \mathbb{R}^{B \times 32}$$

3. **Geração de Logits na Camada de Saída:**
   $$\mathbf{Z} = \mathbf{A}^{(2)} (\mathbf{W}^{(3)})^T + \mathbf{1}_B (\mathbf{b}^{(3)})^T \quad \in \mathbb{R}^{B \times 3}$$

### 3.2. Função de Ativação ReLU (Rectified Linear Unit)
A função não linear de ativação é definida formalmente por:

$$f(u) = \max(0, u) = \begin{cases} u, & \text{se } u > 0 \\ 0, & \text{se } u \le 0 \end{cases}$$

Sua derivada generalizada (subgradiente em $u=0$) é dada por:

$$f'(u) = \begin{cases} 1, & \text{se } u > 0 \\ 0, & \text{se } u < 0 \end{cases}$$

#### Vantagens Críticas sobre Sigmoide e Tangente Hiperbólica
1. **Mitigação do Desaparecimento do Gradiente (*Vanishing Gradient*):** Para ativações saturantes como a Sigmoide $\sigma(u) = \frac{1}{1 + e^{-u}}$ ou a Tanh $\tanh(u) = \frac{e^u - e^{-u}}{e^u + e^{-u}}$, as derivadas máximas são respectivamente $\sigma'(u) \le 0.25$ e $\tanh'(u) \le 1.0$, saturando rapidamente em zero quando $|u| \gg 0$. Durante o algoritmo de retropropagação (*backpropagation*), a multiplicação encadeada de derivadas estritamente inferiores a $0.25$ anula a magnitude do gradiente nas primeiras camadas:
   $$\lim_{l \to 1} \frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(l)}} \approx \mathbf{0}$$
   Com a ReLU, para qualquer neurônio ativo ($u > 0$), a derivada local é identicamente $1.0$, assegurando fluxo irrestrito de gradiente através das camadas lineares.
2. **Eficiência Computacional Extrema:** O cálculo de $\max(0, u)$ requer apenas uma operação de comparação de ponto flutuante no nível de instrução de máquina (ex: instrução AVX/SSE `MAXPS`), dispensando cálculos onerosos de exponenciais e divisões.
3. **Esparsidade de Representação:** Neurônios com ativação nula induzem representações esparsas, promovendo ortogonalidade entre representações latentes.

### 3.3. Mecanismo de Dropout Invertido (Inverted Dropout)
Na camada oculta 1, foi incorporado o operador de Dropout com probabilidade de descarte $p = 0.2$. Em tempo de treinamento, para cada neurônio $j \in \{1, \dots, 64\}$, gera-se uma variável aleatória independente de Bernoulli:

$$r_j \sim \text{Bernoulli}(1 - p), \quad P(r_j = 1) = 1 - p = 0.8$$

O PyTorch adota a implementação de **Inverted Dropout**, aplicando o escalonamento compensatório em tempo de treinamento (*train mode*):

$$\widetilde{a}_j = \frac{r_j}{1 - p} a_j = \frac{r_j}{0.8} a_j$$

#### Propriedades Matemáticas do Dropout Invertido
1. **Invariância de Expectância:** A expectância condicional do neurônio ativo permanece idêntica à de inferência:
   $$\mathbb{E}[\widetilde{a}_j] = \frac{\mathbb{E}[r_j]}{1 - p} a_j = \frac{1 - p}{1 - p} a_j = a_j$$
   Isso elimina a necessidade de multiplicar os pesos por $(1-p)$ durante a avaliação (`model.eval()`), acelerando o tempo de inferência e simplificando o empacotamento do modelo.
2. **Quebra de Co-adaptação Neuronal:** Ao desligar aleatoriamente 20% dos neurônios a cada iteração, nenhum neurônio pode confiar na presença sistemática de outro, forçando a rede a aprender representações redundantes, distribuídas e robustas.
3. **Aproximação de Ensemble Exponencial:** Treinar uma camada com $N$ neurônios sob Dropout equivale a otimizar um ensemble de $2^N$ sub-redes com pesos compartilhados, onde a predição na fase de teste aproxima a média geométrica das predições do ensemble (Srivastava et al., 2014).

### 3.4. Função de Custo CrossEntropyLoss Multiclasse
A camada final emite o vetor de logits $\mathbf{z} = [z_0, z_1, z_2]^T$. A função de probabilidade predita para a classe $c \in \{0, 1, 2\}$ é obtida via operador Softmax:

$$\hat{y}_c = \frac{e^{z_c}}{\sum_{k=0}^{2} e^{z_k}}$$

A perda empírica para uma instância com rótulo real em representação *one-hot* $\mathbf{y} \in \{0, 1\}^3$ (onde $y_c = 1$ para a classe verdadeira e $0$ para as demais) corresponde à perda de Log-Verossimilhança Negativa (*Negative Log-Likelihood*):

$$\mathcal{L}_{\text{CE}}(\mathbf{z}, \mathbf{y}) = - \sum_{c=0}^{2} y_c \log(\hat{y}_c) = - \log\left( \frac{e^{z_t}}{\sum_{k=0}^2 e^{z_k}} \right) = -z_t + \log\left( \sum_{k=0}^2 e^{z_k} \right)$$

onde $t$ é o índice da classe verdadeira ($t = \text{argmax}(\mathbf{y})$).

#### Estabilidade Numérica: O Truque do Log-Sum-Exp (LSE)
O `torch.nn.CrossEntropyLoss` opera diretamente sobre os logits sem exigir a aplicação explícita de `torch.softmax` seguida de `torch.log`. Na computação em ponto flutuante IEEE 754 de precisão simples (FP32), calcular $e^{z_k}$ diretamente acarreta *overflow* aritmético se $z_k > 88.7$, ou *underflow* severo se $z_k \ll 0$. O PyTorch emprega o truque analítico:

$$\log\left(\sum_{k=0}^{2} e^{z_k}\right) = M + \log\left(\sum_{k=0}^{2} e^{z_k - M}\right), \quad M = \max_{k}(z_k)$$

Essa formulação garante que o maior expoente seja sempre $e^0 = 1$, blindando o pipeline de treinamento contra valores `NaN` ou `Inf`.

---

## 4. Conexão com a Literatura Clássica (Russell & Norvig, Goodfellow et al.)

### 4.1. O Teorema da Aproximação Universal e Fronteiras Não Lineares
De acordo com Russell & Norvig (*Artificial Intelligence: A Modern Approach*, Cap. 21) e Goodfellow, Bengio & Courville (*Deep Learning*, Cap. 6), modelos lineares ou Perceptrons de camada única são restritos a fronteiras de decisão hiperplanares, falhando na resolução de problemas não linearmente separáveis (conforme demonstrado historicamente por Minsky & Papert em 1969 no problema do XOR).

O **Teorema da Aproximação Universal** (Cybenko, 1989; Hornik, 1991) estabelece que uma rede feedforward contendo pelo menos uma camada oculta com funções de ativação não lineares contínuas pode aproximar qualquer função contínua de Borel em subconjuntos compactos de $\mathbb{R}^d$ com precisão arbitrária $\epsilon > 0$, desde que possua um número suficiente de neurônios.

No problema de evasão estudantil, a fronteira entre um discente que desiste (*Dropout*) e um que permanece matriculado com dificuldades (*Enrolled*) envolve acoplamentos altamente não lineares (por exemplo: um aluno com baixa nota no 1º semestre pode persistir caso seja bolsista e não possua débitos, enquanto outro com nota mediana pode evadir se possuir dívidas financeiras e residir fora do polo de estudo). A composição em duas camadas ocultas ($34 \to 64 \to 32 \to 3$) dota a [`StudentRetentionMLP`](file:///c:/Users/itofr/Documents/Testes/IA-MlFlow/src/model.py#L18) da capacidade representacional necessária para deformar e projetar o espaço latente de atributos em regiões linearmente separáveis na camada de saída.

### 4.2. O Princípio da Parcimônia (Navalha de Ockham) e Capacidade de Hipótese
O Princípio da Parcimônia (*Navalha de Ockham*), pedra angular da Teoria do Aprendizado Estatístico (Vapnik, 1998; Russell & Norvig, Cap. 18), preconiza que, entre hipóteses concorrentes que explicam satisfatoriamente as observações, a hipótese mais simples deve ser a escolhida.

No projeto da [`StudentRetentionMLP`](file:///c:/Users/itofr/Documents/Testes/IA-MlFlow/src/model.py#L18):
- O conjunto de treino possui $N_{\text{train}} = 2.654$ instâncias.
- A rede possui $\Theta = 4.419$ parâmetros treináveis.
- A relação entre parâmetros e amostras de treino é de aproximadamente $1,66$ parâmetros por amostra ($\frac{4.419}{2.654} \approx 1,66$).

Se a arquitetura fosse superdimensionada (ex: camadas com 512 ou 1024 neurônios, resultando em centenas de milhares de parâmetros), a Dimensão Vapnik-Chervonenkis (Dimensão VC) da classe de hipóteses $\mathcal{H}$ explodiria, permitindo que a rede simplesmente memorizasse os ruídos do conjunto de treino sem generalizar para dados não vistos. O dimensionamento deliberado em $64 \to 32$, associado ao Dropout ($p=0.2$) e ao controle do decaimento de peso ($L_2$), restringe a capacidade efetiva da hipótese, garantindo que a rede capture os padrões estruturais genuínos de retenção discente.
