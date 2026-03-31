# Escopo do Projeto: Módulo de Classificação de Classe e Assunto Processual

## 1. Visão Geral e Objetivo
O objetivo deste projeto é estender o pipeline de auditoria de metadados existente (focado em extração NER/Regex) para suportar a **Classificação de Texto (Text Classification)**. O sistema deverá prever automaticamente duas dimensões processuais categóricas: `Classe` e `Assunto`, a partir do texto não estruturado das petições iniciais (`inteiro_teor`).

## 2. Fonte de Dados
* **Dataset Base:**Os arquivos estão localizados na pasta `data/raw/`.
* **Features de Entrada (X):** `inteiro_teor` (texto bruto da petição).
* **Labels Alvo (Y):** `classe` e `assunto`.
* **Identificador:** `numero_processo`.

## 3. Arquitetura Proposta
Para garantir velocidade de processamento `O(1)/O(n)` e alta precisão, o módulo utilizará uma arquitetura em cascata (fallback) de 3 níveis:

### Nível 1: Heurística Estrutural (Fast-Path)
* **Técnica:** Busca por palavras-chave e padrões no cabeçalho.
* **Objetivo:** Capturar processos com classes explícitas (ex: "AÇÃO DE COBRANÇA") com custo computacional mínimo.
* **Condição de Saída:** Match exato de alta confiança -> FIM. Caso contrário -> Nível 2.

### Nível 2: Classificador Estatístico (Workhorse)
* **Técnica:** Vetorização (TF-IDF com N-grams) combinada com algoritmos baseados em árvores (LightGBM ou XGBoost).
* **Objetivo:** Processar o grande volume de dados identificando correlações estatísticas entre termos e classes/assuntos.
* **Condição de Saída:** Predição com probabilidade > Limiar (ex: 85%) -> FIM. Caso contrário (baixa confiança) -> Nível 3.

### Nível 3: Deep Learning / Semântica
* **Técnica:** Modelos baseados em Transformers (ex: LegalBERT, SetFit) ou RAG com LLMs leves ou GLINER
* **Objetivo:** Resolver casos ambíguos ou classes da "cauda longa" (raras), classificando pelo contexto semântico.
* **Condição de Saída:** Classificação final de desempate.

---

## 4. Instruções para o Agente: Dinâmica de Agents e Skills
Para acelerar o desenvolvimento de forma padronizada, as tarefas devem ser delegadas aos componentes do framework localizados na estrutura `.agent/`. Vc deve instanciar os agentes e skills necessários para executar as tarefas.

### 4.1. Compreensão de Dados (Data Understanding)
Antes de sugerir técnicas definitivas ou escrever código de treinamento, o sistema/agente deve executar scripts de leitura nos seguintes artefatos para obter clareza do cenário:

1.  **Análise de Dados Estruturados (`data/raw/`):** * Ler os arquivos `amostra_processos_assuntos_recorrentes_27022025.csv` e `amostra_processos_classes_recorrentes_27022025.csv`. 
    * **Objetivo:** Quantificar valores únicos, mapear o desbalanceamento das classes (identificar a cauda longa) e calcular a distribuição do tamanho (número de tokens) do `inteiro_teor`.
2.  **Análise de Dados Não-Estruturados (`modelos petições/`):** * Processar os arquivos de `peticao1.pdf` até `peticao10.pdf`.
    * **Objetivo:** Analisar fisicamente a estrutura dos documentos para entender como os advogados redigem os cabeçalhos. Isso balizará as regras de extração do "Nível 1".

### 4.2. Estrutura de Agentes a serem instanciados (`.agent/agents/`)
* `DataScientist_Agent`: Encarregado da Fase de Análise Exploratória (EDA), divisão de datasets (Train/Test) e do treinamento e otimização de hiperparâmetros do modelo estatístico (Nível 2).
* `NLPEngineer_Agent`: Focado nas regras de linguagem. Construirá as Regex/Aho-Corasick para o Nível 1 e fará a implementação dos Embeddings ou prompts semânticos para o Nível 3.
* `BackendIntegration_Agent`: Encarregado de envelopar os modelos `.pkl` gerados e integrá-los no `src/pipeline_manager.py` de forma modular e assíncrona, atualizando a matriz de decisão.

### 4.3. Estrutura de Skills a serem criadas (`.agent/skills/`)
As seguintes *skills* devem ser codificadas para tornar o workflow reutilizável:
* `skill_legal_text_cleaner`: Ferramenta de pré-processamento. Remove stopwords jurídicas ("Excelentíssimo", "Vem respeitosamente"), corrige mojibakes de OCR e aplica tokenização adequada.
* `skill_pdf_to_text`: Extrai e normaliza texto bruto dos arquivos PDF da pasta `modelos petições/` para validar testes unitários no padrão de entrada real.
* `skill_train_evaluate_lgbm`: Pipeline automatizado que recebe um CSV limpo, treina um classificador LightGBM com TF-IDF, avalia métricas (F1-Score, Precision, Recall) e salva o modelo no diretório de saída.

## 5. Fases de Execução
1.  **Fase 1:** Ativar `DataScientist_Agent` utilizando a `skill_legal_text_cleaner` para gerar relatórios de EDA em `data/processed/` a partir dos arquivos brutos. Decisão técnica documentada sobre as classes a serem modeladas.
2.  **Fase 2:** Ativar `NLPEngineer_Agent` utilizando `skill_pdf_to_text` nos PDFs para criar a Heurística do Nível 1.
3.  **Fase 3:** Treinamento em Notebook e exportação do modelo Nível 2.
4.  **Fase 4:** Integração no backend principal.