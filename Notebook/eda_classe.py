# ==============================================================================
# EDA - Análise Exploratória: Dataset de CLASSES Processuais
# ==============================================================================
# Este script analisa o dataset de classes recorrentes para entender:
# 1. Estrutura e qualidade dos dados
# 2. Distribuição das classes (cauda longa)
# 3. Tamanho dos textos (inteiro_teor)
# 4. Dados nulos e anomalias
# 5. Estrutura hierárquica das classes (separador "->")
#
# Saída: Gráficos salvos em data/processed/eda_charts/
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from pathlib import Path

# ==============================================================================
# CONFIGURAÇÕES
# ==============================================================================
FILE_PATH = r'C:\Users\mlzengo\Documents\TJGO\II SEMESTRE\classe-assuno\data\raw\amostra_processos_classes_recorrentes_27022025.csv'
OUTPUT_DIR = r'C:\Users\mlzengo\Documents\TJGO\II SEMESTRE\classe-assuno\data\processed\eda_charts'
DATASET_NAME = 'Classes'
TARGET_COL = 'classe'
TEXT_COL = 'inteiro_teor'
SEP = '#'
SAMPLE_SIZE = 10000

# Criar diretório de saída
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Style
sns.set_theme(style='whitegrid', font_scale=1.2)
plt.rcParams['figure.dpi'] = 120
plt.rcParams['savefig.bbox'] = 'tight'

print("=" * 70)
print(f"  EDA - Dataset: {DATASET_NAME}")
print("=" * 70)

# ==============================================================================
# 1. CARREGAMENTO E INSPEÇÃO ESTRUTURAL
# ==============================================================================
print("\n📂 [1/7] Carregando dados (amostra)...")

df_sample = pd.read_csv(FILE_PATH, sep=SEP, encoding='utf-8', nrows=SAMPLE_SIZE)

print(f"  ✓ Amostra carregada: {len(df_sample):,} linhas")
print(f"  Colunas: {df_sample.columns.tolist()}")
print(f"  Dtypes:\n{df_sample.dtypes.to_string()}")

# Contar linhas totais
print("\n  Contando total de linhas no arquivo completo...")
total_lines = 0
with open(FILE_PATH, 'r', encoding='utf-8') as f:
    for _ in f:
        total_lines += 1
total_lines -= 1
print(f"  ✓ Total de linhas no arquivo: {total_lines:,}")

# ==============================================================================
# 2. ANÁLISE DE VALORES NULOS E QUALIDADE
# ==============================================================================
print("\n🔍 [2/7] Análise de qualidade dos dados...")

nulls = df_sample.isnull().sum()
nulls_pct = (df_sample.isnull().sum() / len(df_sample) * 100).round(2)
quality_df = pd.DataFrame({'nulos': nulls, 'pct': nulls_pct})
print(f"\n  Valores nulos (amostra de {SAMPLE_SIZE:,}):")
print(quality_df.to_string())

duplicates = df_sample['numero_processo'].duplicated().sum()
print(f"\n  Processos duplicados na amostra: {duplicates}")

empty_text = (df_sample[TEXT_COL].isna() | (df_sample[TEXT_COL].str.len() < 50)).sum()
print(f"  Textos vazios ou < 50 chars: {empty_text} ({empty_text/len(df_sample)*100:.1f}%)")

# ==============================================================================
# 3. ANÁLISE DA ESTRUTURA HIERÁRQUICA DAS CLASSES
# ==============================================================================
print("\n🏗️ [3/7] Analisando estrutura hierárquica das classes...")

# Classes contêm hierarquia com " -> "
# Ex: "PROCESSO CÍVEL E DO TRABALHO -> Processo de Execução -> Execução Fiscal"
sample_classes = df_sample[TARGET_COL].dropna()

# Contar níveis hierárquicos
hierarchy_levels = sample_classes.str.count(r'->').add(1)
print(f"  Níveis hierárquicos:")
print(f"    Mínimo: {hierarchy_levels.min()}")
print(f"    Máximo: {hierarchy_levels.max()}")
print(f"    Mediana: {hierarchy_levels.median()}")

# Extrair nível 1 (categoria raiz)
root_classes = sample_classes.str.split(r'\s*->\s*').str[0].str.strip()
print(f"\n  Classes raiz (Nível 1):")
for cls, count in root_classes.value_counts().items():
    pct = count / len(root_classes) * 100
    print(f"    • {cls}: {count:,} ({pct:.1f}%)")

# Extrair nível folha (classificação final)
leaf_classes = sample_classes.str.split(r'\s*->\s*').str[-1].str.strip()
print(f"\n  Total de classes folha únicas (amostra): {leaf_classes.nunique()}")

# ==============================================================================
# 4. DISTRIBUIÇÃO DAS CLASSES - ARQUIVO COMPLETO COM CHUNKS
# ==============================================================================
print("\n📊 [4/7] Contando distribuição de classes (arquivo completo, chunked)...")

classe_counts = pd.Series(dtype='int64')
root_counts_full = pd.Series(dtype='int64')
leaf_counts_full = pd.Series(dtype='int64')
chunk_num = 0

for chunk in pd.read_csv(FILE_PATH, sep=SEP, encoding='utf-8', chunksize=10000,
                          usecols=[TARGET_COL]):
    # Classe completa
    classe_counts = classe_counts.add(chunk[TARGET_COL].value_counts(), fill_value=0)
    
    # Classe raiz
    roots = chunk[TARGET_COL].dropna().str.split(r'\s*->\s*').str[0].str.strip()
    root_counts_full = root_counts_full.add(roots.value_counts(), fill_value=0)
    
    # Classe folha
    leaves = chunk[TARGET_COL].dropna().str.split(r'\s*->\s*').str[-1].str.strip()
    leaf_counts_full = leaf_counts_full.add(leaves.value_counts(), fill_value=0)
    
    chunk_num += 1
    if chunk_num % 30 == 0:
        print(f"    ... processados {chunk_num * 10000:,} linhas")

classe_counts = classe_counts.astype(int).sort_values(ascending=False)
root_counts_full = root_counts_full.astype(int).sort_values(ascending=False)
leaf_counts_full = leaf_counts_full.astype(int).sort_values(ascending=False)

print(f"  ✓ Total de classes únicas (completa): {len(classe_counts)}")
print(f"  ✓ Total de classes raiz únicas: {len(root_counts_full)}")
print(f"  ✓ Total de classes folha únicas: {len(leaf_counts_full)}")

# Top 20
print(f"\n  Top 20 classes mais frequentes:")
for i, (classe, count) in enumerate(classe_counts.head(20).items(), 1):
    pct = count / classe_counts.sum() * 100
    print(f"    {i:2d}. {classe[:70]:<70s} | {count:>7,} ({pct:.1f}%)")

# Pareto
cumsum = classe_counts.cumsum() / classe_counts.sum()
n_80pct = (cumsum <= 0.80).sum()
print(f"\n  📈 Regra 80/20: {n_80pct} classes cobrem 80% do volume "
      f"(de {len(classe_counts)} totais)")

# Classes raras
rare_classes = (classe_counts < 10).sum()
very_rare = (classe_counts < 3).sum()
print(f"  ⚠️  Classes com < 10 amostras: {rare_classes}")
print(f"  ⚠️  Classes com < 3 amostras: {very_rare}")

# --- GRÁFICO: Top 20 Classes ---
fig, ax = plt.subplots(figsize=(14, 10))
top20 = classe_counts.head(20)
colors = sns.color_palette("viridis", n_colors=20)
bars = ax.barh(range(len(top20)), top20.values, color=colors)
ax.set_yticks(range(len(top20)))
ax.set_yticklabels([label[:55] for label in top20.index], fontsize=10)
ax.invert_yaxis()
ax.set_xlabel('Frequência')
ax.set_title(f'Top 20 Classes Processuais Mais Frequentes\n(Total: {len(classe_counts)} classes únicas | {classe_counts.sum():,} processos)')

for bar, val in zip(bars, top20.values):
    ax.text(bar.get_width() + classe_counts.max() * 0.01, bar.get_y() + bar.get_height()/2,
            f'{val:,.0f}', va='center', fontsize=9)

plt.savefig(f'{OUTPUT_DIR}/classe_top20.png')
plt.close()
print(f"  ✓ Gráfico salvo: classe_top20.png")

# --- GRÁFICO: Classes Raiz (Nível 1 da hierarquia) ---
fig, ax = plt.subplots(figsize=(12, 6))
root_colors = sns.color_palette("Set2", n_colors=len(root_counts_full))
bars = ax.barh(range(len(root_counts_full)), root_counts_full.values, color=root_colors)
ax.set_yticks(range(len(root_counts_full)))
ax.set_yticklabels(root_counts_full.index, fontsize=11)
ax.invert_yaxis()
ax.set_xlabel('Frequência')
ax.set_title(f'Distribuição por Classe Raiz (Nível 1 da Hierarquia)')

for bar, val in zip(bars, root_counts_full.values):
    ax.text(bar.get_width() + root_counts_full.max() * 0.01, bar.get_y() + bar.get_height()/2,
            f'{val:,.0f}', va='center', fontsize=9)

plt.savefig(f'{OUTPUT_DIR}/classe_root_distribution.png')
plt.close()
print(f"  ✓ Gráfico salvo: classe_root_distribution.png")

# --- GRÁFICO: Pareto ---
fig, ax = plt.subplots(figsize=(12, 6))
cumsum_pct = (classe_counts.cumsum() / classe_counts.sum() * 100).values
ax.plot(range(1, len(cumsum_pct) + 1), cumsum_pct, linewidth=2, color='#2ecc71')
ax.axhline(y=80, color='#e74c3c', linestyle='--', linewidth=1.5, label='80% do volume')
ax.axvline(x=n_80pct, color='#e74c3c', linestyle='--', linewidth=1.5, alpha=0.5)
ax.fill_between(range(1, n_80pct + 1), cumsum_pct[:n_80pct], alpha=0.2, color='#2ecc71')
ax.set_xlabel('Número de Classes (ordenadas por frequência)')
ax.set_ylabel('% Cumulativa do Volume Total')
ax.set_title(f'Curva de Pareto - Distribuição de Classes\n{n_80pct} classes cobrem 80% do volume')
ax.legend()
ax.set_xlim(0, min(len(cumsum_pct), 100))
plt.savefig(f'{OUTPUT_DIR}/classe_pareto.png')
plt.close()
print(f"  ✓ Gráfico salvo: classe_pareto.png")

# ==============================================================================
# 5. ANÁLISE DO TAMANHO DO TEXTO
# ==============================================================================
print("\n📝 [5/7] Analisando tamanho dos textos (amostra)...")

df_sample['text_chars'] = df_sample[TEXT_COL].fillna('').str.len()
df_sample['text_words'] = df_sample[TEXT_COL].fillna('').str.split().str.len()

print(f"\n  Estatísticas de comprimento (caracteres):")
print(df_sample['text_chars'].describe().to_string())

print(f"\n  Estatísticas de comprimento (palavras):")
print(df_sample['text_words'].describe().to_string())

short_texts = (df_sample['text_chars'] < 200).sum()
long_texts = (df_sample['text_words'] > 5000).sum()
print(f"\n  ⚠️  Textos < 200 chars: {short_texts} ({short_texts/len(df_sample)*100:.1f}%)")
print(f"  ⚠️  Textos > 5000 palavras: {long_texts} ({long_texts/len(df_sample)*100:.1f}%)")

# Estimativa de tokens
df_sample['est_tokens'] = (df_sample['text_words'] / 0.75).astype(int)
fits_512 = (df_sample['est_tokens'] <= 512).sum()
fits_1024 = (df_sample['est_tokens'] <= 1024).sum()
print(f"\n  🔢 Estimativa de tokens (para Transformers):")
print(f"    Cabem em 512 tokens:  {fits_512:,} ({fits_512/len(df_sample)*100:.1f}%)")
print(f"    Cabem em 1024 tokens: {fits_1024:,} ({fits_1024/len(df_sample)*100:.1f}%)")

# --- GRÁFICO: Distribuição de Palavras ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].hist(df_sample['text_words'].clip(upper=5000), bins=50,
             color='#e67e22', edgecolor='white', alpha=0.8)
axes[0].axvline(df_sample['text_words'].median(), color='#e74c3c', linestyle='--',
                label=f'Mediana: {df_sample["text_words"].median():,.0f}')
axes[0].set_xlabel('Número de Palavras')
axes[0].set_ylabel('Frequência')
axes[0].set_title('Distribuição do Tamanho dos Textos (palavras)')
axes[0].legend()

axes[1].boxplot(df_sample['text_words'].clip(upper=5000), vert=True)
axes[1].set_ylabel('Número de Palavras')
axes[1].set_title('Box Plot - Tamanho dos Textos')

plt.suptitle(f'Análise de Tamanho do inteiro_teor ({DATASET_NAME})', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/classe_text_length.png')
plt.close()
print(f"  ✓ Gráfico salvo: classe_text_length.png")

# ==============================================================================
# 6. ANÁLISE CRUZADA: TAMANHO DO TEXTO POR CLASSE
# ==============================================================================
print("\n🔗 [6/7] Análise cruzada: tamanho do texto por classe...")

# Média de palavras por classe raiz
df_sample['classe_root'] = df_sample[TARGET_COL].fillna('').str.split(r'\s*->\s*').str[0].str.strip()
text_by_class = df_sample.groupby('classe_root')['text_words'].agg(['mean', 'median', 'count'])
text_by_class = text_by_class.sort_values('count', ascending=False)

print(f"\n  Tamanho médio do texto por classe raiz:")
print(text_by_class.to_string())

# --- GRÁFICO: Boxplot de palavras por classe raiz ---
top_roots = df_sample['classe_root'].value_counts().head(6).index
df_plot = df_sample[df_sample['classe_root'].isin(top_roots)].copy()
df_plot['text_words_clipped'] = df_plot['text_words'].clip(upper=5000)

fig, ax = plt.subplots(figsize=(14, 7))
sns.boxplot(data=df_plot, x='classe_root', y='text_words_clipped', ax=ax,
            palette='Set2', showfliers=False)
ax.set_xlabel('Classe Raiz')
ax.set_ylabel('Número de Palavras')
ax.set_title('Distribuição do Tamanho do Texto por Classe Raiz')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/classe_text_by_root.png')
plt.close()
print(f"  ✓ Gráfico salvo: classe_text_by_root.png")

# ==============================================================================
# 7. DETECÇÃO DE PROBLEMAS DE OCR / ENCODING
# ==============================================================================
print("\n🔧 [7/7] Verificando qualidade de encoding/OCR...")

mojibake_patterns = [r'Ã§', r'Ã£', r'Ã©', r'Ã¡', r'â€', r'Ã³', r'Ãº']
sample_texts = df_sample[TEXT_COL].dropna().sample(min(1000, len(df_sample)), random_state=42)

mojibake_count = 0
for text in sample_texts:
    for pattern in mojibake_patterns:
        if re.search(pattern, str(text)):
            mojibake_count += 1
            break

print(f"  Textos com possíveis mojibakes: {mojibake_count}/{len(sample_texts)} "
      f"({mojibake_count/len(sample_texts)*100:.1f}%)")

marker_count = sample_texts.str.contains(r'>{3,}\w+<{3,}', regex=True).sum()
print(f"  Textos com marcadores PJe (>>>>>inicio<<<<<): {marker_count}/{len(sample_texts)} "
      f"({marker_count/len(sample_texts)*100:.1f}%)")

# ==============================================================================
# RESUMO
# ==============================================================================
print("\n" + "=" * 70)
print("  📋 RESUMO DA EDA - CLASSES")
print("=" * 70)
print(f"""
  Dataset:             {DATASET_NAME}
  Total de linhas:     {total_lines:,}
  
  ALVO ({TARGET_COL}):
    Classes únicas:    {len(classe_counts)}
    Classes raiz:      {len(root_counts_full)}
    Classes folha:     {len(leaf_counts_full)}
    80% cobertos por:  {n_80pct} classes
    Classes raras (<10): {rare_classes}
    Hierarquia:        separador "->" com até {int(hierarchy_levels.max())} níveis
    
  TEXTO ({TEXT_COL}):
    Mediana palavras:  {df_sample['text_words'].median():,.0f}
    Média palavras:    {df_sample['text_words'].mean():,.0f}
    Cabe em 512 tok:   {fits_512/len(df_sample)*100:.1f}%
    
  QUALIDADE:
    Nulos texto:       {df_sample[TEXT_COL].isna().sum()}
    Mojibakes:         {mojibake_count/len(sample_texts)*100:.1f}%
    
  GRÁFICOS SALVOS em: {OUTPUT_DIR}
""")

print("  📌 RECOMENDAÇÕES:")
print("  1. Aplicar ftfy para corrigir mojibakes antes de qualquer modelagem")
print("  2. Remover marcadores PJe (>>>>>inicio<<<<<)")
if len(classe_counts) > 50:
    print(f"  3. DECISÃO HIERÁRQUICA: Classificar no nível raiz ({len(root_counts_full)} classes)")
    print(f"     ou no nível folha ({len(leaf_counts_full)} classes)? Começar pelo raiz é mais seguro.")
if rare_classes > 0:
    print(f"  4. CLASSES RARAS: {rare_classes} classes com < 10 amostras.")
    print(f"     Opções: agrupar em 'OUTROS', usar classificação hierárquica, ou Nível 3.")
if fits_512 / len(df_sample) < 0.5:
    print(f"  5. TEXTOS LONGOS: Apenas {fits_512/len(df_sample)*100:.1f}% cabem em 512 tokens.")
    print(f"     Para Nível 3 (Transformers), truncar ou usar chunking.")

print("\n✅ EDA concluída com sucesso!")
