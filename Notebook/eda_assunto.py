# ==============================================================================
# EDA - Análise Exploratória: Dataset de ASSUNTOS Processuais
# ==============================================================================
# Este script analisa o dataset de assuntos recorrentes para entender:
# 1. Estrutura e qualidade dos dados
# 2. Distribuição das classes (cauda longa)
# 3. Tamanho dos textos (inteiro_teor)
# 4. Dados nulos e anomalias
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
FILE_PATH = r'C:\Users\mlzengo\Documents\TJGO\II SEMESTRE\classe-assuno\data\raw\amostra_processos_assuntos_recorrentes_27022025.csv'
OUTPUT_DIR = r'C:\Users\mlzengo\Documents\TJGO\II SEMESTRE\classe-assuno\data\processed\eda_charts'
DATASET_NAME = 'Assuntos'
TARGET_COL = 'assunto'
TEXT_COL = 'inteiro_teor'
SEP = '#'
SAMPLE_SIZE = 10000  # Amostra inicial para exploração rápida

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
print("\n📂 [1/6] Carregando dados (amostra)...")

# Primeiro: inspeção rápida com amostra pequena
df_sample = pd.read_csv(FILE_PATH, sep=SEP, encoding='utf-8', nrows=SAMPLE_SIZE)

print(f"  ✓ Amostra carregada: {len(df_sample):,} linhas")
print(f"  Colunas: {df_sample.columns.tolist()}")
print(f"  Dtypes:\n{df_sample.dtypes.to_string()}")

# Contar linhas totais sem carregar tudo na memória
print("\n  Contando total de linhas no arquivo completo...")
total_lines = 0
with open(FILE_PATH, 'r', encoding='utf-8') as f:
    for _ in f:
        total_lines += 1
total_lines -= 1  # Desconta header
print(f"  ✓ Total de linhas no arquivo: {total_lines:,}")

# ==============================================================================
# 2. ANÁLISE DE VALORES NULOS E QUALIDADE
# ==============================================================================
print("\n🔍 [2/6] Análise de qualidade dos dados...")

# Verificar nulos na amostra
nulls = df_sample.isnull().sum()
nulls_pct = (df_sample.isnull().sum() / len(df_sample) * 100).round(2)
quality_df = pd.DataFrame({'nulos': nulls, 'pct': nulls_pct})
print(f"\n  Valores nulos (amostra de {SAMPLE_SIZE:,}):")
print(quality_df.to_string())

# Verificar duplicatas de numero_processo
duplicates = df_sample['numero_processo'].duplicated().sum()
print(f"\n  Processos duplicados na amostra: {duplicates}")

# Verificar textos vazios ou muito curtos
empty_text = (df_sample[TEXT_COL].isna() | (df_sample[TEXT_COL].str.len() < 50)).sum()
print(f"  Textos vazios ou < 50 chars: {empty_text} ({empty_text/len(df_sample)*100:.1f}%)")

# ==============================================================================
# 3. DISTRIBUIÇÃO DO ALVO (ASSUNTO) - USANDO ARQUIVO COMPLETO COM CHUNKS
# ==============================================================================
print("\n📊 [3/6] Contando distribuição de assuntos (arquivo completo, chunked)...")

assunto_counts = pd.Series(dtype='int64')
classe_counts = pd.Series(dtype='int64')
chunk_num = 0

for chunk in pd.read_csv(FILE_PATH, sep=SEP, encoding='utf-8', chunksize=10000,
                          usecols=[TARGET_COL, 'classe']):
    assunto_counts = assunto_counts.add(chunk[TARGET_COL].value_counts(), fill_value=0)
    classe_counts = classe_counts.add(chunk['classe'].value_counts(), fill_value=0)
    chunk_num += 1
    if chunk_num % 50 == 0:
        print(f"    ... processados {chunk_num * 10000:,} linhas")

assunto_counts = assunto_counts.astype(int).sort_values(ascending=False)
print(f"  ✓ Total de assuntos únicos: {len(assunto_counts)}")
print(f"  ✓ Total de classes únicas (coluna classe): {len(classe_counts)}")

# Top 20
print(f"\n  Top 20 assuntos mais frequentes:")
for i, (assunto, count) in enumerate(assunto_counts.head(20).items(), 1):
    pct = count / assunto_counts.sum() * 100
    print(f"    {i:2d}. {assunto[:70]:<70s} | {count:>7,} ({pct:.1f}%)")

# Cauda longa: quantos assuntos representam 80% do volume
cumsum = assunto_counts.cumsum() / assunto_counts.sum()
n_80pct = (cumsum <= 0.80).sum()
print(f"\n  📈 Regra 80/20: {n_80pct} assuntos cobrem 80% do volume "
      f"(de {len(assunto_counts)} totais)")

# Classes com poucos exemplos
rare_classes = (assunto_counts < 10).sum()
very_rare = (assunto_counts < 3).sum()
print(f"  ⚠️  Assuntos com < 10 amostras: {rare_classes}")
print(f"  ⚠️  Assuntos com < 3 amostras: {very_rare} (problemáticos para treino)")

# --- GRÁFICO: Top 20 Assuntos ---
fig, ax = plt.subplots(figsize=(14, 10))
top20 = assunto_counts.head(20)
colors = sns.color_palette("viridis", n_colors=20)
bars = ax.barh(range(len(top20)), top20.values, color=colors)
ax.set_yticks(range(len(top20)))
ax.set_yticklabels([label[:60] for label in top20.index], fontsize=10)
ax.invert_yaxis()
ax.set_xlabel('Frequência')
ax.set_title(f'Top 20 Assuntos Processuais Mais Frequentes\n(Total: {len(assunto_counts)} assuntos únicos | {assunto_counts.sum():,} processos)')

# Adicionar valores nas barras
for bar, val in zip(bars, top20.values):
    ax.text(bar.get_width() + assunto_counts.max() * 0.01, bar.get_y() + bar.get_height()/2,
            f'{val:,.0f}', va='center', fontsize=9)

plt.savefig(f'{OUTPUT_DIR}/assunto_top20.png')
plt.close()
print(f"  ✓ Gráfico salvo: assunto_top20.png")

# --- GRÁFICO: Distribuição Cumulativa (Pareto) ---
fig, ax = plt.subplots(figsize=(12, 6))
cumsum_pct = (assunto_counts.cumsum() / assunto_counts.sum() * 100).values
ax.plot(range(1, len(cumsum_pct) + 1), cumsum_pct, linewidth=2, color='#2ecc71')
ax.axhline(y=80, color='#e74c3c', linestyle='--', linewidth=1.5, label='80% do volume')
ax.axvline(x=n_80pct, color='#e74c3c', linestyle='--', linewidth=1.5, alpha=0.5)
ax.fill_between(range(1, n_80pct + 1), cumsum_pct[:n_80pct], alpha=0.2, color='#2ecc71')
ax.set_xlabel('Número de Assuntos (ordenados por frequência)')
ax.set_ylabel('% Cumulativa do Volume Total')
ax.set_title(f'Curva de Pareto - Distribuição de Assuntos\n{n_80pct} assuntos cobrem 80% do volume')
ax.legend()
ax.set_xlim(0, min(len(cumsum_pct), 200))
plt.savefig(f'{OUTPUT_DIR}/assunto_pareto.png')
plt.close()
print(f"  ✓ Gráfico salvo: assunto_pareto.png")

# ==============================================================================
# 4. ANÁLISE DO TAMANHO DO TEXTO (inteiro_teor)
# ==============================================================================
print("\n📝 [4/6] Analisando tamanho dos textos (amostra)...")

# Calcular métricas de tamanho na amostra
df_sample['text_chars'] = df_sample[TEXT_COL].fillna('').str.len()
df_sample['text_words'] = df_sample[TEXT_COL].fillna('').str.split().str.len()

print(f"\n  Estatísticas de comprimento (caracteres):")
print(df_sample['text_chars'].describe().to_string())

print(f"\n  Estatísticas de comprimento (palavras):")
print(df_sample['text_words'].describe().to_string())

# Textos muito curtos (potencial problema)
short_texts = (df_sample['text_chars'] < 200).sum()
long_texts = (df_sample['text_words'] > 5000).sum()
print(f"\n  ⚠️  Textos < 200 chars: {short_texts} ({short_texts/len(df_sample)*100:.1f}%)")
print(f"  ⚠️  Textos > 5000 palavras: {long_texts} ({long_texts/len(df_sample)*100:.1f}%)")

# Estimativa de tokens para Transformers (~0.75 words per token para PT-BR)
df_sample['est_tokens'] = (df_sample['text_words'] / 0.75).astype(int)
fits_512 = (df_sample['est_tokens'] <= 512).sum()
fits_1024 = (df_sample['est_tokens'] <= 1024).sum()
print(f"\n  🔢 Estimativa de tokens (para Transformers):")
print(f"    Cabem em 512 tokens:  {fits_512:,} ({fits_512/len(df_sample)*100:.1f}%)")
print(f"    Cabem em 1024 tokens: {fits_1024:,} ({fits_1024/len(df_sample)*100:.1f}%)")

# --- GRÁFICO: Distribuição de Palavras ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Histograma de palavras
axes[0].hist(df_sample['text_words'].clip(upper=5000), bins=50, color='#3498db', edgecolor='white', alpha=0.8)
axes[0].axvline(df_sample['text_words'].median(), color='#e74c3c', linestyle='--',
                label=f'Mediana: {df_sample["text_words"].median():,.0f}')
axes[0].set_xlabel('Número de Palavras')
axes[0].set_ylabel('Frequência')
axes[0].set_title('Distribuição do Tamanho dos Textos (palavras)')
axes[0].legend()

# Box plot
axes[1].boxplot(df_sample['text_words'].clip(upper=5000), vert=True)
axes[1].set_ylabel('Número de Palavras')
axes[1].set_title('Box Plot - Tamanho dos Textos')

plt.suptitle(f'Análise de Tamanho do inteiro_teor ({DATASET_NAME})', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/assunto_text_length.png')
plt.close()
print(f"  ✓ Gráfico salvo: assunto_text_length.png")

# ==============================================================================
# 5. DETECÇÃO DE PROBLEMAS DE OCR / ENCODING
# ==============================================================================
print("\n🔧 [5/6] Verificando qualidade de encoding/OCR...")

# Buscar padrões de mojibake comuns
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

# Verificar marcadores do PJe
marker_count = sample_texts.str.contains(r'>{3,}\w+<{3,}', regex=True).sum()
print(f"  Textos com marcadores PJe (>>>>>inicio<<<<<): {marker_count}/{len(sample_texts)} "
      f"({marker_count/len(sample_texts)*100:.1f}%)")

# Detecção de multi-label (@ como separador no assunto)
if TARGET_COL == 'assunto':
    multi_label = df_sample[TARGET_COL].fillna('').str.contains('@').sum()
    print(f"\n  📌 Assuntos com '@' (possível multi-label): {multi_label}/{len(df_sample)} "
          f"({multi_label/len(df_sample)*100:.1f}%)")

# ==============================================================================
# 6. RESUMO E RECOMENDAÇÕES
# ==============================================================================
print("\n" + "=" * 70)
print("  📋 RESUMO DA EDA - ASSUNTOS")
print("=" * 70)
print(f"""
  Dataset:            {DATASET_NAME}
  Total de linhas:    {total_lines:,}
  Colunas:            {df_sample.columns.tolist()}
  
  ALVO ({TARGET_COL}):
    Valores únicos:   {len(assunto_counts)}
    80% cobertos por: {n_80pct} assuntos
    Classes raras (<10): {rare_classes}
    
  TEXTO ({TEXT_COL}):
    Mediana palavras:  {df_sample['text_words'].median():,.0f}
    Média palavras:    {df_sample['text_words'].mean():,.0f}
    Cabe em 512 tok:   {fits_512/len(df_sample)*100:.1f}%
    
  QUALIDADE:
    Nulos texto:       {df_sample[TEXT_COL].isna().sum()}
    Mojibakes:         {mojibake_count/len(sample_texts)*100:.1f}%
    Multi-label (@):   {multi_label/len(df_sample)*100:.1f}%
    
  GRÁFICOS SALVOS em: {OUTPUT_DIR}
""")

print("  📌 RECOMENDAÇÕES:")
print("  1. Aplicar ftfy para corrigir mojibakes antes de qualquer modelagem")
print("  2. Remover marcadores PJe (>>>>>inicio<<<<<)")
if n_80pct < len(assunto_counts) * 0.3:
    print(f"  3. CAUDA LONGA FORTE: Considerar agrupar classes raras ou usar classificação hierárquica")
if multi_label > len(df_sample) * 0.05:
    print(f"  4. MULTI-LABEL detectado: {multi_label/len(df_sample)*100:.1f}% dos assuntos contêm '@'.")
    print(f"     Decisão necessária: tratar como multi-label ou usar apenas o primeiro assunto?")
if fits_512 / len(df_sample) < 0.5:
    print(f"  5. TEXTOS LONGOS: Apenas {fits_512/len(df_sample)*100:.1f}% cabem em 512 tokens.")
    print(f"     Para Transformers, será necessário truncar ou usar chunking.")

print("\n✅ EDA concluída com sucesso!")
