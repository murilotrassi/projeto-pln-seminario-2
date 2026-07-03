# projeto-pln-seminario-2

Contém os códigos dos detectores e explicadores de fake news, usando o Fake.BR Corpus.

O repositório tem cinco notebooks:

- **`Detector_Fake_News_BR.ipynb`** — versão base, que usa os metadados/features linguísticas já prontos, fornecidos junto com o Fake.BR Corpus.
- **`Detector_Fake_News_BR-extracao_pos.ipynb`** — versão com extração de features própria: as mesmas 21 features linguísticas são recalculadas a partir do texto completo da notícia, usando anotação morfossintática própria (spaCy), em vez de vir pronta dos metadados. Inclui também um *sanity check* comparando os valores extraídos com os valores originais e extraindo uma correlação.
- **`Detector_Fake_News_BR-validacao_externa.ipynb`** — parte do pipeline do `extracao_pos` (precisa da extração via spaCy, já que o corpus externo não vem com metadados prontos) e testa os modelos treinados no Fake.BR contra um corpus totalmente externo, o [FakeRecogna](https://github.com/Gabriel-Lino-Garcia/FakeRecogna) (baixado automaticamente na primeira execução), pra medir generalização fora do corpus de treino.
- **`Detector_Fake_News_BR-embeddings.ipynb`** — mesmo experimento no Fake.BR Corpus, mas com o Conteúdo representado por embeddings densos do **BERTimbau** (`neuralmind/bert-base-portuguese-cased`, 768 dimensões), um BERT treinado em português brasileiro, em vez das 21 features linguísticas manuais. A explicabilidade do Conteúdo usa LIME (`lime.lime_text`) em vez de SHAP, já que dimensões de embedding não têm leitura humana individualmente.
- **`Detector_Fake_News_BR-ablacao.ipynb`** — experimento de ablação que remove as *source dummies* (variáveis indicadoras de domínio da URL) do pipeline, investigando quanto do desempenho in-corpus e da queda de generalização no FakeRecogna vem desse atalho de domínio. Usa `is_real_author` para filtrar campos de autor mal preenchidos (ex: "Por G1", datas isoladas) antes de binarizar a feature. Inclui SHAP sobre o FakeRecogna para Conteúdo e Contexto.

## 1. Pré-requisitos

- **Python 3.11** (importante: `numpy==1.24.x`, usado pelo projeto, não tem wheel pré-compilada para Python 3.13+ e falha ao compilar do código-fonte. Use 3.11 mesmo que tenha uma versão mais nova instalada).
- A base **Fake.BR Corpus**: https://github.com/roneysco/Fake.br-Corpus
  - Baixe e coloque a pasta `Fake.br-Corpus-master` na raiz deste projeto (mesmo nível dos notebooks).
- O **FakeRecogna** (usado pelo `validacao_externa` e pelo `ablacao`) é baixado automaticamente pelo próprio notebook na primeira execução — não precisa baixar manualmente.
- O `Detector_Fake_News_BR-embeddings.ipynb` instala `sentence-transformers` (já pinado no `requirements.txt`), que traz `torch`+`transformers` como dependência transitiva (~1-2GB de instalação) e baixa os pesos do **BERTimbau** (`neuralmind/bert-base-portuguese-cased`, ~440MB) do Hugging Face na primeira execução — precisa de internet na primeira vez, depois fica em cache local.

## 2. Configurando o ambiente

Crie um ambiente virtual dedicado ao projeto (evita conflitos com outras instalações de Python na sua máquina e garante que todo o grupo rode com as mesmas versões de pacote):

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
```

> **Nota sobre o PyTorch:** o `requirements.txt` pina `torch==2.12.1+cu126` (build CUDA 12.6), necessário para o `Detector_Fake_News_BR-embeddings.ipynb` usar GPU. O `--extra-index-url` acima faz o pip consultar o servidor de wheels do PyTorch **além** do PyPI (não em vez dele), permitindo encontrar a build CUDA sem quebrar os outros pacotes. Se você **não tiver GPU NVIDIA**, instale o torch CPU antes e depois o restante:
> ```bash
> pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu
> pip install -r requirements.txt
> ```
> O notebook detecta o device automaticamente e cai para CPU se CUDA não estiver disponível.

Crie também as pastas onde cada notebook salva seus gráficos (não vêm versionadas no Git — cada um salva na sua própria subpasta para não sobrescrever os resultados do outro):

```bash
mkdir figures\figures_original figures\figures_extracao_pos figures\figures_validacao_externa figures\figures_embeddings figures\figures_ablacao
```

## 3. Registrando o kernel no Jupyter / VS Code

Para o notebook usar esse ambiente (e não a instalação global do Python), registre o venv como um kernel Jupyter:

```bash
python -m ipykernel install --user --name=pln-seminario-venv --display-name "Python 3.11 (pln-seminario .venv)"
```

Depois, em cada notebook, no seletor de kernel (canto superior direito), selecione **"Python 3.11 (pln-seminario .venv)"**.

## 4. Rodando os notebooks

### `Detector_Fake_News_BR.ipynb` (versão base)

Use **Run All** (executar todas as células em ordem). É necessário ter o Fake.BR Corpus na pasta correta (passo 1).

### `Detector_Fake_News_BR-extracao_pos.ipynb` (Extração de Features própria)

Use **Run All** também. A extração das features via spaCy roda sobre as ~7.200 notícias do corpus e leva alguns minutos (a primeira célula da seção "Extração de features morfossintáticas a partir texto (spaCy)" baixa o modelo `pt_core_news_sm` se ainda não estiver instalado, mas como já está no `requirements.txt`, normalmente é só um no-op rápido).

Depois da extração, há uma célula de **sanity check** que compara, feature a feature, os valores extraídos via spaCy com os valores originais dos metadados (correlação + médias). Esta é útil para validar se a extração está coerente antes de seguir para a modelagem.

### `Detector_Fake_News_BR-validacao_externa.ipynb` (Validação Externa — FakeRecogna)

Use **Run All**. Reexecuta o pipeline do `extracao_pos` (ingestão, extração spaCy, contexto, split/scale, ablação, treino dos 3 modelos) e depois baixa e avalia o FakeRecogna (~11.900 notícias, baixadas automaticamente do GitHub na primeira execução). A extração spaCy roda duas vezes (Fake.BR + FakeRecogna), então o tempo total é maior que o `extracao_pos` isolado.

### `Detector_Fake_News_BR-embeddings.ipynb` (Conteúdo via embeddings densos)

Use **Run All**. A primeira execução baixa o **BERTimbau** (`neuralmind/bert-base-portuguese-cased`) do Hugging Face (precisa de internet). Encodar as ~7.200 notícias do Fake.BR em CPU leva poucos minutos; os modelos com mais dimensões de entrada (SVM, Rede Neural) no passo de seleção via cross-validation podem ficar sensivelmente mais lentos do que nos outros notebooks. As explicações via `LimeTextExplainer` (Conteúdo e Conteúdo+Contexto) usam `num_samples=500` (reduzido do padrão 5000 do LIME por motivo de tempo de execução, já que cada amostra perturbada exige reencodar o texto no BERTimbau).

### `Detector_Fake_News_BR-ablacao.ipynb` (Ablação de Source Dummies)

Use **Run All**. O pipeline é idêntico ao `validacao_externa`, com duas diferenças: (1) as *source dummies* são removidas do conjunto de features, deixando o Contexto com apenas `author`; (2) a binarização de `author` usa `is_real_author`, que filtra campos mal preenchidos como "Por G1" ou datas isoladas. A pasta `figures/figures_ablacao/` é criada automaticamente pelo notebook. O tempo de execução é similar ao `validacao_externa`.

## 5. Observações

- O `.venv/`, os `.pdf` gerados em `figures/` e o `FakeRecogna_no_removal_words.xlsx` baixado automaticamente não são versionados (veja `.gitignore`): cada usuário deve montar o próprio ambiente e criar as pastas de figuras seguindo os passos acima.
- Se o `pip install -r requirements.txt` reclamar de conflito de versões, confirme que está usando Python 3.11 (não 3.12/3.13) — várias versões pinadas (`numpy`, `scipy`) dependem disso.
- O FakeRecogna não tem licença OSS explícita no GitHub; é usado nos notebooks com a mesma finalidade acadêmica do dataset original (publicado em paper peer-reviewed).
