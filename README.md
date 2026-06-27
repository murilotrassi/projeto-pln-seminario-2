# projeto-pln-seminario-2

Contém os códigos dos detectores e explicadores de fake news, usando o Fake.BR Corpus.

O repositório tem dois notebooks:

- **`Detector_Fake_News_BR.ipynb`** — versão base, que usa os metadados/features linguísticas já prontos, fornecidos junto com o Fake.BR Corpus.
- **`Detector_Fake_News_BR-extracao_pos.ipynb`** — versão com extração de features própria: as mesmas 21 features linguísticas são recalculadas a partir do texto completo da notícia, usando anotação morfossintática própria (spaCy), em vez de vir pronta dos metadados. Inclui também um *sanity check* comparando os valores extraídos com os valores originais e extraindo uma correlação.

## 1. Pré-requisitos

- **Python 3.11** (importante: `numpy==1.24.x`, usado pelo projeto, não tem wheel pré-compilada para Python 3.13+ e falha ao compilar do código-fonte. Use 3.11 mesmo que tenha uma versão mais nova instalada).
- A base **Fake.BR Corpus**: https://github.com/roneysco/Fake.br-Corpus
  - Baixe e coloque a pasta `Fake.br-Corpus-master` na raiz deste projeto (mesmo nível dos notebooks).

## 2. Configurando o ambiente

Crie um ambiente virtual dedicado ao projeto (evita conflitos com outras instalações de Python na sua máquina e garante que todo o grupo rode com as mesmas versões de pacote):

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Crie também as pastas onde cada notebook salva seus gráficos (não vêm versionadas no Git — cada um salva na sua própria subpasta para não sobrescrever os resultados do outro):

```bash
mkdir figures\figures_original figures\figures_extracao_pos
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

## 5. Observações

- O `.venv/` e os `.pdf` gerados em `figures/` não são versionados (veja `.gitignore`): cada usuário deve montar o próprio ambiente e criar as pastas de figuras seguindo os passos acima.
- Se o `pip install -r requirements.txt` reclamar de conflito de versões, confirme que está usando Python 3.11 (não 3.12/3.13) — várias versões pinadas (`numpy`, `scipy`) dependem disso.
