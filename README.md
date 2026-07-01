# Detecção e Contagem de Células Cervicais a partir de Anotações Pontuais

Jean Carlos Rodrigues Sousa  
Coorientador: João Antônio Leal de Miranda  
Orientador: Romuere Rodrigues Veloso e Silva

Repositorio do experimento de deteccao e contagem de celulas cervicais a partir
de anotacoes pontuais da base CRIC Cervix. A tarefa e modelada como deteccao de
objeto de classe unica (`cell`), convertendo cada coordenada nuclear
(`nucleus_x`, `nucleus_y`) em uma pseudo-bounding box.

## Ideia do trabalho

A base CRIC Cervix fornece pontos nucleares, mas nao fornece caixas delimitadoras
anatomicas. Este repositorio avalia como a construcao dessas pseudo-caixas afeta
localizacao, calibracao de limiar e erro de contagem por imagem. A configuracao
principal usa pseudo-caixas de `144 px` e compara escalas YOLO11n, YOLO11s e
YOLO11m.

## Estrutura

- `src/cervical_cell_detection/`: codigo da pipeline de download, preparo, treino,
  predicao, avaliacao e materiais do artigo.
- `configs/local_3060.json`: configuracao usada nos experimentos locais.
- `docs/PIPELINE_LOCAL.md`: roteiro detalhado para reproduzir o experimento.
- `results/article_materials/`: tabelas, metricas e figuras leves usadas no
  artigo.
- `LaTeX/sbc-final/`: fonte LaTeX do artigo final no template SBC.
- `LaTeX/eniac-original/`: versao longa/original preservada como historico.
- `cric_cervix/`: base original baixada localmente, ignorada pelo Git.
- `outputs_detection/`: datasets YOLO, checkpoints, predicoes e metricas
  geradas, ignorados pelo Git.

## Preparar ambiente

```powershell
.\setup.ps1
```

## Baixar a base CRIC Cervix

O download usa a API publica do Figshare. A colecao de imagens e o item de
classificacao sao baixados para `cric_cervix/`.

```powershell
.\run.ps1 download-data
```

Para validar o ambiente depois do download:

```powershell
.\run.ps1 check
.\run.ps1 status
```

## Rodar o experimento

```powershell
.\run.ps1 prepare -Force
.\run.ps1 bbox-sweep
.\run.ps1 train
.\run.ps1 eval-val
.\run.ps1 eval-test
.\run.ps1 model-compare -Models s,m
.\run.ps1 model-s -Test
.\run.ps1 materials-s
```

Fluxo completo da configuracao principal, sem a comparacao final de modelos:

```powershell
.\run.ps1 all
```

## Dados e artefatos versionados

O repositorio versiona codigo, configuracoes, documentacao, fontes LaTeX e
materiais leves do artigo. A base CRIC Cervix, checkpoints YOLO, datasets YOLO
gerados, logs, caches e PDFs compilados ficam fora do Git por tamanho ou por
serem artefatos reproduziveis.

## Fonte dos dados

CRIC Cervix Cell Classification, Figshare Collection:
<https://doi.org/10.6084/m9.figshare.c.4960286.v2>.
