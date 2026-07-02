# Deteccao e Contagem de Celulas Cervicais com Validacao Cruzada

Jean Carlos Rodrigues Sousa  
Coorientador: Joao Antonio Leal de Miranda  
Orientador: Romuere Rodrigues Veloso e Silva

Repositorio do experimento de deteccao e contagem de celulas cervicais a partir
de anotacoes pontuais da base CRIC Cervix. A tarefa e modelada como deteccao de
objeto de classe unica (`cell`), convertendo cada coordenada nuclear em uma
pseudo-bounding box quadrada.

## Protocolo atual

O protocolo confirmatorio usa validacao cruzada 5-fold por imagem:

- as 400 imagens entram no particionamento;
- em cada fold, 80 imagens formam o teste externo;
- as 320 imagens restantes sao separadas em 280 para treino e 40 para validacao
  interna;
- a validacao interna escolhe `best.pt` e o limiar operacional;
- o teste externo daquele fold e avaliado uma unica vez;
- os resultados do artigo devem ser reportados como media e desvio entre folds.

A configuracao principal usa pseudo-caixa de `144 px`, YOLO11s, ate 55 epocas e
paciencia 12 em uma RTX 3060 Laptop com 6 GB de VRAM.

## Estrutura

- `src/cervical_cell_detection/`: pipeline de download, preparo, treino,
  predicao, avaliacao e agregacao k-fold.
- `configs/local_3060.json`: configuracao local do protocolo k-fold.
- `docs/PIPELINE_LOCAL.md`: roteiro para reproduzir o experimento atual.
- `results/kfold_box144_yolo11s/`: tabelas finais do k-fold, geradas apos a
  execucao.
- `outputs_detection/kfold/`: datasets, checkpoints, predicoes e metricas por
  fold, ignorados pelo Git.
- `legacy/fixed_split/`: artigo, tabelas e apresentacao do protocolo antigo com
  split fixo, preservados apenas como historico.
- `outputs_detection/legacy_fixed_split/`: artefatos pesados locais das rodadas
  antigas, ignorados pelo Git.

## Preparar ambiente

```powershell
.\setup.ps1
```

## Baixar a base CRIC Cervix

```powershell
.\run.ps1 download-data
```

Para validar o ambiente:

```powershell
.\run.ps1 check
.\run.ps1 status
```

## Rodar o k-fold

Conferir as particoes antes de treinar:

```powershell
.\run.ps1 kfold -DryRun
```

Rodar o protocolo principal:

```powershell
.\run.ps1 kfold
```

Opcao mais economica, caso seja necessario reduzir tempo de relogio:

```powershell
.\run.ps1 kfold -Epochs 45 -Patience 10
```

## Saidas principais

```text
results/kfold_box144_yolo11s/
  kfold_folds.csv
  kfold_resumo.csv
  kfold_folds.md
  kfold_resumo.md
```

`kfold_folds.csv` contem as metricas por fold no teste externo. `kfold_resumo.csv`
contem media, desvio-padrao, minimo e maximo entre folds.

## Fonte dos dados

CRIC Cervix Cell Classification, Figshare Collection:
<https://doi.org/10.6084/m9.figshare.c.4960286.v2>.
