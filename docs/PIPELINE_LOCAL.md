# Pipeline local

## Objetivo

Detectar todas as celulas nas 400 imagens da CRIC Cervix, ignorando a
classificacao Bethesda durante o treino. Cada celula vira uma instancia da
classe unica `cell`.

## Dados

A base nao e versionada no Git. Para reconstruir a pasta local:

```powershell
.\run.ps1 download-data
```

O comando usa:

- Colecao Figshare CRIC Cervix Cell Classification:
  <https://doi.org/10.6084/m9.figshare.c.4960286.v2>
- Item de classificacao:
  <https://doi.org/10.6084/m9.figshare.12233156.v2>

Estrutura esperada apos o download:

```text
cric_cervix/
  classification/
    classifications.csv
    classifications.json
    README.md
  images/
    cric_image_001_*.png
    ...
    cric_image_400_*.png
```

## Por que testar tamanhos de bbox?

O CRIC Cervix fornece coordenadas nucleares, nao caixas delimitadoras. Logo, a
caixa usada no treino e um alvo derivado. Um unico tamanho fixo pode ser pequeno
demais para algumas celulas ou grande demais em imagens densas. Por isso, a
pipeline inclui uma ablacao:

```powershell
.\run.ps1 bbox-sweep
```

Ela treina uma versao leve para cada tamanho em `box_size_candidates` e compara
F1, precisao, revocacao, AP50 e erro medio de contagem na validacao. A
configuracao principal usa `box_size_px = 144`, escolhida como compromisso
operacional no experimento.

## Etapas

Criar ambiente:

```powershell
.\setup.ps1
```

Baixar dados:

```powershell
.\run.ps1 download-data
```

Verificar ambiente:

```powershell
.\run.ps1 check
```

Preparar dataset YOLO:

```powershell
.\run.ps1 prepare -Force
```

Rodar ablacao de bbox:

```powershell
.\run.ps1 bbox-sweep
```

Treinar detector principal:

```powershell
.\run.ps1 train
```

Avaliar validacao e teste:

```powershell
.\run.ps1 eval-val
.\run.ps1 eval-test
```

Comparar YOLO11s/YOLO11m contra a configuracao principal:

```powershell
.\run.ps1 model-s
.\run.ps1 model-m
```

Ou, em uma chamada:

```powershell
.\run.ps1 model-compare -Models s,m
```

Avaliacao de teste e materiais do artigo final com o YOLO11s:

```powershell
.\run.ps1 model-s -Test
.\run.ps1 materials-s
```

## Artefatos principais

- `outputs_detection/metadata_cells_detection.csv`: anotacoes padronizadas.
- `outputs_detection/yolo_dataset/`: dataset YOLO de classe unica.
- `outputs_detection/checkpoints/best_cell_detector.pt`: melhor detector.
- `outputs_detection/metrics/avaliacao_val_limiares.csv`: grade de limiares na
  validacao.
- `outputs_detection/metrics/avaliacao_test_limiares.csv`: grade de limiares no
  teste.
- `outputs_detection/metrics/map_test.csv`: AP por IoU.
- `outputs_detection/metrics/contagem_por_imagem_teste.csv`: erro de contagem
  por imagem.
- `outputs_detection/metrics/intervalos_confianca_bootstrap.csv`: IC95 por
  reamostragem de imagens.
- `outputs_detection/metrics/bbox_sweep_summary.csv`: comparacao dos tamanhos
  de pseudo-bbox.

## Observacao metodologica

As metricas de deteccao sao calculadas contra pseudo-bounding boxes centradas
nos pontos nucleares. Isso deve ser descrito como limitacao e como escolha
metodologica validada por ablacao, nao como ground truth manual de segmentacao
ou contorno celular.
