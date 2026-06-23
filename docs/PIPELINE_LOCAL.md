# Pipeline local Jean

## Objetivo

Detectar todas as celulas nas 400 imagens CRIC, ignorando a classificacao
Bethesda durante o treino. Cada celula vira uma instancia da classe unica
`cell`.

## Por que testar tamanhos de bbox?

O CRIC Classification fornece coordenadas nucleares, nao caixas delimitadoras.
Logo, a caixa usada no treino e um alvo derivado. Um unico tamanho fixo, como
72 px, pode ser pequeno demais para algumas celulas ou grande demais em imagens
densas. Por isso, a pipeline inclui uma ablação:

```powershell
.\run.ps1 bbox-sweep
```

Ela treina uma versao leve para cada tamanho em `box_size_candidates` e compara
F1, precisao, revocacao, AP50 e erro medio de contagem na validacao. O melhor
tamanho deve orientar o treino principal e a discussao do artigo.

Na varredura inicial, `144 px` apresentou o melhor equilibrio operacional:
maior F1 em IoU 0,50 e menor erro medio de contagem. Por isso, o treino
principal usa `box_size_px = 144`.

## Etapas

Criar ambiente:

```powershell
.\setup.ps1
```

Verificar ambiente:

```powershell
.\run.ps1 check
```

Preparar dataset YOLO:

```powershell
.\run.ps1 prepare -Force
```

Rodar ablação de bbox:

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

Comparar YOLO11s/YOLO11m contra o YOLO11n principal sem sobrescrever o modelo
principal:

```powershell
.\run.ps1 model-s
.\run.ps1 model-m
```

Os resultados sao gravados em
`outputs_detection/metrics/model_compare_summary.csv`, enquanto os pesos ficam
em `outputs_detection/model_compare/yolo11s` e
`outputs_detection/model_compare/yolo11m`.

O teste com YOLO11m e opcional e mais arriscado em 6 GB de VRAM. Ele possui
fallbacks com `imgsz` e batch menores.

Gerar materiais do artigo:

```powershell
.\run.ps1 materials
```

## Artefatos principais

- `outputs_detection/metadata_cells_detection.csv`: anotacoes padronizadas.
- `outputs_detection/yolo_dataset/`: dataset YOLO de classe unica.
- `outputs_detection/checkpoints/best_cell_detector.pt`: melhor detector.
- `outputs_detection/metrics/avaliacao_val_limiares.csv`: grade de limiares na validacao.
- `outputs_detection/metrics/avaliacao_test_limiares.csv`: grade de limiares no teste.
- `outputs_detection/metrics/map_test.csv`: AP por IoU.
- `outputs_detection/metrics/contagem_por_imagem_teste.csv`: erro de contagem por imagem.
- `outputs_detection/metrics/intervalos_confianca_bootstrap.csv`: IC95 por reamostragem de imagens.
- `outputs_detection/metrics/bbox_sweep_summary.csv`: comparacao dos tamanhos de pseudo-bbox.

## Observacao metodologica

As metricas de deteccao sao calculadas contra pseudo-bounding boxes centradas
nos pontos nucleares. Isso deve ser descrito como limitacao e como escolha
metodologica validada por ablação, nao como ground truth manual de segmentacao
ou contorno celular.
