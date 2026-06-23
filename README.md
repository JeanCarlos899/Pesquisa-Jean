# Pesquisa Jean - CRIC Cell Detection

Pipeline local para deteccao de todas as celulas em imagens cervicais da base
CRIC, sem classificacao Bethesda. A tarefa e modelada como deteccao de objeto
de classe unica (`cell`) a partir dos pontos nucleares `nucleus_x` e
`nucleus_y`.

## Ideia central do artigo

O diferencial da pesquisa e tratar explicitamente um problema comum em bases de
citologia: a base fornece pontos de celula, mas nao bounding boxes manuais. A
pipeline converte esses pontos em pseudo-bounding boxes, testa tamanhos
alternativos de alvo e reporta como essa escolha afeta deteccao e contagem por
imagem. Assim, o artigo nao fica restrito a "treinei um YOLO"; ele valida a
propria definicao operacional do alvo de deteccao.

A configuracao principal esta ajustada para pseudo-bbox de `144 px`, escolhida
pela varredura inicial por apresentar o melhor equilibrio entre F1, precisao,
revocacao e erro de contagem.

## Rodar

Criar ambiente local:

```powershell
.\setup.ps1
```

```powershell
.\run.ps1 check
.\run.ps1 prepare -Force
.\run.ps1 bbox-sweep
.\run.ps1 train
.\run.ps1 eval-val
.\run.ps1 eval-test
.\run.ps1 materials
```

Comparar capacidade do modelo mantendo bbox `144 px`:

```powershell
.\run.ps1 model-s
.\run.ps1 model-m
```

Ou rodar S e M na mesma etapa:

```powershell
.\run.ps1 model-compare -Models s,m
```

Fluxo completo, usando a configuracao principal:

```powershell
.\run.ps1 all
```

Monitorar GPU:

```powershell
.\run.ps1 gpu
```

## Pastas

- `cric_cervix/`: dataset original preservado.
- `src/jean_pipeline/`: codigo da pipeline.
- `configs/local_3060.json`: configuracao local.
- `outputs_detection/`: dataset YOLO, checkpoints, predicoes e metricas.
- `materiais_artigo/`: tabelas, figuras e CSVs prontos para escrita.
- `artigo/`: template LaTeX SBC/ENIAC limpo para o paper.
- `archive/notebooks/`: notebook antigo preservado como historico.
