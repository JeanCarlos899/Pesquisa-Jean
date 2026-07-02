# Pipeline local

## Objetivo

Detectar todas as celulas nas 400 imagens da CRIC Cervix usando anotacoes
pontuais como supervisao espacial. Cada ponto nuclear e convertido em uma
pseudo-caixa quadrada de 144 px e a tarefa e tratada como deteccao de classe
unica (`cell`).

## Desenho experimental

O protocolo atual usa validacao cruzada 5-fold por imagem. Todas as 400 imagens
entram no particionamento. Em cada fold:

- 80 imagens formam o teste externo;
- as 320 imagens restantes sao repartidas em 280 de treino e 40 de validacao
  interna;
- a validacao interna e usada para early stopping, escolha de `best.pt` e escolha
  do limiar operacional por maior F1 em IoU 0,50;
- o teste externo e avaliado uma unica vez com o checkpoint e o limiar escolhidos
  na validacao interna;
- nenhuma imagem aparece simultaneamente em treino, validacao interna e teste
  externo.

Os folds sao balanceados de forma gulosa pela quantidade de celulas por imagem,
reduzindo diferencas de carga celular entre os testes externos. Como a CRIC nao
fornece identificador de paciente, a independencia e controlada no nivel de
imagem, que e o maior controle suportado pelos metadados disponiveis.

## Dados

A base nao e versionada no Git. Para reconstruir a pasta local:

```powershell
.\run.ps1 download-data
```

Estrutura esperada:

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

## Execucao

Criar ambiente:

```powershell
.\setup.ps1
```

Verificar GPU e dependencias:

```powershell
.\run.ps1 check
```

Conferir as particoes sem treinar:

```powershell
.\run.ps1 kfold -DryRun
```

Rodar o protocolo principal:

```powershell
.\run.ps1 kfold
```

Opcao mais curta para reduzir tempo de relogio:

```powershell
.\run.ps1 kfold -Epochs 45 -Patience 10
```

## Artefatos atuais

Resultados leves para o artigo:

```text
results/kfold_box144_yolo11s/
  kfold_folds.csv
  kfold_resumo.csv
  kfold_folds.md
  kfold_resumo.md
```

Artefatos pesados por fold:

```text
outputs_detection/kfold/yolo11s_box144/fold_01/
...
outputs_detection/kfold/yolo11s_box144/fold_05/
```

Cada pasta de fold contem:

- `yolo_dataset/`: imagens e labels repartidos em `train`, `val` e `test`;
- `runs/`: historico de treino Ultralytics;
- `checkpoints/best_cell_detector.pt`: melhor checkpoint do fold;
- `predicoes_val.csv` e `predicoes_test.csv`;
- `metrics/avaliacao_val_limiares.csv`: grade de limiares da validacao interna;
- `metrics/limiar_operacional.json`: limiar selecionado na validacao interna;
- `metrics/avaliacao_test_limiares.csv`: grade do teste externo para auditoria;
- `metrics/contagem_por_imagem_test_fold.csv`: contagem por imagem no teste
  externo usando o limiar selecionado.

## Legado

As rodadas antigas com split fixo e comparacao de modelos foram movidas para:

```text
legacy/fixed_split/
outputs_detection/legacy_fixed_split/
```

Elas ficam preservadas como historico, mas nao fazem parte do protocolo
confirmatorio atual.
