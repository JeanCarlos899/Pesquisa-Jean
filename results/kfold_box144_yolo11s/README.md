# K-fold box 144

Validacao cruzada por imagem. Cada fold usa um quinto das imagens como teste externo e separa uma validacao interna apenas dentro dos quatro quintos restantes.

- `kfold_folds.csv`: metricas por fold no teste externo, usando o limiar escolhido na validacao interna daquele fold.
- `kfold_resumo.csv`: media, desvio-padrao, minimo e maximo entre folds.
- `kfold_folds.md` e `kfold_resumo.md`: versoes rapidas para leitura.

Artefatos pesados de cada fold ficam nos `output_dir` registrados em `kfold_folds.csv`.
