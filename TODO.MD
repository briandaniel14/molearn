### TODO

- Score cards
    - save rest of inversion ratios
    - wasserstein distance
    - RMSD csvs

- Create scoring function
    - run / var in
    - fetches 5 scores (RMSD mean, RMSD std, wasserstein distance, inversion ratios, CCC) aggregated across runs (from csvs)
    - returns these 5 scores
    - Additionally states if model passes checks (inversion ratio > 0, wasserstein distance high enough, CCC high enough, etc)
    - thresholds defineable by user
    - run these everything

- train different MLPs and compare them with UMAP and PCA?

- Create inline version and train (with new obj function)
- Finish reading Masters Latent_z variation paper
- A* bs