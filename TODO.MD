### TODO

- Score cards
    - Redo RMSD
    - CCC

- Create scoring function
    - run / var in
    - fetches 5 scores (RMSD mean, RMSD std, wasserstein distance, inversion ratios, CCC) aggregated across runs (from csvs)
    - returns these 5 scores
    - Additionally states if model passes checks (inversion ratio > 0, wasserstein distance high enough, CCC high enough, etc)
    - thresholds defineable by user
    - run these everything
    - plot table with 

- train different MLPs and compare them with UMAP and PCA?
    - How to decide mlp loss funcs?
    - Which models to do them on? (pick 1 high performing model for now? 2D baseline?)
    - Use same table based approach? Need to validate eval approach?
    - 


- Create inline version and train (with new obj function)
    - After evaluating MLP loss funcs


- Finish reading Masters Latent_z variation paper
- A* bs