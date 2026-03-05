### TODO


Finish reading Masters Latent_z variation paper

Pull changes from main

Complete PCA

Decouple loss function from model

Understand Dope score


Create PCA plot 

### Journal

even earlier
- training script

earlier
- creating FromND / ToND methods
- plotting PCA method

tuesday 20th jan 2026
- noting the success of a 6D latent z, chose this as the hyper parameter upon which to continue work
- This choice should be subject to scrutiny!
- latent z of 1 giving interesting results (PCA with 1 component accounting for all variance)
- successfully did PCA on higher dim
- trying to create grid from PCA data then plot it

### Writing notes

Don’t just say: first I did this, then I did that. At each choice point, explain the alternatives and why you chose the alternative you did. Bonus points if you tie the reason to principles you’ve learned.



### Prompt for refactoring the creation of the mesh grid plots 

#file:cnn_multi_dim.ipynb 

Currently I have a lot of messy code for plotting Dope Score visualizations. I want to neaten this into a single protocol which has the following as inputs:

encoded dataset: shape(N x latent_dim)
mapping: latent_dim -> 2D
inverse_mapping: 2D -> latent_dim

This protocol should then create the plot with the dope score grid 