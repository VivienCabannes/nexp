## Quickstart

#### experiments to be run
- Implement a sweep function to find the best hyperparameters
    Mainly should be a script that generates many bash scripts and lauch them all.
- Do it in SSL mode
- Look at classification results with SSL full graph

#### Improving efficiency
- Add profiler to find ideal batch-size, for each model
    For GPUs utilization, one can simply ssh into the different cluster machine and run command such as `nvidia-smi`.
- Incorporate fast data loaders such as FFCV or NVIDIA-DALI

#### Visualization tools
- Use visualization like tensorboard or wandb for easy visualization accross runs.
- Keep track of training and testing loss during training.

#### Nice features to add
- Deal with multinodes.
- Handle job termination.
- Deal with mixed-precision

#### Deal with datasets
Vision: ImageNet, MNIST

## Projects

#### Active Learning through Similarity Graph
TODO

#### Memorization in Transformers
TODO

#### Laplacian spectral embedding
TODO
