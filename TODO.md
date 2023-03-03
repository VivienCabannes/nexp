## Quickstart

- Add checkpoints
    Think about the right checkpoint system to avoid confusion between different jobs.
    implement an easy way to start from last checkpoints --checkpoint arguments.

- Train on CIFAR with multinodes.

- Handle job termination.

- Implement a sweep function to find the best hyperparameters
    Mainly should be a script that generates many bash scripts and lauch them all.

- Keep track of training and testing loss during training.

- Add profiler to find ideal batch-size, for each model
    For GPUs utilization, one can simply ssh into the different cluster machine and run command such as `nvidia-smi`.

- Do it in SSL mode

- Look at classification results with SSL full graph

- Incorporate fast data loaders such as FFCV or NVIDIA-DALI

- Deal with mixed-precision

- Use visualization like tensorboard

- Isolate slurm launcher to improve readability



#### Deal with datasets
Vision: ImageNet, MNIST

## Projects

#### Active Learning through Similarity Graph
TODO

#### Memorization in Transformers
TODO

#### Laplacian spectral embedding
TODO
