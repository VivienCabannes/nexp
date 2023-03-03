## Quickstart
- [DONE] Train CIFAR on one GPU
- [DONE] Train CIFAR on several GPUs
- [DONE] Add logging 
- [DONE] Run it on SLURM

- Add checkpoints
    Think about the right checkpoint system to avoid confusion between different jobs
- Organize it well: handle job termination, and multinode jobs.
    Create a folder with the name of the job + the day of the job + an extra number if needed
    Eventually add modifications to be able to send emails

- Do it in SSL mode
- Look at classification results with SSL full graph

- Incorporate fast data loaders such as FFCV or NVIDIA-DALI
- Add profiler to find ideal batch-size, for each model
    For GPUs utilization, one can simply ssh into the different cluster machine and run command such as `nvidia-smi`.
- Deal with mixed-precision
- Use visualization like tensorboard

#### Deal with datasets
Vision: ImageNet, MNIST

## Projects

#### Active Learning through Similarity Graph
TODO

#### Memorization in Transformers
TODO

#### Laplacian spectral embedding
TODO
