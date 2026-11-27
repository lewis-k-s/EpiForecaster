# Training and evaluation run on one node

EpiForecaster training and evaluation always run in one process on one MN5 node, so the code has no `torch.distributed` reduction and the evaluation accumulators synchronize only between device and host. Do not add multi-node or DDP-aware designs until a model or dataset no longer fits one node; that change would need a reduction step in every evaluation accumulator and in the loss ledger.
