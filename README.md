# 💃Disco🕺: 🪩 Distributed Spectral Conditioned Optimizer for Muon/Scion 🪩

# Overview
- Disco is a distributed Spectral Conditioned optimizer. (Works for DDP/FSDP/TP/EP/CP/PP etc.)
- It supports `DTensor`-friendly semi-orthogonal initialization in [`orthognal_init.py`](./disco/orthognal_init.py).
- The core optimizer is implemented in [`disco.py`](./disco/disco.py). Lmo in Spectral Conditioned's framework lives in [`abstract_disco.py`](./disco/abstract_disco.py).
- It can track the norms of the parameters and the updates in [`norm_helper.py`](./disco/norm_helper.py). 
    - one can easily to extend these code to have Spectral Clip which forces weights live on Stiefel manifold in distributed manner.
- It integrates with TorchTitan’s model and parallel‑dimension abstractions. Please follow [TorchTitan’s](https://github.com/pytorch/torchtitan) conventions for model definitions and world‑mesh/parallel‑dimension.



# Quick Start
- The optimizer expects a standard PyTorch `model` and a TorchTitan parallel‑dimension description (e.g., `world_mesh`, `dp/fsdp/tp` flags). For an end‑to‑end example of model and parallel setup, please refer to TorchTitan’s documentation.
    - Alternatively, you can use a dummy parallel dimension setup which assume you are running on DDP.

- Minimal usage sketch (see [`example.py`](./example.py) for a runnable script):

  ```python
  import torch
  from disco import Disco, create_disco_param_groups
  from disco.dummy_parallel_dims import DummyParallelDims # if you are not using TorchTitan

  # Substitute your own model and TorchTitan parallel-dimension setup here.
  model = YourModel(...).cuda()
  parallel_dims = DummyParallelDims()  # placeholder for real TorchTitan dims

  optimizer_kwargs = {
      "parallel_dims": parallel_dims,   # from TorchTitan
      "is_light": False,
      "weight_decay": 0.0,
      "lr": 1e-2,
      "momentum": 0.1,
      "nesterov": False,
      "eps": 1e-20,
      "norm_factor": "spectral",
      "backend": "newtonschulz5",      # or: polar_express, svd, identity
      "backend_steps": 5,
      "extra_param_group_split_rules": [
          # token embedding
          {
              "str_match": "tok_embeddings.weight",
              "lr": 1e-2,
              "norm_factor": "embed_sqrt",
              "backend": "identity"
          },
          # final output
          {
              "str_match": "output.weight",
              "lr": 1e-2,
              "norm_factor": "unembed_sqrt",
              "backend": "identity"
          },
      ],
      "name_of_embedding": "tok_embeddings",
  }

  # For `torchtitan` users, you can parse the optimizer_config to get the optimizer_kwargs
  # optimizer_kwargs = create_disco_optimizer_kwargs_from_optimizer_config(optimizer_config, parallel_dims)
    

  params, cleaned = create_disco_param_groups(model, optimizer_kwargs)
  opt = Disco(params, **cleaned)

  loss = torch.nn.CrossEntropyLoss()(
      model(torch.randint(0, 10, (1, 10)).cuda()),
      torch.randint(0, 10, (1, 10)).cuda(),
  )
  loss.backward()

  opt.calculate_norm_at_next_step()
  opt.step()

  tracked_norms = opt.get_norms_at_current_step()
  for name, tensor in tracked_norms.items():
      print(f"{name}: {tensor.item()}")
  ```

- Call `opt.calculate_norm_at_next_step()` before `opt.step()` to queue norm tracking for the upcoming step, and retrieve the results afterward via `opt.get_norms_at_current_step()`.



# Related Works
If you want to get to know what is Spectral Condition, and why it is useful, these are minimal papers to read:
- Spectral Condition [A Spectral Condition for Feature Learning](https://arxiv.org/abs/2310.17813)
- Stepest descent on "norm" perspective in [Scalable Optimization in the Modular Norm
](https://arxiv.org/abs/2405.14813) and [Training Deep Learning Models with Norm-Constrained LMOs](https://arxiv.org/abs/2502.07529)


For people who are interested in kernels of NS5, [Dion](https://github.com/microsoft/dion) implements the triton kernels for the Newton-Schulz iteration.

<!-- 
# Citations
If you find this work useful, please cite:
```bibtex
@misc{wang2025disco,
    title={Disco: Distributed Spectral Conditioned Optimizer for Muon/Scion},
    author={},
    year={2025},
}
``` -->