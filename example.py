"""
Minimal example of constructing DiSCO.

DiSCO integrates with PyTorch/TorchTitan. Please follow TorchTitan’s
conventions for model definitions and parallel dimension setup:
https://github.com/pytorch/torchtitan
"""

import torch
from disco import Disco, create_disco_param_groups
from disco import create_disco_optimizer_kwargs_from_optimizer_config
from disco.dummy_parallel_dims import DummyParallelDims
from disco import build_init_fn


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_embeddings = torch.nn.Embedding(10, 256)
        self.block_1 = torch.nn.Linear(256, 512)
        self.block_2 = torch.nn.Linear(512, 1024)
        self.output = torch.nn.Linear(1024, 10)

        self.init_weights()

    def forward(self, x):
        return self.output(self.block_2(self.block_1(self.tok_embeddings(x))))

    def init_weights(self):
        embedding_init = build_init_fn("scion_normal_input")
        embedding_init(self.tok_embeddings.weight)

        output_init = build_init_fn("scion_normal_output")
        output_init(self.output.weight)

        linear_layer_init = build_init_fn("scaled_orthogonal")
        linear_layer_init(self.block_1.weight)
        linear_layer_init(self.block_2.weight)


if __name__ == "__main__":
    # Pseudo-usage: assumes you already have a model and parallel_dims
    # from TorchTitan. See TorchTitan’s docs for details.

    model = DummyModel().cuda()
    parallel_dims = DummyParallelDims()  # parallel_dims in torchtitan-like style

    # Example optimizer configuration (fill in real values as appropriate):
    optimizer_kwargs = {
        "parallel_dims": parallel_dims,
        "is_light": False,
        "weight_decay": 0.0,
        "lr": 1e-2,  # choose a learning rate
        "momentum": 0.1,
        "nesterov": False,
        "eps": 1e-20,
        "norm_factor": "spectral",
        "backend": "newtonschulz5",  # or "polar_express"/"svd"
        "backend_steps": 5,
        "extra_param_group_split_rules": [
            {
                "str_match": "tok_embeddings.weight",
                "lr": 1e-2,
                "norm_factor": "embed_sqrt",
                "backend": "identity",
            },  # for LLM's embedding
            {
                "str_match": "output.weight",
                "lr": 1e-2,
                "norm_factor": "unembed_sqrt",
                "backend": "identity",
            },  # for LLM's output linear layer
        ],
        "name_of_embedding": "tok_embeddings",  # this varible is important for tracking embedding's norms
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
    for norm_name, norm_value in tracked_norms.items():
        print(f"{norm_name}: {norm_value.item()}")
