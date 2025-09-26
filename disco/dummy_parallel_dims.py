import os
import tempfile
import torch
from torch.distributed.device_mesh import init_device_mesh


class DummyParallelDims:
    def __init__(self, world_mesh=None):

        if world_mesh is None:
            if torch.distributed.is_available():
                if not torch.distributed.is_initialized():
                    backend = "nccl" if torch.cuda.is_available() else "gloo"
                    init_kwargs = {
                        "backend": backend,
                        "world_size": 1,
                        "rank": 0,
                    }
                    if os.environ.get("MASTER_ADDR", None) is None:
                        os.environ["MASTER_ADDR"] = "127.0.0.1"
                    if os.environ.get("MASTER_PORT", None) is None:
                        os.environ["MASTER_PORT"] = "29500"

                    torch.distributed.init_process_group(**init_kwargs)

                world_size = torch.distributed.get_world_size()
            else:
                world_size = 1

            device_type = "cuda" if torch.cuda.is_available() else "cpu"
            world_mesh = init_device_mesh(
                device_type,
                mesh_shape=(world_size,),
                mesh_dim_names=("dp_replicate",),
            )
            world_mesh._flatten(mesh_dim_name="dp_cp")

        self.world_mesh = world_mesh

        self.dp_replicate_enabled = True
        self.fsdp_enabled = False
        self.ep_enabled = False
        self.tp_enabled = False
