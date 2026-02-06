import os

import torch
import torch.distributed as dist


def main() -> None:
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)

    tensor = torch.tensor([1.0], device='cuda')
    dist.all_reduce(tensor)

    if local_rank == 0:
        print(f'NCCL check passed. Sum: {tensor.item()}')

    dist.destroy_process_group()


if __name__ == '__main__':
    main()
