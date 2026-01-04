import random
import torch

def set_seed(seed: int, rank: int = 0):
    seed = seed + rank
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)