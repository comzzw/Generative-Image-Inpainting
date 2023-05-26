from .dataset import InpaintingData

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def sample_data(loader): 
    while True:
        for batch in loader:
            yield batch


def create_loader(args): 
    dataset = InpaintingData(args)
    sampler = DistributedSampler(dataset) if args.distributed else None
    data_loader = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=(sampler is None), sampler=sampler, num_workers=args.num_workers, pin_memory=True)
    
    return data_loader, sample_data(data_loader)