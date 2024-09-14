


import os
import logging
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.helper import AverageMeter, save_checkpoint, create_dir, setup_logging, get_optimizer_model, get_device, set_seed
from dataset.shapenet_dataset import Shapes3dDataset, PointCloudField, PointsField, VoxelsField
from models.vqvae_model import VQVAE

"""
def get_dataloader(args, split="train"):
    
    if args.dataset_name == "Shapenet":
        pointcloud_field = PointCloudField("pointcloud.npz")
        points_field = PointsField("points.npz",unpackbits=True)
        voxel_fields = VoxelsField("model.binvox")

        fields = {}

        fields['pointcloud'] = pointcloud_field
        fields['points'] = points_field
        fields['voxels'] = voxel_fields

        if split == "train":
            dataset = Shapes3dDataset(args.dataset_path, fields, split=split,
                     categories=args.categories, no_except=True, transform=None, num_points=args.num_points,           num_sdf_points=args.num_sdf_points, sampling_type=args.sampling_type)

            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
            total_shapes = len(dataset)
        else:
            dataset = Shapes3dDataset(args.dataset_path, fields, split=split,
                     categories=args.categories, no_except=True, transform=None, num_points=args.num_points, num_sdf_points=args.test_num_sdf_points,  sampling_type=args.sampling_type)
            dataloader = DataLoader(dataset, batch_size=args.test_batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False)
            total_shapes = len(dataset)
        return dataloader, total_shapes 
  
    
    else:
        raise ValueError("Dataset name is not defined {}".format(args.dataset_name))

############################################# data loader #################################################
"""
def train_one_epoch(model, args, train_dataloader, optimizer, scheduler, loss_meter, epoch):
    model.train()
    for iteration, data in enumerate(train_dataloader):
        optimizer.zero_grad()

        data_input = data['voxels'].unsqueeze(1).type(torch.FloatTensor).to(args.device)  # Adding channel dimension

        vq_loss, x_recon, perplexity = model(data_input)
        recon_loss = F.mse_loss(x_recon, data_input)
        total_loss = recon_loss + vq_loss
        total_loss.backward()
        optimizer.step()
        loss_meter.update(total_loss.item(), data_input.size(0))

        if iteration % args.print_every == 0:
            logging.info("[Train] Epoch {}, Iteration {} Total Loss: {}, Recon Loss: {}, VQ Loss: {}, Perplexity: {}".format(
                epoch, iteration, loss_meter.avg, recon_loss.item(), vq_loss.item(), perplexity.item()))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_type", type=str, default='Voxel', help='Input representation')
    parser.add_argument("--output_type", type=str, default='Voxel', help='Output representation')
    parser.add_argument('--dataset_path', type=str, default='/home/lk/Clip-Forge/occupancy_networks/data/ShapeNet', help='Dataset path')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--print_every', type=int, default=50, help='Print interval')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--log_file', type=str, default='train.log', help='Log file')
    parser.add_argument('--log_level', type=str, default='info', help='Log level')
    parser.add_argument('--filemode', type=str, default='a', help='File mode for logging')
    parser.add_argument('--gpu', nargs='+', default=['0'], help='GPU ids to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    # Check for GPU availability
    if not torch.cuda.is_available():
        logging.warning("CUDA is not available. Using CPU instead.")
        args.device = 'cpu'

    create_dir(args.checkpoint_dir)
    setup_logging(args.log_file, args.log_level, args.filemode)
    set_seed(args.seed)
    
    gpu_string, gpu_array = get_device(args)
    if args.device == 'cuda':
        torch.cuda.set_device(gpu_array[0])
    
    fields = {
        'pointcloud': PointCloudField('pointcloud.npz'),
        'points': PointsField('points.npz', unpackbits=True),
        'voxels': VoxelsField('model.binvox')
    }
    dataset = Shapes3dDataset(args.dataset_path, fields, split='train')
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = VQVAE(in_channels=1, num_hiddens=128, num_residual_layers=2, num_residual_hiddens=32,
                  num_embeddings=512, embedding_dim=64, commitment_cost=0.25).to(args.device)
    optimizer = get_optimizer_model('Adam', model, lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    loss_meter = AverageMeter()

    for epoch in range(args.epochs):
        train_one_epoch(model, args, train_dataloader, optimizer, scheduler, loss_meter, epoch)
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            save_checkpoint(os.path.join(args.checkpoint_dir, f'epoch_{epoch + 1}.pt'), model, args, optimizer, scheduler, epoch)

if __name__ == "__main__":
    main()
