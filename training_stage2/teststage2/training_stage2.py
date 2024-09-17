import os
import os.path as osp
import logging

import argparse
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D

import torch
from torchvision import transforms
from torch.optim import lr_scheduler

import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from utils.helper import AverageMeter, save_checkpoint, create_dir, setup_logging, get_optimizer_model, get_device, set_seed
from utils import visualization
from dataset import shapenet_dataset
from vqvae_models import vqvae_model
from network_components import MaskedTransformer, masked_token_prediction_loss
import clip




######################import Clip models################################

def get_clip_model(args):
    if args.clip_model_type == "B-16":
        print("Bigger model is being used B-16")
        clip_model, clip_preprocess = clip.load("ViT-B/16", device=args.device)
        cond_emb_dim = 512
    elif args.clip_model_type == "RN50x16":
        print("Using the RN50x16 model")
        clip_model, clip_preprocess = clip.load("RN50x16", device=args.device)
        cond_emb_dim = 768
    else:
        clip_model, clip_preprocess = clip.load("ViT-B/32", device=args.device)
        cond_emb_dim = 512

    input_resolution = clip_model.visual.input_resolution

    print("cond_emb_dim:", cond_emb_dim)
    print("Input resolution:", input_resolution)
    #print("train_cond_embs length:", train_cond_embs_length)
    
    args.n_px = input_resolution
    args.cond_emb_dim = cond_emb_dim
    return args, clip_model



########################################data loader################################################

def get_dataloader(args, split="train", dataset_flag=False):
    
    dataset_name = args.dataset_name
                
    if dataset_name == "Shapenet":
        #pointcloud_field = shapenet_dataset.PointCloudField("pointcloud.npz")
        #points_field = shapenet_dataset.PointsField("points.npz", unpackbits=True)
        voxel_fields = shapenet_dataset.VoxelsField("model.binvox")
        
        if split == "train":
            image_field =  shapenet_dataset.ImagesField("img_choy2016", random_view=True, n_px=args.n_px)
        else:
            image_field =  shapenet_dataset.ImagesField("img_choy2016", random_view=False, n_px=args.n_px)
            

        fields = {}

        #fields['pointcloud'] = pointcloud_field
        #fields['points'] = points_field
        fields['voxels'] = voxel_fields
        fields['images'] = image_field
        
        def my_collate(batch):
            batch =  list(filter(lambda x : x is not None, batch))
            return torch.utils.data.dataloader.default_collate(batch)
        
        

        if split == "train":
            dataset = shapenet_dataset.Shapes3dDataset(args.dataset_path, fields, split=split,
                     categories=args.categories, no_except=True, transform=None, num_points=args.num_points)

            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True, collate_fn=my_collate)
            total_shapes = len(dataset)
        else:
            dataset = shapenet_dataset.Shapes3dDataset(args.dataset_path, fields, split=split,
                     categories=args.categories, no_except=True, transform=None, num_points=args.num_points)
            dataloader = DataLoader(dataset, batch_size=args.test_batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False, collate_fn=my_collate)
            total_shapes = len(dataset)


        if dataset_flag == True:  
            return dataloader, total_shapes, dataset
            
        return dataloader, total_shapes 
    
    else:
        raise ValueError("Dataset name is not defined {}".format(dataset_name))

######################################## get embeddings ########################################

def get_condition_embeddings(args, vqvae, clip_model, dataloader, times):# times is the number of views of a image
    vqvae.eval()
    clip_model.eval()
    shape_embeddings = []
    cond_embeddings = []
    with torch.no_grad():
        for i in range(times):
            for data in tqdm(dataloader):

                images = data['images'].type(torch.FloatTensor).to(args.device)
                voxels = data['voxels'].unsqueeze(1).type(torch.FloatTensor).to(args.device) # adding one channel dimention          


                encoded = vqvae._encoder(voxels)
                z = vqvae._pre_vq_conv(encoded)
                _, quantized, _, encoding_indices = vqvae._vq(z)
                shape_emb = encoding_indices.detach().cpu().numpy()

                # Reshape encoding_indices back to original dimensions
                original_shape = z.shape  # (batch_size, channels, depth, height, width)
                seq_size, codebook_size = shape_emb.shape   #codebook size is the num of embeddings which is 512
                batch_size, _, depth, height, width = original_shape
                encoding_indices = encoding_indices.view(batch_size, depth, height, width, codebook_size)

                
                image_features = clip_model.encode_image(images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                shape_embeddings.append(encoding_indices.cpu().numpy())
                cond_embeddings.append(image_features.detach().cpu().numpy())
            logging.info("Number of views done: {}/{}".format(i, times))

    shape_embeddings = np.concatenate(shape_embeddings, axis=0)
    cond_embeddings = np.concatenate(cond_embeddings, axis=0)
    print(f"shape_embeddings shape: {shape_embeddings.shape}")

    return shape_embeddings, cond_embeddings



#################################Train one epoch################################
def train_one_epoch(args, transformer, dataloader, optimizer):
    transformer.train()
    total_loss = 0

    for batch  in tqdm(dataloader):
        optimizer.zero_grad()
        
        shape_embs, cond_embs = batch
        shape_embs = shape_embs.type(torch.FloatTensor).to(args.device)
        cond_embs = cond_embs.type(torch.FloatTensor).to(args.device)

    
        # Mapping,Cross Attention and Transformer are in the transformer class
        logits, mask = transformer(shape_embs, cond_embs)  # forward pass
        loss = masked_token_prediction_loss(logits, shape_embs, mask)  # calculate the loss

        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

"""
# test val
def val_one_epoch(args, transformer, dataloader, optimizer):
    transformer.eval()
    total_loss = 0
    for batch in tqdm(dataloader):
        optimizer.zero_grad()

        shape_embeddings, cond_embeddings = batch
        shape_embeddings = shape_embeddings.type(torch.FloatTensor).to(args.device)
        cond_embeddings = cond_embeddings.type(torch.FloatTensor).to(args.device)

            
            # Mapping,Cross Attention and Transformer are in the transformer class
        logits, mask = transformer(shape_embeddings, cond_embeddings)  # forward pass
        loss = masked_token_prediction_loss(logits, shape_embeddings, mask)  # calculate the loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()



    return total_loss / len(dataloader)
"""
#this is the real val 

def val_one_epoch(args, transformer, dataloader):
    transformer.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader):

            shape_embeddings, cond_embeddings = batch
            shape_embeddings = shape_embeddings.type(torch.FloatTensor).to(args.device)
            cond_embeddings = cond_embeddings.type(torch.FloatTensor).to(args.device)

            
            # Mapping,Cross Attention and Transformer are in the transformer class
            logits, mask = transformer(shape_embeddings, cond_embeddings)  # forward pass
            loss = masked_token_prediction_loss(logits, shape_embeddings, mask)  # calculate the loss

            total_loss += loss.item()


    
    return total_loss / len(dataloader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='Shapenet')
    parser.add_argument('--dataset_path', type=str, default='/home/lk/Clip-Forge/occupancy_networks/data/ShapeNet', help='Dataset path')
    parser.add_argument('--categories',   nargs='+', default=None, metavar='N')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--num_points', type=int, default=2048)
    parser.add_argument('--num_workers', type=int, default=1)
    #parser.add_argument('--emb_dim', type=int, default=64) # 256
    parser.add_argument('--codebook_size', type=int, default=512)
    parser.add_argument('--cond_emb_dim', type=int, default=512)
    parser.add_argument('--hidden_dim', type=int, default=128)  #256    
    parser.add_argument('--num_heads', type=int, default=2) # 8
    parser.add_argument('--num_blocks', type=int, default=2) # 8
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--max_position_embeddings', type=int, default=512)
    parser.add_argument('--initializer_range', type=float, default=0.02)
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--vqvae_checkpoint', type=str, default='/home/lk/vqvae_3d_v2/checkpoints/epoch_300.pt')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--gpu', nargs='+', default="7", help='GPU ids to use')
    parser.add_argument('--clip_model_type', type=str, default='ViT-B/32', help='CLIP model type: ViT-B/32, ViT-B/16, or RN50x16')
    parser.add_argument("--n_px",  type=int, default=224, help='Resolution of the image')
    parser.add_argument('--num_views', type=int, default=1)
    parser.add_argument('--mask_prob', type=float, default=0.15, help='Probability of masking a token')
    parser.add_argument('--log_file', type=str, default='train.log', help='Log file')
    parser.add_argument('--log_level', type=str, default='info', help='Log level')
    parser.add_argument('--filemode', type=str, default='a', help='File mode for logging')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')


    args = parser.parse_args()

    """
{'04256520': {'id': '04256520', 'name': 'sofa,couch,lounge'}, '02691156': {'id': '02691156', 'name': 'airplane,aeroplane,plane'}, '03636649': {'id': '03636649', 'name': 'lamp'}, 
'04401088': {'id': '04401088', 'name': 'telephone,phone,telephone set'}, '04530566': {'id': '04530566', 'name': 'vessel,watercraft'}, '03691459': {'id': '03691459', 'name': 
'loudspeaker,speaker,speaker unit,loudspeaker system,speaker system'}, '03001627': {'id': '03001627', 'name': 'chair'}, '02933112': {'id': '02933112', 'name': 'cabinet'}, 
'04379243': {'id': '04379243', 'name': 'table'}, '03211117': {'id': '03211117', 'name': 'display,video display'}, 
'02958343': {'id': '02958343', 'name': 'car,auto,automobile,machine,motorcar'}, '02828884': {'id': '02828884', 'name': 'bench'}, '04090263': {'id': '04090263', 'name': 'rifle'}

nohup python training_stage2.py --num_views 1 --gpu 0 --checkpoint_dir checkpoint/View1 --log_file auto_logging_view1.log --categories 02958343  02691156 > view1.log 2>&1 &

nohup python training_stage2.py --num_views 5 --gpu 1 --checkpoint_dir checkpoint/View5 --log_file auto_logging_view5.log --categories 02958343  02691156  > view5.log 2>&1 &

nohup python training_stage2.py --num_views 1 --gpu 1 --checkpoint_dir checkpoint/View1_2 --log_file auto_logging_view1_2.log --categories 02958343  02691156  > view1_2.log 2>&1 &


            car, airplane

            bench, car, rifle, airplan, watercraft
            02828884 02958343 04090263 02691156 04530566

            
                nohup python script.py > script.log 2>&1 &

    """

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
    

    # Initialize the process group
    
     # Load CLIP model
    print("loading CLIP model.............")
    logging.info("###########loading CLIP model.##################")

    args, clip_model = get_clip_model(args)
    print("loading CLIP model complet.............")
    logging.info("###########loading CLIP model complet##################")


    print("loading data .............")
    logging.info("############loading data#################")

    train_dataloader, total_shapes  = get_dataloader(args, split="train")
    args.total_shapes = total_shapes
    logging.info("Train Dataset size: {}".format(total_shapes))

    val_dataloader, total_shapes_val  = get_dataloader(args, split="val")



    logging.info("Test Dataset size: {}".format(total_shapes_val))
    logging.info("#############################")

    print("dataload complet .............")

    logging.info("#############load VQVAE################")
    print("load vqvae..................................")

    vqvae = vqvae_model.VQVAE(in_channels=1, num_hiddens=128, num_residual_layers=2, num_residual_hiddens=32,
                  num_embeddings=512, embedding_dim=64, commitment_cost=0.25).to(args.device)
    checkpoint = torch.load(args.vqvae_checkpoint, map_location=args.device)
    vqvae.load_state_dict(checkpoint['model'])
    vqvae.eval()

    print("initialize mapping network, masked transformer .............")


 # Use DataParallel to use multiple GPUs

    # Initialize the transformer model
    transformer = MaskedTransformer(args).to(args.device)

    


     # Optimizer and Scheduler
    optimizer = torch.optim.Adam(transformer.parameters(), lr=args.lr)
    
    # we added this scheduler
    scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min',       # Because we're minimizing the loss
            factor=0.5,       # Reduce the learning rate by half
            patience=10,      # Wait for 10 epochs without improvement
            min_lr=1e-6,      # Minimum learning rate to prevent it from going too low
        )

 # Compute condition embeddings
    logging.info("Computing train embeddings...")
    train_shape_embeddings, train_cond_embeddings = get_condition_embeddings(args, vqvae, clip_model, train_dataloader, times=args.num_views)
    logging.info("train Embedding Shape {}, train Condition Embedding {}".format(train_shape_embeddings.shape, train_cond_embeddings.shape))
    train_dataset_new = torch.utils.data.TensorDataset(torch.from_numpy(train_shape_embeddings), torch.from_numpy(train_cond_embeddings))
    train_dataloader_new = DataLoader(train_dataset_new, batch_size=args.batch_size, num_workers=args.num_workers,shuffle=True)
                # shuffle=True


    logging.info("Computing validation embeddings...")
    val_shape_embeddings, val_cond_embeddings = get_condition_embeddings(args, vqvae, clip_model, val_dataloader, times=args.num_views)
    logging.info("Val Embedding Shape {}, Val Condition Embedding {}".format(val_shape_embeddings.shape, val_cond_embeddings.shape))
    val_dataset_new = torch.utils.data.TensorDataset(torch.from_numpy(val_shape_embeddings), torch.from_numpy(val_cond_embeddings))
    val_dataloader_new = DataLoader(val_dataset_new, batch_size=args.test_batch_size, num_workers=args.num_workers, shuffle=False)
    

    logging.info("#############################")

    """ # Training loop
    best_loss = float('inf')
    for epoch in range(args.epochs):

        logging.info(f"Epoch {epoch + 1}/{args.epochs}")
        train_loss = train_one_epoch(args, transformer, train_dataloader_new, optimizer)
        
        val_loss = val_one_epoch(args, transformer, val_dataloader_new, optimizer)

        logging.info(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        
        # Save best model
        #if (epoch + 1) % 10 == 0:
            #save_checkpoint(os.path.join(args.checkpoint_dir, f'epoch_{epoch + 1}.pt'), args, optimizer, scheduler, epoch)
        
        if val_loss < best_loss:
            best_loss = val_loss
            checkpoint_path = os.path.join(args.checkpoint_dir, f"best_transformer_epoch_{epoch + 1}.pt")
            logging.info("Saving Model........{}".format(checkpoint_path))

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': transformer.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args': args
            }, checkpoint_path) """



    best_loss = float('inf')
    for epoch in range(args.epochs):

        logging.info(f"Epoch {epoch + 1}/{args.epochs}")
        train_loss = train_one_epoch(args, transformer, train_dataloader_new, optimizer)
        
        val_loss = val_one_epoch(args, transformer, val_dataloader_new)

        logging.info(f" Training Loss: {train_loss:.4f}")
        logging.info(f" Validation Loss: {val_loss:.4f}")

        
        # Save best model
        #if (epoch + 1) % 10 == 0:
            #save_checkpoint(os.path.join(args.checkpoint_dir, f'epoch_{epoch + 1}.pt'), args, optimizer, scheduler, epoch)
        
        if val_loss < best_loss:
            best_loss = val_loss
            checkpoint_path = os.path.join(args.checkpoint_dir, f"best_transformer_epoch_{epoch + 1}.pt")
            logging.info("Saving Model........{}".format(checkpoint_path))

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': transformer.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args': args
            }, checkpoint_path)

        # Step the scheduler
        scheduler.step()




if __name__ == "__main__":
    main()





    """

    nohup python script.py > script.log 2>&1 &

    check the training status:

    tail -f training_stage2.log

    bg, kill %1


    
    """