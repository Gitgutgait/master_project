# Import necessary libraries
import torch
import numpy as np
from torch.utils.data import DataLoader
from skimage import measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

# Import custom modules (assuming they are in the same directory)
from models.vqvae_model import VQVAE
from dataset.shapenet_dataset import Shapes3dDataset, PointCloudField, PointsField, VoxelsField

# Load the model from a checkpoint
def load_model_from_checkpoint(checkpoint_path, device='cuda:1'):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args = checkpoint['args']
    
    # Reconstruct the model with the same architecture
    model = VQVAE(
        in_channels=1, 
        num_hiddens=128, 
        num_residual_layers=2, 
        num_residual_hiddens=32,
        num_embeddings=512, 
        embedding_dim=64, 
        commitment_cost=0.25
    ).to(device)
    
    model.load_state_dict(checkpoint['model'])
    model.eval()  # Set model to evaluation mode
    
    return model, args

# Prepare the test dataloader
def prepare_test_dataloader(dataset_path, batch_size=32):
    fields = {
        'pointcloud': PointCloudField('pointcloud.npz'),
        'points': PointsField('points.npz', unpackbits=True),
        'voxels': VoxelsField('model.binvox')
    }
    dataset = Shapes3dDataset(dataset_path, fields, split='test')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader

# Run inference on the test dataset and calculate metrics
def run_inference_and_evaluate(model, dataloader, device='cuda:1'):
    model.eval()  # Ensure the model is in evaluation mode
    results = []
    
    with torch.no_grad():  # Disable gradient calculation for inference
        for data in dataloader:
            data_input = data['voxels'].unsqueeze(1).type(torch.FloatTensor).to(device)
            _, reconstructed_output, _ = model(data_input)
            
            reconstructed_output_np = reconstructed_output.squeeze().cpu().numpy()
            original_np = data['voxels'].cpu().numpy()
            
            for i in range(len(original_np)):
                original = original_np[i]
                reconstructed = reconstructed_output_np[i]
                
                # Calculate MSE
                mse = np.mean((original - reconstructed) ** 2)
                
                # Calculate IoU
                intersection = np.logical_and(original > 0.5, reconstructed > 0.5)
                union = np.logical_or(original > 0.5, reconstructed > 0.5)
                iou = np.sum(intersection) / np.sum(union)
                
                # Calculate SSIM
                ssim_value = measure.compare_ssim(original, reconstructed, data_range=reconstructed.max() - reconstructed.min())
                
                # Store the results in a dictionary
                results.append({
                    'mse': mse,
                    'iou': iou,
                    'ssim': ssim_value
                })
    
    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate averages
    avg_mse = results_df['mse'].mean()
    avg_iou = results_df['iou'].mean()
    avg_ssim = results_df['ssim'].mean()
    
    return results_df, avg_mse, avg_iou, avg_ssim

# Example usage
checkpoint_path = 'checkpoints/epoch_270.pt'
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
model, args = load_model_from_checkpoint(checkpoint_path, device)

# Adjust the batch size for inference
inference_batch_size = 64  # Change this to a larger batch size if needed
test_dataloader = prepare_test_dataloader(args.dataset_path, inference_batch_size)

# Run inference and calculate metrics
results_df, avg_mse, avg_iou, avg_ssim = run_inference_and_evaluate(model, test_dataloader, device)

# Save results to CSV
results_df.to_csv('reconstruction_results.csv', index=False)

# Print the average metrics
print(f"Average MSE: {avg_mse}")
print(f"Average IoU: {avg_iou}")
print(f"Average SSIM: {avg_ssim}")

#nohup python train.py > custom_output1.log 2>&1 &

