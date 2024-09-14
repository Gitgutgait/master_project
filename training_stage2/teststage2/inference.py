import torch
import argparse
from vqvae_models import vqvae_model
from network_components import MaskedTransformer
import clip
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Load pre-trained CLIP models based on training_stage2.py settings
def get_clip_model(args):
    if args.clip_model_type == "B-16":
        print("Using CLIP ViT-B/16 model")
        clip_model, clip_preprocess = clip.load("ViT-B/16", device=args.device)
        args.cond_emb_dim = 512
    elif args.clip_model_type == "RN50x16":
        print("Using CLIP RN50x16 model")
        clip_model, clip_preprocess = clip.load("RN50x16", device=args.device)
        args.cond_emb_dim = 768
    else:
        clip_model, clip_preprocess = clip.load("ViT-B/32", device=args.device)
        args.cond_emb_dim = 512

    return clip_model, clip_preprocess

# Load the VQ-VAE and MaskedTransformer from checkpoints
def load_models(args):
    # Load pre-trained VQ-VAE
    vqvae = vqvae_model.VQVAE(in_channels=1, num_hiddens=128, num_residual_layers=2, 
                              num_residual_hiddens=32, num_embeddings=512, embedding_dim=64, 
                              commitment_cost=0.25).to(args.device)
    checkpoint_vqvae = torch.load(args.vqvae_checkpoint, map_location=args.device)
    vqvae.load_state_dict(checkpoint_vqvae['model'])
    vqvae.eval()

    # Load pre-trained MaskedTransformer
    transformer = MaskedTransformer(args).to(args.device)
    checkpoint_transformer = torch.load(args.transformer_checkpoint, map_location=args.device)
    transformer.load_state_dict(checkpoint_transformer['model_state_dict'])
    transformer.eval()

    return vqvae, transformer

# Extract features from sketches using CLIP
def extract_clip_features(args, sketch_path, clip_model, preprocess):
    sketch_image = Image.open(sketch_path)
    sketch_preprocessed = preprocess(sketch_image).unsqueeze(0).to(args.device)
    
    with torch.no_grad():
        clip_features = clip_model.encode_image(sketch_preprocessed)
        clip_features = clip_features / clip_features.norm(dim=-1, keepdim=True)
    
    return clip_features

# Initialize fully masked shape embeddings
def initialize_masked_sequence(args, vqvae, batch_size, seq_length):

    num_embeddings = 512  # Assuming the VQ layer is named '_vq'
    # Create a fully masked sequence of shape tokens (assumed last token is the mask token)
    masked_sequence = torch.full((batch_size, seq_length), fill_value=num_embeddings - 1).to(args.device)
    return masked_sequence

# Iterative decoding using the MaskedTransformer model
def iterative_decoding(args, transformer, masked_sequence, clip_features, num_iterations=5):
    for i in range(num_iterations):
        logits, mask = transformer(masked_sequence, clip_features)
        predicted_tokens = torch.argmax(logits, dim=-1)
        masked_sequence[mask.bool()] = predicted_tokens[mask.bool()]
    return masked_sequence



# Reconstruct the final 3D shape using VQ-VAE's decoder
def reconstruct_3d_shape(vqvae, final_sequence):
    with torch.no_grad():
        quantized = vqvae._vq_embedding(final_sequence)
        decoded_shape = vqvae.decode(quantized)
    return decoded_shape

# Visualize the 3D shape using matplotlib
def visualize_3d_shape(shape):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(shape.squeeze().cpu().numpy(), edgecolor='k')
    plt.show()

# Main inference function
def inference(args):
    # Load models
    vqvae, transformer = load_models(args)

    # Load CLIP model and preprocess
    clip_model, preprocess = get_clip_model(args)

    # Extract features from the input sketch
    clip_features = extract_clip_features(args, args.sketch_path, clip_model, preprocess)

    # Initialize a fully masked sequence of shape tokens
    masked_sequence = initialize_masked_sequence(args, vqvae, batch_size=1, seq_length=512)

    # Perform iterative decoding with the transformer
    final_sequence = iterative_decoding(args, transformer, masked_sequence, clip_features)

    # Reconstruct the 3D shape from the final sequence of tokens
    reconstructed_3d_shape = reconstruct_3d_shape(vqvae, final_sequence)

    # Visualize the reconstructed 3D shape
    visualize_3d_shape(reconstructed_3d_shape)

# Command-line arguments for inference 

    # Path to the input sketch
def parse_args():
    parser = argparse.ArgumentParser(description="Sketch-to-3D Inference")
    parser.add_argument('--sketch_path', type=str, default='/home/lk/airplane.png', help='Path to the input sketch image')
    parser.add_argument('--vqvae_checkpoint', type=str, default='/home/lk/vqvae_3d_v2/checkpoints/epoch_300.pt')
    parser.add_argument('--transformer_checkpoint', type=str, default='/home/lk/teststage2/checkpoint/View1/best_transformer.pt', help='Path to the MaskedTransformer checkpoint')
    parser.add_argument('--clip_model_type', type=str, default='ViT-B/32', help='CLIP model type: ViT-B/32, ViT-B/16, or RN50x16')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for inference')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference')
    parser.add_argument('--num_iterations', type=int, default=12, help='Number of decoding iterations')

    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension of the transformer')
    parser.add_argument('--num_heads', type=int, default=2, help='Number of attention heads')
    parser.add_argument('--num_blocks', type=int, default=2, help='Number of transformer blocks')
    parser.add_argument('--codebook_size', type=int, default=512, help='VQ-VAE codebook size')
    parser.add_argument('--cond_emb_dim', type=int, default=512, help='Dimension of CLIP condition embeddings')
    parser.add_argument('--max_position_embeddings', type=int, default=512, help='Maximum position embeddings')
    parser.add_argument('--initializer_range', type=float, default=0.02, help='Initializer range for weights')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout rate')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    inference(args)
