import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
import torch.utils.checkpoint as checkpoint
import functools



import numpy as np

## not self attention, but cross attention, uses query, (key, value) to create relations between 2 sequences
class CrossAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob, initializer_fn):
        super(CrossAttention, self).__init__()
        self.num_attention_heads = num_attention_heads  # 2
        self.attention_head_size = hidden_size // num_attention_heads  # 512/2 = 256
        self.all_head_size = self.num_attention_heads * self.attention_head_size  # 2 * 256= 512

        # Initialize query, key, value 

        self.query = nn.Linear( hidden_size, self.all_head_size)  
        self.key = nn.Linear( hidden_size, self.all_head_size)
        self.value = nn.Linear( hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)
        self.dense = nn.Linear(self.all_head_size, hidden_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, layer_input, cond_embeddings, deterministic=True):
        #print(f"layer_input shape in CrossAttention: {layer_input.shape}")
        #print(f"cond_embeddings shape in CrossAttention: {cond_embeddings.shape}")

         # Apply linear layers
        mixed_query_layer = self.query(layer_input)
        mixed_key_layer = self.key(cond_embeddings)
        mixed_value_layer = self.value(cond_embeddings)

        #print(f"mixed_query_layer shape: {mixed_query_layer.shape}")
        #print(f"mixed_key_layer shape: {mixed_key_layer.shape}")
        #print(f"mixed_value_layer shape: {mixed_value_layer.shape}")

      
# Reshape for multi-head attention
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        
        #print(f"query_layer shape: {query_layer.shape}")
        #print(f"key_layer shape: {key_layer.shape}")
        #print(f"value_layer shape: {value_layer.shape}")
        

        logging.info("###########Inside of CrossAttention, calculating the hard stuff.##################")

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.reshape(*new_context_layer_shape)
        attention_output = self.dense(context_layer)
        attention_output = self.dropout(attention_output)

        return attention_output

## mapping the images embeddings from the CLIP models to the desired output shape
class MappingNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MappingNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.fc2 = nn.Linear(output_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x

# FFN with 2 linear layer and a ReLU that often happens in a transformer architecture.
class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, intermediate_size, dropout_prob, initializer_fn):
        super(FeedForwardNetwork, self).__init__()
        self.dense1 = nn.Linear(hidden_size, intermediate_size)
        self.dense2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.relu = nn.ReLU()

    def forward(self, x, deterministic=True):
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dropout(x) if not deterministic else x
        x = self.dense2(x)
        x = self.dropout(x) if not deterministic else x
        return x

    # asingle transformer layer that Combines cross-attention, feed-forward network, layer normalization, and dropout.
#Applies cross-attention to the input, followed by feed-forward processing and normalization.
class TransformerLayer(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, intermediate_size, hidden_dropout_prob, attention_probs_dropout_prob, initializer_fn):
        super(TransformerLayer, self).__init__()
        self.self_attention = CrossAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob, initializer_fn)
        self.feed_forward = FeedForwardNetwork(hidden_size, intermediate_size, hidden_dropout_prob, initializer_fn)
        self.layer_norm1 = nn.LayerNorm(hidden_size, eps=1e-12)
        self.layer_norm2 = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, cond_states, deterministic=True):
        print("crossss attention in transformerlayer")

        # Create a partial function that fixes the `deterministic` argument
        self_attention_partial = functools.partial(self.self_attention, deterministic=deterministic)
        attention_output = checkpoint.checkpoint(self_attention_partial, hidden_states, cond_states)

        attention_output = self.self_attention(hidden_states, cond_states, deterministic)
        attention_output = self.dropout(attention_output) if not deterministic else attention_output
        attention_output = self.layer_norm1(hidden_states + attention_output)


        # Use gradient checkpointing for feed-forward network
        #feed_forward_output = checkpoint.checkpoint(self.feed_forward, attention_output, deterministic)
        
        feed_forward_output = self.feed_forward(attention_output, deterministic)
        feed_forward_output = self.dropout(feed_forward_output) if not deterministic else feed_forward_output
        layer_output = self.layer_norm2(attention_output + feed_forward_output)

        return layer_output

# implements a masked language modeling, some tokens in the input are masked, and the model tries to predict these masked tokens
class MlmLayer(nn.Module):
    def __init__(self, hidden_size, vocab_size, initializer_fn):
        super(MlmLayer, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)  # Change here to use vocab_size
        self.bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, hidden_states, embedding_weights):
        hidden_states = self.dense(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = F.linear(hidden_states, embedding_weights)
        hidden_states += self.bias
        return hidden_states
    
    # convert tokens into embeddings
class Embed(nn.Module):
    def __init__(self, embedding_size, hidden_dropout_prob, vocab_size, max_position_embeddings, initializer_fn, hidden_size=None):
        super(Embed, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.initializer_fn = initializer_fn
        self.hidden_size = hidden_size

        self.word_embeddings = nn.Embedding(vocab_size, embedding_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, embedding_size)

        if hidden_size:
            self.embedding_hidden_mapping = nn.Linear(embedding_size, hidden_size)

        self.embeddings_ln = nn.LayerNorm(embedding_size if hidden_size is None else hidden_size, eps=1e-12)
        self.embeddings_dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_ids):
        batch_size, seq_length, *spatial_dims = input_ids.size()
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_length)

        # Flatten the input_ids for word embeddings
        input_ids = input_ids % self.vocab_size  # Ensure values are within the valid range
        word_embeddings = self.word_embeddings(input_ids.view(batch_size, seq_length, -1))
        word_embeddings = word_embeddings.view(batch_size, seq_length, *spatial_dims, -1)

        # Get position embeddings
        position_embeddings = self.position_embeddings(position_ids)

        # Add position embeddings to word embeddings
        embeddings = word_embeddings + position_embeddings.unsqueeze(2).unsqueeze(3).unsqueeze(4)

        #print(f"word_embeddings shape: {word_embeddings.shape}")
        #print(f"position_embeddings shape: {position_embeddings.shape}")
        #print(f"embeddings shape: {embeddings.shape}")

        if self.hidden_size:
            embeddings = self.embedding_hidden_mapping(embeddings)

        embeddings = self.embeddings_ln(embeddings)
        embeddings = self.embeddings_dropout(embeddings)

        return embeddings

class MaskedTransformer(nn.Module):
    def __init__(self, args):
        super(MaskedTransformer, self).__init__()
        self.hidden_size = args.hidden_dim
        self.num_attention_heads = args.num_heads
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.attention_probs_dropout_prob = args.dropout_rate
        self.hidden_dropout_prob = args.dropout_rate
        self.initializer_range = args.initializer_range

        self.embedding_layer = Embed(
            embedding_size=self.hidden_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            vocab_size=args.codebook_size,
            max_position_embeddings=args.max_position_embeddings,
            initializer_fn=truncated_normal(self.initializer_range),
            hidden_size=self.hidden_size
        )
        """
        self.cross_attention = CrossAttention(
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_attention_heads,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            initializer_fn=truncated_normal(self.initializer_range)
        )
        """

        self.layers = nn.ModuleList([
            TransformerLayer(
                intermediate_size=4 * self.hidden_size,
                hidden_size=self.hidden_size,
                hidden_dropout_prob=self.hidden_dropout_prob,
                num_attention_heads=self.num_attention_heads,
                attention_probs_dropout_prob=self.attention_probs_dropout_prob,
                initializer_fn=truncated_normal(self.initializer_range)
            ) for _ in range(args.num_blocks)
        ])

        self.mapping_net = MappingNetwork(input_dim=args.cond_emb_dim, output_dim=self.hidden_size)   # 512 to hidden_size

        self.mlm_layer = MlmLayer(
            hidden_size=self.hidden_size,
            vocab_size=args.codebook_size,  # This should be set to 512
            initializer_fn=truncated_normal(self.initializer_range)
        )

    def apply_masking(self, embeddings, mask_prob=0.15):
        """
        Apply random masking to the input embeddings.

        Args:
        - embeddings: Tensor of shape [batch_size, depth, height, width, num_tokens, embedding_dim]
        - mask_prob: Probability of masking each token

        Returns:
        - masked_embeddings: Tensor of the same shape with some tokens replaced by [MASK]
        - mask: Binary mask indicating which tokens were masked
        """
        batch_size, depth, height, width, num_tokens, embedding_dim = embeddings.shape
        
        # Flatten spatial dimensions and tokens for easier masking
        flat_embeddings = embeddings.view(batch_size, -1, embedding_dim)
        
        # Generate a random mask
        rand = torch.rand(flat_embeddings.size()[:-1], device=embeddings.device)
        mask = rand < mask_prob
        
        # Assume the last token in the vocabulary is the [MASK] token
        mask_token_id = embeddings.size(-1) - 1  # Using the last index as [MASK]
        
        # Create masked embeddings by replacing selected tokens with [MASK]
        masked_embeddings = flat_embeddings.clone()
        masked_embeddings[mask] = mask_token_id

        # Reshape back to the original dimensions
        masked_embeddings = masked_embeddings.view(batch_size, depth, height, width, num_tokens, embedding_dim)
        
        return masked_embeddings, mask


    def forward(self, shape_embeddings, cond_embeddings, deterministic=True):
        shape_embeddings = shape_embeddings.long()


        input_embeddings = self.embedding_layer(shape_embeddings)

        # Check the shape of input_embeddings
        #print(f"input_embeddings shape: {input_embeddings.shape}")

        # Apply masking
        masked_shape_embeddings, mask = self.apply_masking(input_embeddings)
        #print(f"masked_shape_embeddings shape: {masked_shape_embeddings.shape}")

        if cond_embeddings.dtype == torch.float16:
            cond_embeddings = cond_embeddings.float()

        cond_embeddings_mapped = self.mapping_net(cond_embeddings)      #512 to 128
        #print(f"cond_embeddings_mapped shape: {cond_embeddings_mapped.shape}")

        # Calculate the correct flattened shape
        batch_size, depth, height, width, codebook, hidden_size = masked_shape_embeddings.shape

        # Print the shapes before pooling
        #print(f"Before pooling - batch_size: {batch_size}, depth: {depth}, height: {height}, width: {width}, codebook: {codebook}, hidden_size: {hidden_size}")

        # Reshape for pooling
        masked_shape_embeddings = masked_shape_embeddings.reshape(batch_size * codebook, depth, height, width, hidden_size)



        # Check if the input tensor is suitable for pooling (i.e., it has spatial dimensions)
        if masked_shape_embeddings.shape[1] > 1 and masked_shape_embeddings.shape[2] > 1 and masked_shape_embeddings.shape[3] > 1:
            # Apply pooling only if the tensor represents 3D data
            pooled_embeddings = nn.functional.avg_pool3d(masked_shape_embeddings.permute(0, 4, 1, 2, 3), kernel_size=2, stride=2)
            pooled_embeddings = pooled_embeddings.permute(0, 2, 3, 4, 1).reshape(batch_size, codebook, 4, 4, 4, hidden_size)     

        else:
            # Skip pooling for token embeddings
            pooled_embeddings = masked_shape_embeddings.permute(0, 4, 1, 2, 3)

    
        # Apply pooling while maintaining the hidden size dimension
        #print(f"pooled_embeddings shape after pooling: {pooled_embeddings.shape}") 


        
        # Using maxpooling to reduce the size with a 2x2 size kernal. so this mean that the input shape embedding data has been reduced by half
        # from word_embeddings shape: torch.Size([8, 64, 8, 8, 8, 128]) to pooled_embeddings shape after restoring seq length: torch.Size([8, 64, 4, 4, 4, 128])
        # Restore sequence length dimension and hidden size dimension
        #print(f"pooled_embeddings shape after restoring seq length: {pooled_embeddings.shape}")

        # Check total number of elements
        #print(f"Total elements expected: {batch_size * codebook * 4 * 4 * 4 * hidden_size}")
        #print(f"Total elements in pooled_embeddings: {pooled_embeddings.numel()}")

        # Flatten the spatial dimensions of pooled_embeddings
        spatial_size = pooled_embeddings.shape[2] * pooled_embeddings.shape[3] * pooled_embeddings.shape[4]
        input_embeddings_flat = pooled_embeddings.reshape(batch_size, codebook * spatial_size, hidden_size)

        # Print shape after flattening
        #print(f"input_embeddings_flat shape: {input_embeddings_flat.shape}")

        # Ensure cond_embeddings_mapped matches the shape of input_embeddings_flat
        cond_embeddings_expanded = cond_embeddings_mapped.unsqueeze(1).expand(batch_size, codebook * spatial_size, hidden_size)
        #print(f"cond_embeddings_expanded shape: {cond_embeddings_expanded.shape}")


        
        layer_input = input_embeddings_flat
        for layer in self.layers:
            print(f"layer_input shape before attention: {layer_input.shape}")
            layer_output = layer(layer_input, cond_embeddings_expanded, deterministic)
            print(f"layer_output shape after attention: {layer_output.shape}")
            layer_input = layer_output
        logging.info("###########Getting output layer, after transformer.##################")
        
        

       
        """
        layer_input = input_embeddings_flat  # Move input to CPU
        layer_input_cpu =layer_input.to('cpu')
        for layer in self.layers:
            print(f"layer_input shape before attention: {layer_input.shape}")
            
            # Move the layer to CPU, if it is on GPU
            layer_cpu = layer.to('cpu')
            
            # Pass the data through the layer on CPU
            layer_output = layer_cpu(layer_input_cpu, cond_embeddings_expanded.to('cpu'), deterministic)
            
            print(f"layer_output shape after attention: {layer_output.shape}")
            
            # Move the result back to GPU for the next steps
            layer_output = layer_output.to('cuda')
            """
       

        # Continue with the next steps on GPU
        
        logits = self.mlm_layer(layer_output.transpose(1, 2).contiguous().reshape(batch_size, -1, self.hidden_size), self.embedding_layer.word_embeddings.weight)
        #print(f"logits shape after attention: {logits.shape}")

          # Merge the sequence length and hidden size dimensions for upsampling
        logits = logits.view(batch_size * codebook, codebook, 4, 4, 4)

        # Upsample to the target shape
        logits = F.interpolate(logits, size=(8, 8, 8), mode='trilinear', align_corners=True)

        # Reshape back to the original dimensions
        logits = logits.reshape(batch_size, codebook, codebook, 8, 8, 8).permute(0, 1, 3, 4, 5, 2)
        #print(f"logits shape after upsample: {logits.shape}")


        return logits, mask

def truncated_normal(stddev, dtype=torch.float32):
    def init_(tensor):
        return nn.init.trunc_normal_(tensor, mean=0, stddev=stddev, a=-2 * stddev, b=2 * stddev)
    return init_



def masked_token_prediction_loss(logits, target, mask):

    logging.info("###########calculating loss.##################")

    # Print the target tensor
    #print("Target tensor shape:", target.shape)

    # Print unique values
    unique_target_values = torch.unique(target)
    #print("Unique target values:", unique_target_values)
    #print("Unique target values shape:", unique_target_values.shape)


    

    # Logits have shape [batch_size, seq_length, hidden_size, 8, 8, 8]
    batch_size, seq_length, depth, height, width,hidden_size = logits.shape

    # Reshape logits to shape [batch_size * seq_length * depth * height * width, hidden_size]
    logits = logits.reshape(batch_size * seq_length * depth * height * width, hidden_size)

    # Flatten the target to match the logits shape
    target = target.reshape(batch_size * seq_length * depth * height * width)

    # Flatten the mask to match the logits and target shapes



    # Assuming the mask is still in its original shape, we should reduce its dimensions
    if mask.dim() > 1:  # Check if mask has more than one dimension
        mask = mask.reshape(-1)  # Flatten the mask to [N]
    #print(f"Mask shape after flatten: {mask.shape}")

    # Apply the mask to logits and target
    masked_logits = logits[mask.bool()]
    masked_target = target[mask.bool()]

    # Ensure that masked_logits and masked_target are aligned
    assert masked_logits.size(0) == masked_target.size(0), \
        "Masked logits and target must have the same number of elements."

    # Calculate cross-entropy loss only on masked elements
    loss = F.cross_entropy(masked_logits, masked_target.long(), reduction='mean')
    logging.info("###########done with loss.##################")

    return loss





