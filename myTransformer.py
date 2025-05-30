
import torch
from torch import nn
import torch.nn.functional as F

"""basic self-attention operation"""
batch_size = 2 #number of sequences
sequence_length = 4 #number of tokens in each sequence
embedding_dim = 8 #dimension of each token embedding

#random input tensor x with shape [2, 4, 8]
x = torch.randn(batch_size, sequence_length, embedding_dim)


#compute raw attention scores which are the dot products between tokens
#output shape should be [2,4,4]
raw_weights = torch.bmm(x, x.transpose(1, 2))


#turn in porbabilities so they sum to 1 with softmax
weights = F.softmax(raw_weights, dim=2)


#each output vector is a weighted average of all input vectors in the sequence
#results in a batch of output matrices Y is size (2,4,8) whose rows are weighted averages of the input vectors X
y = torch.bmm(weights, x)

print("Input x shape:", x.shape)
print("Attention weights shape:", weights.shape)
print("Output y shape:", y.shape)

#2 matrix multiplications + 1 softmax = basic self-attention operation




"""multihead self-attention"""
#allows the model to focus on different aspects of the sentence in parallel
#k=input embedding size, heads=number of attention heads
class SelfAttention(nn.Module):
    def __init__(self, k, heads=4):
        #define a self-attention layer with k-dimensional input embeddings and 4 attention heads.
        super().__init__()
        assert k % heads == 0, "Embedding dimension must be divisible by number of heads"
        self.k, self.heads = k, heads #save the input parameters for later use in forward pass
        #define 3 layers to project the input embeddings into queries, keys, and values
        #eaach is of shape (batch_size, sequence_length, k)
        self.toqueries = nn.Linear(k, k, bias=False)
        self.tokeys    = nn.Linear(k, k, bias=False)
        self.tovalues  = nn.Linear(k, k, bias=False)
        #after multi-head attention is done, we combine all the heads back together with this linear projection
        #back to og dimension k
        self.unifyheads = nn.Linear(k, k)


    #x is the input tensor of shape (batch_size, sequence_length, k)
    def forward(self, x, mask=False):
        #extract the batch size, sequence length, and embedding dimension from the input tensor
        b, t, k = x.size()
        #number of heads
        h = self.heads
        #dimension of each head
        s = k // h
        #compute queries, keys, and values for all heads
        #reshape from (b, t, k) to (b, t, h, s)
        #splits the embedding vector across h heads
        queries = self.toqueries(x).view(b, t, h, s)
        keys    = self.tokeys(x).view(b, t, h, s)
        values  = self.tovalues(x).view(b, t, h, s)
        #fold heads into the batch dimension
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, s)
        keys    = keys.transpose(1, 2).contiguous().view(b * h, t, s)
        values  = values.transpose(1, 2).contiguous().view(b * h, t, s)
        #compute the dot product attention scores and scale them by the square root of the dimension of the keys
        dot = torch.bmm(queries, keys.transpose(1, 2)) / (s ** 0.5)
        if mask:
            indices = torch.triu_indices(t, t, offset=1)
            dot[:, indices[0], indices[1]] = float('-inf')
        #normalize the attention scores to probabilities using softmax
        dot = F.softmax(dot, dim=2)
        #apply self-attention by multiplying the attention scores with the values
        out = torch.bmm(dot, values).view(b, h, t, s)
        #unify the attention heads, transpose the output to (b, t, h, s) and then reshape it to (b, t, k)
        #merges all attention heads back toegther for each token
        out = out.transpose(1, 2).contiguous().view(b, t, s * h)
        #apply last lienar layer to combine info across heads and product output od shape [b,t,k]
        return self.unifyheads(out)




#Transformer = a stack of transformer blocks
"""Transformer Block=stack of blocks=(att+FFN+normalization+residuals)"""
class TransformerBlock(nn.Module): #1 layer of transformer
    def __init__(self, k, heads):
        super().__init__()
        #adds a multi-head self-attention layer
        #each token in the sequence can attend to every other token and combine information from different "attention heads"
        self.attention = SelfAttention(k, heads=heads)

        #applied normalization after adding attention results back into input(residual)
        self.norm1 = nn.LayerNorm(k)
        self.norm2 = nn.LayerNorm(k)
        #feedforward network (FFN) that processes the output of the attention layer
        #add non-linearity and allows the model to learn complex transformations
        self.ff = nn.Sequential(
            nn.Linear(k, 4 * k),
            nn.ReLU(),
            nn.Linear(4 * k, k)
        )

    def forward(self, x, mask=False):
        #apply attention and add og input (residual connection), then mormalize
        attended = self.attention(x, mask=mask)
        x = self.norm1(attended + x)
        #apply feedforward network and add og input (residual connection), then normalize
        fedforward = self.ff(x)
        return self.norm2(fedforward + x)

"""Transformer Classifier"""
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, max_seq_len, k, heads, depth, num_classes):
        super().__init__()
        #turn each token index into learned vector of size k
        self.token_embedding = nn.Embedding(vocab_size, k)
        #add info about toker positions
        self.position_embedding = nn.Embedding(max_seq_len, k)
        #stack mult transformer blocks==deeper understanding
        self.blocks = nn.Sequential(*[TransformerBlock(k, heads) for _ in range(depth)])
        #final classifier layer-map output vectors to class logits
        self.to_logits = nn.Linear(k, num_classes)

    def forward(self, x):
        b, t = x.size()
        #create position indices for each token
        positions = torch.arange(t, device=x.device).unsqueeze(0).expand(b, t)
        #combine token and positional embeddings
        x = self.token_embedding(x) + self.position_embedding(positions)
        #pass embedded input through stacked Transformer blocks
        x = self.blocks(x)
        #avg of all token outputs = 1 vector per sequence
        x = x.mean(dim=1)
        #classify each sequence into one of the output classes
        return self.to_logits(x)


model = TransformerClassifier(vocab_size=10000, max_seq_len=128, k=256, heads=4, depth=6, num_classes=2)
output = model(torch.randint(0, 10000, (32, 128)))
