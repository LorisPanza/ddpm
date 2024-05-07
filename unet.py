import torch
import torch.nn as nn
import math


def time_embedding(time_embed_dim, timestamp):


    factor_denomin = 10000**((2 * torch.arange(start=0, end=time_embed_dim//2))/time_embed_dim)

    timesteps_dimensionality_half = timestamp[:,None].repeat(1, time_embed_dim//2)/factor_denomin

    t_embed = torch.cat(tensors=[torch.sin(timesteps_dimensionality_half),torch.cos(timesteps_dimensionality_half)],dim=-1)

    return t_embed



class TimeEmbedding(nn.Module):
    """
    ### Embeddings for $t$
    """

    def __init__(self, n_channels: int):
        """
        * `n_channels` is the number of dimensions in the embedding
        """
        super().__init__()
        self.n_channels = n_channels
        # First linear layer
        self.lin1 = nn.Linear(self.n_channels // 4, self.n_channels)
        # Activation
        self.act = nn.SiLU()
        # Second linear layer
        self.lin2 = nn.Linear(self.n_channels, self.n_channels)

    def forward(self, t: torch.Tensor):
        # Create sinusoidal position embeddings
        # [same as those from the transformer](../../transformers/positional_encoding.html)
        #
        # \begin{align}
        # PE^{(1)}_{t,i} &= sin\Bigg(\frac{t}{10000^{\frac{i}{d - 1}}}\Bigg) \\
        # PE^{(2)}_{t,i} &= cos\Bigg(\frac{t}{10000^{\frac{i}{d - 1}}}\Bigg)
        # \end{align}
        #
        # where $d$ is `half_dim`
        half_dim = self.n_channels // 8
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)

        #print(f"EMB size {emb.shape}")

        # Transform with the MLP
        emb = self.act(self.lin1(emb))
        emb = self.lin2(emb)
        
        return emb



class DownBlock(nn.Module):
    def __init__(self,in_channels, out_channels, t_embed_dim,has_attn):
        super().__init__()

        self.residual_block = ResidualBlock(in_channels,out_channels,t_embed_dim)

        if(has_attn):
            self.attention_block = Attention(out_channels)
        else:
            self.attention_block = nn.Identity()

    def forward(self,x,t):
        x = self.residual_block(x,t)
        x = self.attention_block(x)
        return x


class UpBlock(nn.Module):
    def __init__(self,in_channels, out_channels, t_embed_dim,has_attn):
        super().__init__()

        self.residual_block = ResidualBlock(in_channels+out_channels,out_channels,t_embed_dim)

        if (has_attn):
            self.attention_block = Attention(out_channels)
        else:
            self.attention_block = nn.Identity()

    def forward(self,x,t):

        x = self.residual_block(x,t)
        x = self.attention_block(x)
        return x

class MiddleBlock(nn.Module):
    def __init__(self,n_channels,t_embed_dim):
        super().__init__()

        self.residual_block1 = ResidualBlock(n_channels, n_channels,t_embed_dim)
        self.attention_block = Attention(n_channels)
        self.residual_block2 = ResidualBlock(n_channels, n_channels,t_embed_dim)
    
    def forward(self,x,t):
        x = self.residual_block1(x,t)
        x = self.attention_block(x)
        x = self.residual_block2(x,t)
        return x



class ResidualBlock(nn.Module):
    def __init__(self, in_channels,out_channels, time_embed_dim, dropout=0.6, n_groups=2):
        super().__init__()

        
        self.resnet_block1 = nn.Sequential(
            nn.GroupNorm(n_groups,in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3,3), stride=1, padding=1)
        )

        self.t_emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, out_channels),
        )

        self.resnet_block2 = nn.Sequential(
            nn.GroupNorm(n_groups,out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3,3), stride=1, padding=1)
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.shortcut = nn.Identity()


    def forward(self,x,t):

        h = self.resnet_block1(x)

        h += self.t_emb_layer(t)[:,:,None,None]

        h = self.resnet_block2(h)

        return h + self.shortcut(x)



        
class Attention(nn.Module):
    def __init__(self, embed_dim,n_heads=2):
        super().__init__()

        self.dk = embed_dim

        self.qkv_dim = embed_dim

        self.linear = nn.Linear(embed_dim, 3*embed_dim)

        self.softmax = nn.Softmax(dim=-1) 

        self.scale = self.dk ** 0.5

        self.n_heads = n_heads

    
    def forward(self,input):

        #CASO IN CUI C è UGUALE ALLA DIMENSIONE DELLA MATRICE DI QKV
        batch_size, n_features, height,width = input.shape

        # (B,C,L)->(B,L,C)  L could be also (H,W)
        x = input.view(batch_size,n_features,-1).permute(0,2,1)

        if(self.qkv_dim != n_features):
            print("ERROR - QKV MATRIX FEATURES AND INPUT FEATURES DIM ARE DIFFERENT")

        sub_dim = int((self.qkv_dim*3)/self.n_heads)

        # (B,L,C) -> (B,L,C*3)->(B,L,N_HEADS,C*3/N_HEADS)
        qkv = self.linear(x).view(batch_size,-1,self.n_heads,sub_dim)

        # (B,L,N_HEADS,C*3/N_HEADS) -> (B,L,N_HEADS,C/H_HEADS) for k,q,v
        q,k,v = torch.chunk(qkv,3,dim=-1)

        # (B,L,N_HEADS,C/H_HEADS) -> (B, N_HEADS, L, C/N_HEADS)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # (B, N_HEADS, L, C/H_HEADS) * (B, N_HEADS, C/N_HEADS, L) -> (B, N_HEADS, L, L)
        qk = torch.matmul(q,(k.transpose(-1,-2)))/self.scale

        # (B, N_HEADS, L, L) * (B, N_HEADS, L, C/N_HEADS) ->  (B, N_HEADS, L, C/N_HEADS)
        results_attn = torch.matmul(self.softmax(qk),v)


        # (B, N_HEADS, L, C/N_HEADS ) -> (B, L, N_HEADS, C/N_HEADS)
        results_attn = results_attn.transpose(1,2)

        # (B, L, N_HEADS, C/N_HEAD) -> (B, L, C) 
        results_attn = results_attn.reshape(batch_size,-1,self.qkv_dim)

        #Residual connection (possibile grazie al fatto che qvk dim è uguale al numero di features dell'input)
        results_attn +=x

        results_attn = results_attn.reshape(batch_size,self.qkv_dim,height,width)

        #print(f"Attention shape {results_attn.shape}")

        return results_attn


class Unet(nn.Module):
    def __init__(self,image_channels, channels, ch_mults =(1,2,2,4), time_embedding_size=12, is_attn = (False,False,True,True),n_blocks=2):
        super().__init__()

        unet_depth = len(ch_mults)

        self.time_embedding_size=time_embedding_size


        self.conv = nn.Conv2d(image_channels,channels,kernel_size=(3,3),padding=(1,1)) 

        in_channels = channels

        self.channels = channels

        self.down_list = nn.ModuleList()


        for i in range(unet_depth):
            out_channels = in_channels * ch_mults[i]
            
            for j in range(n_blocks):
                module = DownBlock(in_channels, out_channels, time_embedding_size,has_attn=is_attn[i])
                self.down_list.append(module)
                in_channels=out_channels
            
            if i < unet_depth-1:
                self.down_list.append(nn.Conv2d(in_channels,in_channels,(3,3),(2,2),(1,1)))
            
        self.middle=MiddleBlock(in_channels,time_embedding_size)

        self.upper_list = nn.ModuleList()

        for i in reversed(range(unet_depth)):
            out_channels = in_channels

            for j in range(n_blocks):
                module = UpBlock(in_channels,out_channels,time_embedding_size,has_attn=is_attn[i])
                self.upper_list.append(module)

            out_channels=out_channels//ch_mults[i]
            self.upper_list.append(UpBlock(in_channels,out_channels,time_embedding_size,has_attn= is_attn[i]))
            in_channels=out_channels

            if(i>0):
                self.upper_list.append(nn.ConvTranspose2d(in_channels, in_channels, (4, 4), (2, 2), (1, 1)))

        #self.norm = nn.GroupNorm()
        self.norm = nn.GroupNorm(8, channels)
        self.act = nn.SiLU()
        self.final = nn.Conv2d(in_channels, image_channels, kernel_size=(3, 3), padding=(1, 1))

    def forward(self,x,t):
        #print(f"Input size {x.shape}")

        x = self.conv(x)

        #print(f"First convolution {x.shape}")

        t = time_embedding(self.time_embedding_size,t)

        #print(f"T shape {t.size()}")

        h = [x]
        #print("\n----DOWN---\n")
        for m in self.down_list:
            if(isinstance(m,nn.Conv2d)):
                x=m(x)
                #print(f"Conv block {x.shape}")
            else:
                x = m(x,t)
                #print(f"Downsample block {x.shape}")
            h.append(x)
        
        #print("\n----MIDDLE---\n")
        x = self.middle(x,t)

        #print("\n----UP---\n")
        for m in (self.upper_list):
            if (isinstance(m,nn.ConvTranspose2d)):
                x = m(x)
            else:
                residual = h.pop()
                #print(f"Residual {residual.shape}")
                #print(f"x to be concatenated {x.shape}")
                #print(f"dim concatenated {torch.cat((x,residual),dim=1).shape}")
                x = (torch.cat((x,residual),dim=1))
                x = m(x,t)
        
        return self.final(self.act(self.norm(x)))