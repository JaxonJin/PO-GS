import torch
from torch import nn

def Myln(x, eps=1e-5):
    n = x.size(-1)
    sum_x = torch.sum(x, dim=-1, keepdim=True)
    mean = sum_x / n
    squared_diff = (x - mean) ** 2
    sum_squared_diff = torch.sum(squared_diff, dim=-1, keepdim=True)
    std = torch.sqrt(sum_squared_diff / n + eps)
    normed_x = (x - mean) / (std + eps)
    return  normed_x


class MHA(nn.Module):
    def __init__(self, input_dim, num_heads):
        super(MHA, self).__init__()
        self.num_heads = num_heads
        self.input_dim = input_dim
        self.head_dim = input_dim // num_heads

        self.query_linear = nn.Linear(input_dim, input_dim)
        self.key_linear = nn.Linear(input_dim, input_dim)
        self.value_linear = nn.Linear(input_dim, input_dim)
        self.out_linear = nn.Linear(input_dim, input_dim)

        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)

        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        batch_size, input_dim = x.shape
        Q = self.query_linear(x)
        K = self.key_linear(x)
        V = self.value_linear(x)

        Q = Q.view(batch_size, self.num_heads, self.head_dim)
        K = K.view(batch_size, self.num_heads, self.head_dim)
        V = V.view(batch_size, self.num_heads, self.head_dim)

        attention_scores = torch.einsum('bhd,bhd->bh', Q, K) / (self.head_dim ** 0.5)

        attention_weights = torch.softmax(attention_scores, dim=-1)

        attention_output = torch.einsum('bh,bhd->bhd', attention_weights, V)

        attention_output = attention_output.contiguous().view(batch_size, self.input_dim)

        x = self.norm1(attention_output + x)

        output = self.out_linear(x)
        output = self.dropout(output)
        return output



class GRA(nn.Module):
    def __init__(self):
        super(GRA, self).__init__()
        self.conv_x = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1)
        self.conv_y = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1)
        self.sigmoid_x = nn.Sigmoid()
        self.sigmoid_y = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(0)
        b, batch_size, input_dim = x.shape

        x_avg_pool = torch.mean(x, dim=-1, keepdim=True)

        y_avg_pool = torch.mean(x, dim=1, keepdim=True)

        x_conv = self.conv_x(x_avg_pool.permute(0, 2, 1))
        y_conv = self.conv_y(y_avg_pool)

        x_norm = Myln(x_conv.squeeze(-2))
        y_norm = Myln(y_conv.squeeze(-2))

        x_sigmoid = self.sigmoid_x(x_norm.unsqueeze(-1))
        y_sigmoid = self.sigmoid_y(y_norm.unsqueeze(-2))

        out = x * x_sigmoid * y_sigmoid
        out = out.contiguous().view(batch_size, input_dim)
        return out


class Hierarchical_Granular_Structural_Attention(nn.Module):
    def __init__(self, input_dim, num_heads):
        super(Hierarchical_Granular_Structural_Attention, self).__init__()
        self.gra = GRA()
        self.mha = MHA(input_dim, num_heads)

    def forward(self, x):
        x = self.gra(x)
        x = self.mha(x)
        return x