import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from torch.nn import MultiheadAttention
import torch.nn.functional as F
import math
from modules import embedding


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=120):
        super(PositionalEmbedding, self).__init__()
        self.pos_emb_table = embedding(max_len, d_model, zeros_pad=False, scale=False)
        pos_vector = torch.arange(max_len)
        self.dropout = nn.Dropout(p=dropout)
        self.register_buffer('pos_vector', pos_vector)

    def forward(self, x):
        pos_emb = self.pos_emb_table(self.pos_vector[:x.size(0)].unsqueeze(1).repeat(1, x.size(1)))
        x += pos_emb
        return self.dropout(x)


class LocPredictor(nn.Module):
    def __init__(self, nuser, nloc, ntime, nreg, user_dim, loc_dim, time_dim, reg_dim, nhid, nhead_enc, nhead_dec, nlayers, dropout=0.5, **extra_config):
        super(LocPredictor, self).__init__()
        self.emb_user = embedding(nuser, user_dim, zeros_pad=True, scale=True)
        self.emb_loc = embedding(nloc, loc_dim, zeros_pad=True, scale=True)
        self.emb_reg = embedding(nreg, reg_dim, zeros_pad=True, scale=True)
        self.emb_time = embedding(ntime, time_dim, zeros_pad=True, scale=True)
        if not ((user_dim == loc_dim) and (user_dim == time_dim) and (user_dim == reg_dim)):
            raise Exception('user, location, time and region should have the same embedding size')
        ninp = user_dim
        pos_encoding = extra_config.get("position_encoding", "transformer")
        if pos_encoding == "embedding":
            self.pos_encoder = PositionalEmbedding(ninp, dropout)
        elif pos_encoding == "transformer":
            self.pos_encoder = PositionalEncoding(ninp, dropout)
        self.enc_layer = TransformerEncoderLayer(ninp, nhead_enc, nhid, dropout)
        self.encoder = TransformerEncoder(self.enc_layer, nlayers)
        if not extra_config.get("use_location_only", False):
            if extra_config.get("embedding_fusion", "multiply") == "concat":
                if extra_config.get("user_embedding", False):
                    self.lin = nn.Linear(user_dim + loc_dim + reg_dim + time_dim, ninp)
                else:
                    self.lin = nn.Linear(loc_dim + reg_dim + time_dim, ninp)

        ident_mat = torch.eye(ninp)
        self.register_buffer('ident_mat', ident_mat)

        self.layer_norm = nn.LayerNorm(ninp)
        self.extra_config = extra_config
        self.dropout = dropout


    def forward(self, src_user, src_loc, src_reg, src_time, src_square_mask, src_binary_mask, trg_loc, mem_mask, ds=None):
        loc_emb_src = self.emb_loc(src_loc)
        if self.extra_config.get("user_location_only", False):
            src = loc_emb_src
        else:
            user_emb_src = self.emb_user(src_user)
            reg_emb = self.emb_reg(src_reg)
            time_emb = self.emb_time(src_time)
            if self.extra_config.get("embedding_fusion", "multiply") == "multiply":
                if self.extra_config.get("user_embedding", False):
                    src = loc_emb_src * reg_emb * time_emb * user_emb_src
                else:
                    src = loc_emb_src * reg_emb * time_emb
            else:
                if self.extra_config.get("user_embedding", False):
                    src = torch.cat([user_emb_src, loc_emb_src, reg_emb, time_emb], dim=-1)
                else:
                    src = torch.cat([loc_emb_src, reg_emb, time_emb], dim=-1)
                src = self.lin(src)

        if self.extra_config.get("size_sqrt_regularize", True):
            src = src * math.sqrt(src.size(-1))

        src = self.pos_encoder(src)
        # shape: [L, N, ninp]
        src = self.encoder(src, mask=src_square_mask)
        # shape: [(1+K)*L, N, loc_dim]
        loc_emb_trg = self.emb_loc(trg_loc)

        if self.extra_config.get("use_attention_as_decoder", False):
            # multi-head attention
            output, _ = F.multi_head_attention_forward(
                query=loc_emb_trg,
                key=src,
                value=src,
                embed_dim_to_check=src.size(2),
                num_heads=1,
                in_proj_weight=None,
                in_proj_bias=None,
                bias_k=None,
                bias_v=None,
                add_zero_attn=None,
                dropout_p=0.0,
                out_proj_weight=self.ident_mat,
                out_proj_bias=None,
                training=self.training,
                key_padding_mask=src_binary_mask,
                need_weights=False,
                attn_mask=mem_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.ident_mat,
                k_proj_weight=self.ident_mat,
                v_proj_weight=self.ident_mat
            )
            
            if self.training:
                src = src.repeat(loc_emb_trg.size(0) // src.size(0), 1, 1)
            else:
                src = src[torch.tensor(ds) - 1, torch.arange(len(ds)), :]
                src = src.unsqueeze(0).repeat(loc_emb_trg.size(0), 1, 1) 

            output += src
            output = self.layer_norm(output)
        else:
            # No attention
            if self.training:
                output = src.repeat(loc_emb_trg.size(0) // src.size(0), 1, 1)
            else:
                output = src[torch.tensor(ds) - 1, torch.arange(len(ds)), :]
                output = output.unsqueeze(0).repeat(loc_emb_trg.size(0), 1, 1)

        # shape: [(1+K)*L, N]
        output = torch.sum(output * loc_emb_trg, dim=-1)
        return output

    def save(self, path):
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        self.load_state_dict(torch.load(path))


class QuadKeyLocPredictor(nn.Module):
    def __init__(self, nuser, nloc, ntime, nquadkey, user_dim, loc_dim, time_dim, reg_dim, nhid, nhead_enc, nhead_dec, nlayers, dropout=0.5, **extra_config):
        super(QuadKeyLocPredictor, self).__init__()
        self.emb_user = embedding(nuser, user_dim, zeros_pad=True, scale=True)
        self.emb_loc = embedding(nloc, loc_dim, zeros_pad=True, scale=True)
        self.emb_reg = embedding(nquadkey, reg_dim, zeros_pad=True, scale=True)
        self.emb_time = embedding(ntime, time_dim, zeros_pad=True, scale=True)
        ninp = user_dim
        pos_encoding = extra_config.get("position_encoding", "transformer")
        if pos_encoding == "embedding":
            self.pos_encoder = PositionalEmbedding(loc_dim + reg_dim, dropout)
        elif pos_encoding == "transformer":
            self.pos_encoder = PositionalEncoding(loc_dim + reg_dim, dropout)
        self.enc_layer = TransformerEncoderLayer(loc_dim + reg_dim, nhead_enc, loc_dim + reg_dim, dropout)
        self.encoder = TransformerEncoder(self.enc_layer, nlayers)
        self.region_pos_encoder = PositionalEmbedding(reg_dim, dropout, max_len=20)
        self.region_enc_layer = TransformerEncoderLayer(reg_dim, 1, reg_dim, dropout=dropout)
        self.region_encoder = TransformerEncoder(self.region_enc_layer, 2)
        if not extra_config.get("use_location_only", False):
            if extra_config.get("embedding_fusion", "multiply") == "concat":
                if extra_config.get("user_embedding", False):
                    self.lin = nn.Linear(user_dim + loc_dim + reg_dim + time_dim, ninp)
                else:
                    self.lin = nn.Linear(loc_dim + reg_dim, ninp)

        ident_mat = torch.eye(ninp)
        self.register_buffer('ident_mat', ident_mat)

        self.layer_norm = nn.LayerNorm(ninp)
        self.extra_config = extra_config
        self.dropout = dropout

        #self.region_gru_encoder = torch.nn.GRU(input_size=reg_dim, hidden_size=reg_dim, num_layers=2, dropout=0.0, bidirectional=True)
        #self.h_0 = nn.Parameter(torch.randn((4, 1, reg_dim), requires_grad=True))

    def forward(self, src_user, src_loc, src_reg, src_time, src_square_mask, src_binary_mask, trg_loc, trg_reg, mem_mask, ds=None):
        loc_emb_src = self.emb_loc(src_loc)
        if self.extra_config.get("user_location_only", False):
            src = loc_emb_src
        else:
            user_emb_src = self.emb_user(src_user)
            # (L, N, LEN_QUADKEY, REG_DIM)
            reg_emb = self.emb_reg(src_reg)
            reg_emb = reg_emb.view(reg_emb.size(0) * reg_emb.size(1), reg_emb.size(2), reg_emb.size(3)).permute(1, 0, 2)
            # (LEN_QUADKEY, L * N, REG_DIM)
            
            reg_emb = self.region_pos_encoder(reg_emb)
            reg_emb = self.region_encoder(reg_emb)
            #avg pooling 
            reg_emb = torch.mean(reg_emb, dim=0)
            
            #reg_emb, _ = self.region_gru_encoder(reg_emb, self.h_0.expand(4, reg_emb.size(1), -1).contiguous())
            #reg_emb = reg_emb[-1, :, :]

            reg_emb = reg_emb.view(loc_emb_src.size(0), loc_emb_src.size(1), reg_emb.size(1))
            time_emb = self.emb_time(src_time)
            if self.extra_config.get("embedding_fusion", "multiply") == "multiply":
                if self.extra_config.get("user_embedding", False):
                    src = loc_emb_src * reg_emb * time_emb * user_emb_src
                else:
                    src = loc_emb_src * reg_emb * time_emb
            else:
                if self.extra_config.get("user_embedding", False):
                    src = torch.cat([user_emb_src, loc_emb_src, reg_emb, time_emb], dim=-1)
                else:
                    src = torch.cat([loc_emb_src, reg_emb], dim=-1)
 
        if self.extra_config.get("size_sqrt_regularize", True):
            src = src * math.sqrt(src.size(-1))

        src = self.pos_encoder(src)
        # shape: [L, N, ninp]
        src = self.encoder(src, mask=src_square_mask)
 
        # shape: [(1+K)*L, N, loc_dim]
        loc_emb_trg = self.emb_loc(trg_loc)

        reg_emb_trg = self.emb_reg(trg_reg)
        reg_emb_trg = reg_emb_trg.view(reg_emb_trg.size(0) * reg_emb_trg.size(1), reg_emb_trg.size(2), reg_emb_trg.size(3)).permute(1, 0, 2)
        reg_emb_trg = self.region_pos_encoder(reg_emb_trg)
        reg_emb_trg = self.region_encoder(reg_emb_trg)
        reg_emb_trg = torch.mean(reg_emb_trg, dim=0)
        reg_emb_trg = reg_emb_trg.view(loc_emb_trg.size(0), loc_emb_trg.size(1), reg_emb_trg.size(1))

        loc_emb_trg = torch.cat([loc_emb_trg, reg_emb_trg], dim=-1)
        if self.extra_config.get("use_attention_as_decoder", False):
            # multi-head attention
            output, _ = F.multi_head_attention_forward(
                query=loc_emb_trg,
                key=src,
                value=src,
                embed_dim_to_check=src.size(2),
                num_heads=1,
                in_proj_weight=None,
                in_proj_bias=None,
                bias_k=None,
                bias_v=None,
                add_zero_attn=None,
                dropout_p=0.0,
                out_proj_weight=self.ident_mat,
                out_proj_bias=None,
                training=self.training,
                key_padding_mask=src_binary_mask,
                need_weights=False,
                attn_mask=mem_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.ident_mat,
                k_proj_weight=self.ident_mat,
                v_proj_weight=self.ident_mat
            )
            
            if self.training:
                src = src.repeat(loc_emb_trg.size(0) // src.size(0), 1, 1)
            else:
                src = src[torch.tensor(ds) - 1, torch.arange(len(ds)), :]
                src = src.unsqueeze(0).repeat(loc_emb_trg.size(0), 1, 1) 

            output += src
            output = self.layer_norm(output)
        else:
            # No attention
            if self.training:
                output = src.repeat(loc_emb_trg.size(0) // src.size(0), 1, 1)
            else:
                output = src[torch.tensor(ds) - 1, torch.arange(len(ds)), :]
                output = output.unsqueeze(0).repeat(loc_emb_trg.size(0), 1, 1)

        # shape: [(1+K)*L, N]
        output = torch.sum(output * loc_emb_trg, dim=-1)
        return output

    def save(self, path):
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        self.load_state_dict(torch.load(path))



class GRU4Rec(nn.Module):
    def __init__(self, nloc, loc_dim, num_layers=1, dropout=0.0):
        super(GRU4Rec, self).__init__()
        self.emb_loc = embedding(nloc, loc_dim, zeros_pad=True, scale=True)
        self.encoder = torch.nn.GRU(input_size=loc_dim, hidden_size=loc_dim, num_layers=num_layers, dropout=dropout)
        self.h_0 = nn.Parameter(torch.randn((num_layers, 1, loc_dim), requires_grad=True))


    def forward(self, src_user, src_loc, src_reg, src_time, src_square_mask, src_binary_mask, trg_loc, mem_mask, ds=None):
        loc_emb_src = self.emb_loc(src_loc)
        # shape: [L, N, ninp]
        src, _ = self.encoder(loc_emb_src, self.h_0.expand(-1, loc_emb_src.size(1), -1).contiguous())
        # shape: [(1+K)*L, N, loc_dim]
        loc_emb_trg = self.emb_loc(trg_loc)

        if self.training:
            output = src.repeat(loc_emb_trg.size(0) // src.size(0), 1, 1)
        else:
            output = src[torch.tensor(ds) - 1, torch.arange(len(ds)), :]
            output = output.unsqueeze(0).repeat(loc_emb_trg.size(0), 1, 1)

        output = torch.sum(output * loc_emb_trg, dim=-1)
        return output

    def save(self, path):
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        self.load_state_dict(torch.load(path))