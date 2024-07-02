import torch
import torch.nn as nn
from utils import weight_init
import numpy
import math
import torch.nn.functional as F

class MLPLayer(nn.Module):

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int) -> None:
        super(MLPLayer, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        ).double()
        self.apply(weight_init)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
    

class MLPLayer_proj_real(nn.Module):

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 ouput_dim:int) -> None:
        super(MLPLayer_proj_real, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, ouput_dim),
        ).double()
        self.apply(weight_init)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
    

import torch.nn.functional as F
class Entropy_encode(nn.Module):
    def __init__(self, input_dim, attention_dim, max_len=100) -> None:
        super(Entropy_encode, self).__init__()
        self.query = nn.Linear(input_dim, attention_dim)
        self.key = nn.Linear(input_dim, attention_dim)
        self.value = nn.Linear(input_dim, attention_dim)
        self.scale = attention_dim ** 0.5
        self.fpn = nn.Linear(attention_dim,attention_dim)
        
        self.position_embedding = nn.Embedding(max_len, attention_dim)
        self.register_buffer("position_ids", torch.arange(max_len).expand((1, -1)))
        self.norm = nn.LayerNorm(attention_dim)
        self.apply(weight_init)

    def forward(self, x:torch.Tensor, mask_start_vector:torch.Tensor):
        """
        x : entropy vector of magnitude or theta, size=[n, history_steps:default=50] 
        mask_start_vector: the start step of entropy vector, size=[n]
        """
        batch_size, seq_len, _= x.size()
        position_ids = self.position_ids[:, :seq_len]
        position_embeddings = self.position_embedding(position_ids)

        Q = self.query(x) + position_embeddings
        K = self.key(x) + position_embeddings
        V = self.value(x) + position_embeddings

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        mask = torch.zeros((batch_size, seq_len, seq_len), dtype=torch.bool, device=x.device)

        mask_start_matrix = mask_start_vector.view(batch_size, 1, 1).expand(batch_size, seq_len, seq_len)
        sequence_indices = torch.arange(seq_len, device=x.device).view(1, 1, seq_len).expand(batch_size, seq_len, seq_len)
        mask = sequence_indices >= mask_start_matrix
        attention_scores = attention_scores.masked_fill(mask, float('-inf'))
        attention_weights = F.softmax(attention_scores, dim=-1)

        attention_output = torch.matmul(attention_weights, V)

        attn_output = attention_output + V
        attn_output = self.norm(attn_output)
        attn_output_ = self.fpn(attn_output) + attn_output

        return attn_output_
    
import numpy
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        self.d_model = d_model
        self.max_len = max_len
        super(PositionalEncoding, self).__init__()
        self.apply(weight_init)
        
    def forward(self, x):
        device = x.device
        pe = torch.zeros(self.max_len, self.d_model).to(device)
        position = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1).to(device)
        div_term = torch.exp(torch.arange(0,self.d_model, 2).float() * (-math.log(10000.0) / self.d_model)).to(device).to(device)

        pe[:, 0::2] = torch.sin(position * div_term).to(device)
        pe[:, 1::2] = torch.cos(position * div_term).to(device)
        pe = pe.unsqueeze(0).to(device)
        
        self.register_buffer('pe', pe)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x
        
class Traj_pred_pts(nn.Module):
    def __init__(self,
                 traj_dim = 2,
                 traj_hidden_dim = 128,
                 histotry_steps = 50,
                 pred_step = 60,) -> None:
        super(Traj_pred_pts, self).__init__()
        self.traj_dim = traj_dim
        self.pred_step = pred_step
        self.histotry_steps = histotry_steps
        self.timestep_proj = nn.Linear(histotry_steps, pred_step)

        self.position = PositionalEncoding(traj_dim * histotry_steps, 100)
        self.traj_feat_encode = MLPLayer(input_dim=traj_dim * histotry_steps, hidden_dim=traj_hidden_dim)
        self.apply(weight_init)
        
    def forward(self, x:torch.Tensor,):
        """  
        x: traj history position, size=[batch, history_steps, 2]
        start: start timestep idx, size=[batch]
        """
        device = x.device
        batch_size, seq_len, _= x.size()
        ref = x[:,-1,:].unsqueeze(1).repeat(1,seq_len,1)
        x_norm = x - ref
        mask = torch.tensor(numpy.tril(numpy.ones(self.histotry_steps),k=1).T[::-1].copy()).to(x.device)
        if self.pred_step > self.histotry_steps: 
            mask = torch.cat((mask, torch.ones(self.histotry_steps,self.pred_step - self.histotry_steps).to(x.device)),dim=1)
        elif self.pred_step == self.histotry_steps: 
            pass
        else: # TO DO : how to process the scene that pred_step is shorter than history step
            raise ValueError('our model cannot process the scene that pred_step is shorter than history step')
        mask = mask.unsqueeze(-1).repeat(1,1,self.traj_dim)
        mask = mask.unsqueeze(0).repeat(batch_size,1,1,1)
        print(x_norm)
        x_expand = x_norm.unsqueeze(2).repeat(1,1,self.pred_step,1)
        traj_masked = x_expand * mask # size = [batch_size, hisotry_steps, pred_steps, traj_dim]
        
        traj_masked = traj_masked.permute(2,0,1,3) # size = [pred_steps, batch_size, hisotry_steps, traj_dim]
        timestep_embedding = self.position(torch.zeros(batch_size, self.pred_step, self.traj_dim * self.histotry_steps).to(device))
        traj_masked = traj_masked.reshape(self.pred_step,batch_size,-1)
        traj_masked = (traj_masked.permute(1,0,2) + timestep_embedding).permute(1,0,2)
        
        traj_pred_pts_ = []
        for each_pred_proj in traj_masked: # each_pred_proj size = [batch_size, hisotry_steps * traj_dim]
            each_pred_proj_hidden = self.traj_feat_encode(each_pred_proj)
            traj_pred_pts_.append(each_pred_proj_hidden)
        traj_pred_pts = torch.stack(traj_pred_pts_,dim=1)
        return traj_pred_pts
    

def vec_diff(A:torch.Tensor, B:torch.Tensor):
    magnitude_A = torch.norm(A, dim=1)
    magnitude_B = torch.norm(B, dim=1)
    magnitude_change = magnitude_B - magnitude_A

    unit_A = A / magnitude_A.unsqueeze(1)
    unit_B = B / magnitude_B.unsqueeze(1)

    dot_product = torch.sum(unit_A * unit_B, dim=1)
    dot_product = torch.clamp(dot_product, -1.0, 1.0)

    angle_change = torch.acos(dot_product)
    return magnitude_change, angle_change

class Area_anchor(nn.Module):
    def __init__(self,
                 traj_dim=2,                                     # setting the geo dim in prediction
                 history_steps=50,
                 pred_steps = 60,
                 entropy_hidden_dim = 128,
                 mod = 6,
                 ) -> None:      
        super(Area_anchor, self).__init__()
        self.traj_dim = traj_dim
        self.history_steps = history_steps
        self.pred_steps = pred_steps

        self.entropy_magnitude = Entropy_encode(1, entropy_hidden_dim)
        self.entropy_theta = Entropy_encode(1, entropy_hidden_dim)
        self.traj_pred_pts = Traj_pred_pts()
        self.history_2_pred_magnitude = nn.Linear(self.history_steps, self.pred_steps)
        self.history_2_pred_theta = nn.Linear(self.history_steps, self.pred_steps)

        self.proj_mods = MLPLayer_proj_real(entropy_hidden_dim, entropy_hidden_dim, traj_dim * mod)
        self.apply(weight_init)
        
    def forward(self,data,):
        pred_mask = data['agent']['predict_mask'].any(dim=-1, keepdim=True).squeeze()
        valid_train_data = data['agent']['position'][pred_mask,:self.history_steps,:self.traj_dim]
        batch = valid_train_data.shape[0]

        magnitude_change, angle_change, start = self.entropy(data, self.history_steps, self.traj_dim)
        magnitude_change = magnitude_change.unsqueeze(dim=-1)


        #print('before',magnitude_change[19])
        #magnitude_change = self.scaler.fit_transform(magnitude_change)
        #print('after',magnitude_change)

        
        entropy_magnitude = self.entropy_magnitude(magnitude_change, start)
        #print('entropy_magnitude',entropy_magnitude)
        entropy_theta = self.entropy_theta(magnitude_change, start) # size = [batch, history_steps, entropy_hidden_dim]
        #print('entropy_theta',entropy_theta)
        traj_pred_pts = self.traj_pred_pts(valid_train_data)        # size = [batch, history_steps, entropy_hidden_dim]
        #print('traj_pred_pts',traj_pred_pts)
        
        entropy_magnitude_pred = self.history_2_pred_magnitude(entropy_magnitude.permute(0,2,1)).permute(0,2,1)
        entropy_theta_pred = self.history_2_pred_theta(entropy_theta.permute(0,2,1)).permute(0,2,1)
        fuse = (traj_pred_pts + entropy_magnitude_pred + \
                entropy_theta_pred).permute(1,0,2) # size = [history_steps, batch,entropy_hidden_dim]
        
        all_traj_proj_2_real_ = []
        for each_traj in fuse:
            each_traj_proj_2_real = self.proj_mods(each_traj)
            all_traj_proj_2_real_.append(each_traj_proj_2_real)
        all_traj_proj_2_real = torch.stack(all_traj_proj_2_real_,dim=1).reshape(batch, -1, self.traj_dim)

        return all_traj_proj_2_real
    

    def entropy(self, data, history_steps, dim):
        device = data['agent']['predict_mask'].device
        pred_mask = data['agent']['predict_mask'].any(dim=-1, keepdim=True).squeeze()
        valid_train_data = data['agent']['position'][pred_mask,:history_steps,:]
        tmp_1 = []
        tmp_2 = []
        src = []
        for each_train_data,i in zip(valid_train_data, range(valid_train_data.shape[0])):
            void_magnitude_change = torch.zeros(history_steps).to(device)
            void_angle_change = torch.zeros(history_steps).to(device)

            src_idx = data['agent']['valid_mask'][pred_mask][i,1:history_steps] ^ data['agent']['valid_mask'][pred_mask][i,:history_steps-1]
            src_idx = (torch.arange(0,history_steps-1).to(device))[src_idx]
            ref = each_train_data[-1,:dim]
            if torch.tensor([]).shape == src_idx.shape:
                src_idx = torch.tensor([history_steps-1]).to(device)
            each_train_data = each_train_data[src_idx:history_steps, :dim]
            if src_idx < history_steps -2:
                vec_src =  each_train_data[1:] - each_train_data[:-1] # [history_steps-1,2]
                vec_except = torch.cat((each_train_data[0].unsqueeze(0) ,each_train_data[1:] + vec_src)) - ref

                vec_except_ = vec_except[1:-1] - (each_train_data[1:-1] - ref)
                vec_train = each_train_data[2:] - each_train_data[1:-1]
                
                each_train_data=each_train_data.detach().cpu().numpy()
                magnitude_change, angle_change = vec_diff(vec_except_, vec_train)
                void_magnitude_change[src_idx+2:] = magnitude_change
                void_angle_change[src_idx+2:] = angle_change
                if vec_except.shape[0] == 0:
                    raise ValueError('vec_except has been wrong processed')
            elif src_idx == history_steps -2 :
                void_magnitude_change = torch.zeros(history_steps).to(device)
                void_angle_change = torch.zeros(history_steps).to(device)
            elif src_idx == history_steps -1 :
                void_magnitude_change = torch.zeros(history_steps).to(device)
                void_angle_change = torch.zeros(history_steps).to(device)
            else:
                raise ValueError('each_train_data size is wrong')
            src.append(src_idx)
            tmp_1.append(void_magnitude_change)
            tmp_2.append(void_angle_change)
        return torch.stack(tmp_1,dim=0), torch.stack(tmp_2,dim=0), torch.cat(src)