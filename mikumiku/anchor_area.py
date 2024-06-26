import torch
import torch.nn as nn
from torch_cluster import radius_graph
from torch_cluster import radius
from torch_geometric.data import Batch

import matplotlib.pyplot as plt
from layers import FourierEmbedding

from utils.circle import minimum_enclosing_circle
from modules.encoder import QCNetEncoder

from torch_geometric.utils import dense_to_sparse

from layers.attention_layer import AttentionLayer
from utils.geometry import angle_between_2d_vectors

# Area_anchor_self_attn: 使用自注意力area_anchor_query提取信息
class Area_anchor_self_attn(nn.Module):
    def __init__(self,
                 input_dim_anchor_pts=2,
                 head_num = 8,
                 anchor_query_hidden_dim=128,
                 anchor_area_query_num_freq_bands=64,
                 anchor_area_max_pts_num = 400,
                 dropout=0.1) -> None:
        super(Area_anchor_self_attn, self).__init__()
        self.input_dim_anchor_pts = input_dim_anchor_pts
        self.anchor_query_hidden_dim = anchor_query_hidden_dim
        self.anchor_area_max_pts_num = anchor_area_max_pts_num
        self.anchor_area_query_num_freq_bands = anchor_area_query_num_freq_bands
        self.anchor_area_pos_embed = FourierEmbedding(input_dim=self.input_dim_anchor_pts, 
                                                       hidden_dim=self.anchor_query_hidden_dim, 
                                                       num_freq_bands=self.anchor_area_query_num_freq_bands)

        self.pos_embed_mlp = nn.Linear(anchor_query_hidden_dim,anchor_query_hidden_dim)
        self.self_attn = nn.MultiheadAttention(self.anchor_query_hidden_dim,head_num,dropout)
        self.attn_postnorm = nn.LayerNorm(anchor_query_hidden_dim)

    def forward(self,
                query_content:torch.Tensor, 
                query_pos:torch.Tensor, ):   
        query_pos_shape = query_pos.shape
        query_pos_embed = query_pos.view(-1, query_pos.size(-1)) 
        query_pos_embed = self.anchor_area_pos_embed(query_pos_embed)
        query_pos_embed = query_pos_embed.view(query_pos_shape[0], query_pos_shape[1], -1)
        query_pos_embed = self.pos_embed_mlp(query_pos_embed)
        
        q = k = query_content + query_pos_embed
        v = query_content
        tmp,_  = self.self_attn(q.permute(1,0,2),k.permute(1,0,2),v.permute(1,0,2))
        tmp = tmp.permute(1,0,2)
        new_query_content = self.attn_postnorm(tmp + query_content) 
        new_query_pos = query_pos
        return new_query_content, new_query_pos
    
# Area_Attn: 交叉注意力机制提取area_anchor_query和traj_feat交互后的特征
class Area_Attn(nn.Module):
    def __init__(self,
                 input_dim_anchor_pts=2,
                 head_num = 8,
                 anchor_query_hidden_dim=128,
                 anchor_area_query_num_freq_bands=64,
                 anchor_area_max_pts_num = 400,
                 dropout=0.1) -> None:
        super(Area_Attn,self).__init__()
        self.anchor_area_max_pts_num = anchor_area_max_pts_num
        self.input_dim_anchor_pts = input_dim_anchor_pts
        self.anchor_query_hidden_dim = anchor_query_hidden_dim
        self.anchor_area_query_num_freq_bands = anchor_area_query_num_freq_bands
        self.anchor_area_pos_embed = FourierEmbedding(input_dim=self.input_dim_anchor_pts, 
                                                       hidden_dim=self.anchor_query_hidden_dim, 
                                                       num_freq_bands=self.anchor_area_query_num_freq_bands)
        self.attn = nn.MultiheadAttention(self.anchor_query_hidden_dim,head_num,dropout)
        self.attn_postnorm = nn.LayerNorm(anchor_query_hidden_dim)
        self.query_content_mlp = nn.Linear(anchor_query_hidden_dim,anchor_query_hidden_dim)
        self.new_query_pos = nn.Linear(anchor_query_hidden_dim,input_dim_anchor_pts)
        
    def forward(self, 
                query_content:torch.Tensor, 
                query_pos:torch.Tensor, 
                agent_history_traj_feat:torch.Tensor,
                ):
        query_pos_shape = query_pos.shape
        query_pos = query_pos.view(-1, query_pos.size(-1))
        area_anchor_query_pos = self.anchor_area_pos_embed(continuous_inputs=query_pos,categorical_embs=None)
        area_anchor_query_pos = area_anchor_query_pos.view(-1, self.anchor_area_max_pts_num, self.anchor_query_hidden_dim)
        q_area_query = (area_anchor_query_pos +  query_content).permute(1,0,2)
        k_agent_history_traj_feat = v_agent_history_traj_feat = agent_history_traj_feat.permute(1,0,2)

        tmp,_ = self.attn(q_area_query, k_agent_history_traj_feat, v_agent_history_traj_feat)
        tmp = tmp.permute(1,0,2)
        query_content = self.attn_postnorm(query_content + tmp)
        new_query_content = self.attn_postnorm(self.query_content_mlp(query_content) + query_content)
        new_query_pos = query_pos.reshape(query_pos_shape) + self.new_query_pos(new_query_content)

        return  new_query_content, new_query_pos
    



class Area_anchor(nn.Module):
    def __init__(self,
                 dim=2,                                     # setting the geo dim in prediction
                 agent_feat_dim=4,                          # setting the num of agent input feature
                 self_attn_num_layers = 1,                  # setting the num of agent_self_attn
                 agent_query_hidden_dim=128,                # setting the num of query_hidden_dim used for anchor_query
                 history_steps=50,
                 time_span = None,
                 input_dim_anchor_pts=2,                    # setting the num of dim used for query_pos
                 anchor_query_hidden_dim=128,               # used like query_pos hidden dim 
                 anchor_query_num_freq_bands=64,            # used for query_pos fouri embed
                 anchor_area_query_num_freq_bands=64,       # used for agent_traj embed
                 agent2map_pts_max_length=400,              # setting the max num of anchor pts
                 agent_self_attn_num_head=8,                # setting agent_self_attn TR model's head
                 agent_self_attn_head_dim=16,               # setting agent_self_attn TR model's head dim hidden=head_num*head_dim
                 agent_self_attn_dropout=0.1,
                 agent_self_attn_fouri_embed=64,
                 query_pts_num=400
                 ) -> None:      
        super(Area_anchor, self).__init__()
        self.dim = dim
        self.query_pts_num = query_pts_num
        self.feat_dim = agent_query_hidden_dim
        self.self_attn_num_layers = self_attn_num_layers
        self.history_steps = history_steps
        self.agent2map_pts_max_length = agent2map_pts_max_length
        self.time_span = time_span if time_span is not None else history_steps
        self.query_pos = FourierEmbedding(input_dim=input_dim_anchor_pts, 
                                          hidden_dim=anchor_query_hidden_dim, 
                                          num_freq_bands=anchor_query_num_freq_bands)
        self.agent_self_pos_embed = FourierEmbedding(input_dim=dim, 
                                        hidden_dim=agent_query_hidden_dim, 
                                        num_freq_bands=agent_self_attn_fouri_embed)

        self.query_embed = nn.Parameter(torch.rand(self.agent2map_pts_max_length, self.feat_dim))
        self.type_agent_history_traj_emb =  nn.Embedding(10, agent_query_hidden_dim)
        self.agent_history_traj_emb = FourierEmbedding(input_dim=agent_feat_dim, hidden_dim=agent_query_hidden_dim, num_freq_bands=anchor_area_query_num_freq_bands)

        # setting agent_traj_self_attn
        self.self_attn_num_layers = self_attn_num_layers
        self.agent_traj_self_attn = nn.ModuleList(
            [AttentionLayer(hidden_dim=agent_query_hidden_dim, 
                            num_heads=agent_self_attn_num_head, 
                            head_dim=agent_self_attn_head_dim, 
                            dropout=agent_self_attn_dropout,
                            bipartite=False, has_pos_emb=True) for _ in range(self_attn_num_layers)])
        self.area_attn = Area_Attn(head_num=agent_self_attn_num_head)
        self.area_anchor_self_attn = Area_anchor_self_attn()

    def forward(self,data):
        init_anchor_pts = self.init_anchor_area(data, 
                                                self.history_steps, 
                                                self.dim, 
                                                self.agent2map_pts_max_length,
                                                relative_pos=False)
        init_area_anchor_query_content = self.query_embed.unsqueeze(0).repeat(data['agent']['num_nodes'],1,1)

        # ==============================agent_history_traj self_attn============================== #
        pos_a = data['agent']['position'][:, :self.history_steps, :self.dim].contiguous()
        motion_vector_a = torch.cat([pos_a.new_zeros(data['agent']['num_nodes'], 1, self.dim),pos_a[:, 1:] - pos_a[:, :-1]], dim=1)
        head_a = data['agent']['heading'][:, :self.history_steps].contiguous()
        head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)
        vel = data['agent']['velocity'][:, :self.history_steps, :self.dim].contiguous()
        agent_traj_feat = torch.stack( #agent_traj_feat shape: [agent_num, self.history_steps, self.dim]
            [torch.norm(motion_vector_a[:, :, :2], p=2, dim=-1),
                angle_between_2d_vectors(ctr_vector=head_vector_a, nbr_vector=motion_vector_a[:, :, :2]),
                torch.norm(vel[:, :, :2], p=2, dim=-1),
                angle_between_2d_vectors(ctr_vector=head_vector_a, nbr_vector=vel[:, :, :2])], dim=-1)  
        categorical_embs = [self.type_agent_history_traj_emb(data['agent']['type'].long()).repeat_interleave(repeats=self.history_steps,dim=0)]
        agent_traj_feat = self.agent_history_traj_emb(continuous_inputs=agent_traj_feat.view(-1, agent_traj_feat.size(-1)), categorical_embs=categorical_embs)
        agent_traj_feat = agent_traj_feat.view(-1, self.history_steps, self.feat_dim) # agent_traj_feat shape [agent_num,self.history_steps,self.feat_dim]
        
        # set timestep index
        mask = data['agent']['valid_mask'][:, :self.history_steps].contiguous()
        mask_t = mask.unsqueeze(2) & mask.unsqueeze(1)
        edge_index_t = dense_to_sparse(mask_t)[0]
        edge_index_t = edge_index_t[:, edge_index_t[1] > edge_index_t[0]]
        edge_index_t = edge_index_t[:, edge_index_t[1] - edge_index_t[0] <= self.time_span]
        
        # set pos_embed
        pos_t = pos_a.reshape(-1, self.dim) 
        rel_pos_t = pos_t[edge_index_t[0]] - pos_t[edge_index_t[1]]
        r_t = rel_pos_t[:, :2]
        r_t = self.agent_self_pos_embed(continuous_inputs=r_t, categorical_embs=None)
        for i in range(self.self_attn_num_layers):
            agent_traj_feat = agent_traj_feat.reshape(-1, self.feat_dim)
            agent_traj_feat = self.agent_traj_self_attn[i](agent_traj_feat, r_t, edge_index_t)
        
        agent_traj_feat = agent_traj_feat.reshape(-1,self.history_steps,self.feat_dim)
        # ======================================================================================== #

        # ====================================choose pred traj==================================== #
        pred_mask = data['agent']['predict_mask'].any(dim=-1, keepdim=True)
        init_area_anchor_query_content_need_pred = init_area_anchor_query_content[pred_mask.squeeze()]
        agent_traj_feat_need_pred = agent_traj_feat[pred_mask.squeeze()]
        init_anchor_pts_need_pred = init_anchor_pts[pred_mask.squeeze()]
        # ======================================================================================== #
        init_area_anchor_query_content_need_pred,init_anchor_pts_need_pred= self.area_anchor_self_attn(init_area_anchor_query_content_need_pred,
                                                                                                       init_anchor_pts_need_pred)
        _,new_query_pos = self.area_attn(query_content = init_area_anchor_query_content_need_pred,
                       query_pos = init_anchor_pts_need_pred,
                       agent_history_traj_feat = agent_traj_feat_need_pred)
        return new_query_pos, init_anchor_pts


    def init_anchor_area(self, data, history_steps, dim, query_pts_num=400, relative_pos=True):
        device = data['agent']['predict_mask'].device
        pred_mask = data['agent']['predict_mask'].any(dim=-1, keepdim=True)
        init_anchor_list = []
        for each_agent_idx in range(data['agent']['num_nodes']):
            if not pred_mask[each_agent_idx]:
                init_anchor_list.append(torch.zeros(query_pts_num, dim).to(device))
                continue
            ref = data['agent']['position'][each_agent_idx,self.history_steps,:dim]
            valid_history_steps = (data['agent']['position'][each_agent_idx,:history_steps,:dim].permute(1,0) \
            * data['agent']['valid_mask'][each_agent_idx,:history_steps]).permute(1,0)
            valid_history_steps = valid_history_steps - ref
            proj_pred_pts = - valid_history_steps[data['agent']['valid_mask'][each_agent_idx,:history_steps]]
            try:
                tmp = proj_pred_pts.repeat(query_pts_num // proj_pred_pts.shape[0],1)
                _,r = minimum_enclosing_circle(valid_history_steps[data['agent']['valid_mask'][each_agent_idx,:history_steps],:])
                tmp = (torch.cat((tmp,proj_pred_pts[:query_pts_num % proj_pred_pts.shape[0]]), dim=0) + torch.randn(query_pts_num,dim)*r*0.1)
            except:
                tmp = torch.randn(query_pts_num,dim).to(device)
            if not relative_pos:
                tmp = (tmp + ref).to(device)
            init_anchor_list.append(tmp)
        init_anchor_pts = torch.stack(init_anchor_list,dim=0)
        return init_anchor_pts
    

# 不完整
class Pts_anchor(nn.Module):
    def __init__(self) -> None:
        super(Pts_anchor,self).__init__()
        #self.adj_attn = Adj_atten
        #self.map_attn = Map_attn

    def forward(self, data, history_step=50, dim = 2):
        agent_pts_t0=data['agent']['position'][:,history_step,:dim]
        agent_vel_t0=data['agent']['velocity'][:,history_step,:dim]
        anchor_t1 = self.generate_trajectories(agent_pts_t0, agent_vel_t0)
        print(anchor_t1.shape)
        

        return None
    def generate_trajectories(self, a, b, theta=torch.pi/6, num_traj=7, timestep=0.1):
        """ 
        theta: 最大偏转角
        timestep: 时间步间隔, default=0.1
        num_traj: the number of traj needed to be pred, default = 7 and should be odd
        """
        num_agents = a.shape[0]
        thetas = torch.linspace(-theta, theta, num_traj)
        thetas = thetas.view(num_traj, 1, 1)
        cos_thetas = torch.cos(thetas)  # (num_traj, 1, 1)
        sin_thetas = torch.sin(thetas)  # (num_traj, 1, 1)
        b_expanded = b.unsqueeze(0)  # (1, num_agents, 2)
        rotated_b_x = b_expanded[:, :, 0:1] * cos_thetas - b_expanded[:, :, 1:2] * sin_thetas
        rotated_b_y = b_expanded[:, :, 0:1] * sin_thetas + b_expanded[:, :, 1:2] * cos_thetas
        rotated_b = torch.cat([rotated_b_x, rotated_b_y], dim=2)  # (num_traj, num_agents, 2)
        new_positions = a.unsqueeze(0) + rotated_b * timestep # (num_traj, num_agents, 2)
        c = torch.cat([new_positions, rotated_b], dim=2)  # (num_traj, num_agents, 4)
        return c