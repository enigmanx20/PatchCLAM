import torch
import torch.nn as nn
import torch.nn.functional as F

class Avg_Pool(nn.Module):
    def __init__(self, dropout = 0.0, n_classes = 1, **kwargs):
        super(Avg_Pool, self).__init__()
        self.n_classes = n_classes
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x):
        """
        A: attention matrix CxNxN, where NxN matrices are diagonal
        """
        N = x.size(0)
        A = A.repeat(1, N).view(self.n_classes, N, N) * \
            torch.eye(N).to(x.device, x.dtype).view(1, N, N).repeat(self.n_classes, 1, 1)
        A = self.attn_drop(A)
        return A, x 

class Attn_Net(nn.Module):
    """
    Attention Network without Gating (2 fc layers)
    args:
        L: input feature dimension
        D: hidden layer dimension
        dropout: whether to use dropout (p = 0.25)
        n_classes: number of classes 
    """
    def __init__(self, L = 1024, D = 256, dropout = 0.0, n_classes = 1):
        super(Attn_Net, self).__init__()
        self.net = nn.Sequential(*[
                                        nn.Linear(L, D, bias=True),
                                        nn.Tanh(),
                                        nn.Dropout(dropout), # original dropout rate is 0.25 if droupout
                                        nn.Linear(D, n_classes, bias=True)
                                        ])
        self.n_classes = n_classes

    def forward(self, x):
        """
        A: attention matrix CxNxN, where NxN matrices are diagonal
        """
        N = x.size(0)
        A = self.net(x) #NxC
        A = A.T
        A = A.repeat(1, N).view(self.n_classes, N, N) * \
            torch.eye(N).to(x.device, x.dtype).view(1, N, N).repeat(self.n_classes, 1, 1)
        return A, x 
    
    
class Attn_Net_Gated(nn.Module):
    """
    Attention Network with Sigmoid Gating (3 fc layers)
    args:
        L: input feature dimension
        D: hidden layer dimension
        dropout: whether to use dropout (p = 0.25)
        n_classes: number of classes 
    """
    def __init__(self, L = 1024, D = 256, dropout = 0.0, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = nn.Sequential(*[
                                            nn.Linear(L, D, bias=True),
                                            nn.Tanh(),
                                            nn.Dropout(dropout), # original dropout rate is 0.25 if droupout
                                            ])
        
        self.attention_b = nn.Sequential(*[
                                            nn.Linear(L, D, bias=True),
                                            nn.Sigmoid(),
                                            nn.Dropout(dropout), # original dropout rate is 0.25 if droupout
                                            ])
        
        self.attention_c = nn.Linear(D, n_classes, bias=True)

        self.n_classes = n_classes

    def forward(self, x):
        """
        A: attention matrix CxNxN
        """
        N = x.size(0)
        a = self.attention_a(x)  # NxD
        b = self.attention_b(x)  # NxD
        A = a * b                # NxD elementwise multiplication
        A = self.attention_c(A)  # Nxn_classes
        A = A.T
        A = A.repeat(1, N).view(self.n_classes, N, N) * \
            torch.eye(N).to(x.device, x.dtype).view(1, N, N).repeat(self.n_classes, 1, 1)
        return A, x
    
class Attn_Net_SDP(nn.Module):
    """
    Scaled dot-product attention
    args:
        L: input feature dimension
        D: hidden layer dimension
        dropout: whether to use dropout (p = 0.25)
        n_classes: number of classes 
    """
    def __init__(self, L = 1024, D = 256, dropout = 0.0, n_classes = 1):
        super(Attn_Net_SDP, self).__init__()
        self.n_classes = n_classes
        self.scale = D ** -0.5

        self.q_proj = nn.Linear(L, D*n_classes, bias=True)
        self.k_proj = nn.Linear(L, D*n_classes, bias=True)

        self.attn_drop = nn.Dropout(dropout)

    def _separate_heads(self, x, num_heads):
        N, D = x.shape
        x = x.reshape(N, num_heads, D // num_heads)
        return x  # N x C x D

    def forward(self, x):
        """
        A: self-attention matrix CxNxN
        """
        N = x.size(0)
        q = self.q_proj(x)  # NxD
        k = self.k_proj(x)  # NxD

        q = self._separate_heads(q, self.n_classes)
        k = self._separate_heads(k, self.n_classes)

        A = q.permute(1, 0, 2) @ k.permute(1, 2, 0) # CxNxN
        A = A * self.scale
        A = self.attn_drop(A)

        return A, x


class CLAM_SB(nn.Module):
    """
    args:
        gate: whether to use gated attention network
        size_arg: config for network size
        dropout: whether to use dropout
        k_sample: number of positive/neg patches to sample for instance-level training
        dropout: whether to use dropout (p = 0.25)
        n_classes: number of classes 
        instance_loss_fn: loss function to supervise instance-level training
        subtyping: whether it's a subtyping problem
    """
    def __init__(self, 
                 atten_type="3fc",  # [avg", "2fc", "3fc", "sdp"]
                 apply_max=False,
                 size_arg="small", 
                 dropout=0.0,       # original dropout rate is 0.25 if dropout
                 num_features=1024, # embedding feature size
                 k_sample=8,        # number of positive and negative samples
                 n_classes=2):
        super(CLAM_SB, self).__init__()
        self.k_sample = k_sample
        self.n_classes = n_classes

        self.size_dict = {"small": [num_features, 512, 256], "big": [num_features, 512, 384]}
        self.size = self.size_dict[size_arg]
        
        self.atten_dict = {
                           "avg": Avg_Pool,
                           "2fc": Attn_Net,
                           "3fc": Attn_Net_Gated,
                           "sdp": Attn_Net_SDP
                           }
        if atten_type == "avg":
            apply_max = False
        self.apply_max = apply_max

        fc = [
                nn.Linear(self.size[0], self.size[1], bias=True),
                nn.ReLU(),
                nn.Dropout(dropout)
                ]

        self.fc = nn.Sequential(*fc)

        attention_net = self.atten_dict[atten_type](L = self.size[1], D = self.size[2], dropout = dropout, n_classes = 1)
        self.attention_net = attention_net
        self.classifiers  = nn.Linear(self.size[1], n_classes, bias=True)    
        self.instance_classifiers = nn.Linear(self.size[1], n_classes, bias=True)
        self.apply(self._initialize_weights)
    
    def _initialize_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    @torch.no_grad()
    def get_topk_ids(self, A):
        """
        A: CxNxN
        """
        C, N, _ = A.size()
        A = F.softmax(A.view(C, -1), dim=-1).view(C, N, N)  # softmax over attention matrix
        A = A.sum(dim=-1) # CxN
        top_p_ids = torch.topk(A, self.k_sample, dim=1)[1]  #CxK
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1] #CxK
        return top_p_ids, top_n_ids

    def intermediate_forward(self, x):
        """
        h: NxL
        """
        N = x.size()
        h = self.fc(x)
        A, h = self.attention_net(h)  # CXNxN, NxL
        if self.apply_max:
            out, indices = F.max_pool2d_with_indices(A.unsqueeze(0), N)
            A = (F.max_unpool2d(out, indices, N) > 0).to(x.dtype)
        return h, A

    def get_attention_map(self, x):
        h, A = self.intermediate_forward(x)
        return A
    
    def bag_forward(self, h, A):
        """
        h: NxL
        A: CxNxN
        """
        C, N, _ = A.size()
        A = F.softmax(A.view(C, -1), dim=1).view(C, N, N)  # softmax over attention matrix
        M = (A @ h).sum(dim=1) # CxNxL ->CxL
        logits = self.classifiers(M) # 1xL -> 1xC
        
        return logits
    
    def instance_forward(self, h):
        instance_logits = self.instance_classifiers(h) # NxC
        return instance_logits

    def forward(self, x):
        """
        x: NxL
        """
        h, A = self.intermediate_forward(x)
        A_raw = A
        logits = self.bag_forward(h, A)
        
        return logits, A_raw
    
class CLAM_MB(CLAM_SB):
    """
    args:
        gate: whether to use gated attention network
        size_arg: config for network size
        dropout: whether to use dropout
        k_sample: number of positive/neg patches to sample for instance-level training
        dropout: whether to use dropout (p = 0.25)
        n_classes: number of classes 
        instance_loss_fn: loss function to supervise instance-level training
        subtyping: whether it's a subtyping problem
    """
    def __init__(self, 
                 atten_type="3fc", # ["avg", "2fc", "3fc", "sdp"]
                 apply_max=False,
                 size_arg="small", # default dropout rate is 0.25 if dropout
                 dropout=0.0, 
                 num_features=1024,
                 k_sample=8,
                 n_classes=2):
        super(CLAM_MB, self).__init__(atten_type, apply_max, size_arg, dropout, num_features, k_sample, n_classes)
        
        attention_net = self.atten_dict[atten_type](L = self.size[1], D = self.size[2], dropout = dropout, n_classes = n_classes)
        self.attention_net = attention_net

        self.classifiers  = nn.Conv1d(n_classes, n_classes, kernel_size=self.size[1],\
                                      groups=n_classes, stride=1, padding=0, bias=True)
        self.apply(self._initialize_weights)
        
    def bag_forward(self, h, A):
        """
        h: NxL
        A: CxNxN
        """
        C, N, _ = A.size()
        A = F.softmax(A.view(C, -1), dim=1).view(C, N, N)  # softmax over attention matrix
        M = (A @ h).sum(dim=1) # CxNxL ->CxL
        #print(M.size())
        logits = self.classifiers(M.unsqueeze(0)) # 1xCxL -> 1xC
        
        return logits.view(-1, self.n_classes)


class ClamWrapper(nn.Module):
    def __init__(self, clam_config, base_encoder=None, num_enmedding_features=1024):
        super(ClamWrapper, self).__init__()
        self.encoder = base_encoder    

        if clam_config['clam_type'] == 'SB':
            self.clam = CLAM_SB(atten_type=clam_config['atten_type'], num_features=num_enmedding_features, dropout=clam_config['drop_out'], 
                                apply_max=clam_config['apply_max'], k_sample=clam_config['k_sample'], n_classes=clam_config['n_classes'])
        elif clam_config['clam_type'] == 'MB':
            self.clam = CLAM_MB(atten_type=clam_config['atten_type'], num_features=num_enmedding_features, dropout=clam_config['drop_out'],
                                apply_max=clam_config['apply_max'], k_sample=clam_config['k_sample'], n_classes=clam_config['n_classes'])
            
    def eval_forward(self, input1):
        shape = input1.shape
        n = shape[1]
        input1 = input1.view(-1,shape[2],shape[3],shape[4])
        
        x = self.encoder.forward_features(input1)
        x = x.view(n,-1)
        
        h, A = self.clam.intermediate_forward(x)
        
        bag_logit = self.clam.bag_forward(h, A)
        
        return bag_logit
        
    def forward(self, input1):
        shape = input1.shape
        n = shape[1]
        input1 = input1.view(-1,shape[2],shape[3],shape[4])
        
        x = self.encoder.forward_features(input1)
        x = x.view(n,-1)
        
        h, A = self.clam.intermediate_forward(x)
        
        bag_logit = self.clam.bag_forward(h, A)
        inst_logit = self.clam.instance_forward(h)   
        
        top_p_ids, top_n_ids = self.clam.get_topk_ids(A)

        return bag_logit, inst_logit, top_p_ids, top_n_ids