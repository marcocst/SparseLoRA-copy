#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Function

from typing import Optional, List
from sklearn.metrics import mutual_info_score

import os
STE_DECAY = float(os.environ.get("STE_DECAY", "1e-4"))
LORA_R_SCALE = int(os.environ.get("LORA_R_SCALE", "2"))
LORA_SPARSE  = int(os.environ.get("LORA_SPARSE", "1"))



class GSTE(nn.Module):
    def __init__(self, num_bits):
        super(GSTE, self).__init__()
        init_range = 2.0
        self.n_val = 2 ** num_bits
        # print("==================")
        # print(self.n_val)
        # print("========================")
        self.interval = init_range / self.n_val
        self.start = nn.Parameter(torch.Tensor([0.0]).float(), requires_grad=True)
        self.a = nn.Parameter(torch.Tensor([self.interval] * self.n_val).float(), requires_grad=True)
        self.scale1 = nn.Parameter(torch.Tensor([1.0]).float(), requires_grad=True)
        self.scale2 = nn.Parameter(torch.Tensor([1.0]).float(), requires_grad=True)

        self.two = nn.Parameter(torch.Tensor([2.0]).float(), requires_grad=False)
        self.one = nn.Parameter(torch.Tensor([1.0]).float(), requires_grad=False)
        self.zero = nn.Parameter(torch.Tensor([0.0]).float(), requires_grad=False)
        self.minusone = nn.Parameter(torch.Tensor([-1.0]).float(), requires_grad=False)
        self.eps = nn.Parameter(torch.Tensor([1e-3]).float(), requires_grad=False)

    def forward(self, x, mask):
        device = x.device
        dtype = x.dtype  # 获取输入张量的数据类型

        # 使用临时变量将参数移动到输入的设备上，并确保类型一致
        scale1 = self.scale1.to(device, dtype=dtype)
        scale2 = self.scale2.to(device, dtype=dtype)
        start = self.start.to(device, dtype=dtype)
        a = self.a.to(device, dtype=dtype)
        zero = self.zero.to(device, dtype=dtype)
        eps = self.eps.to(device, dtype=dtype)

        # 量化处理
        x_forward = x * scale1
        x_backward = x

        # 提前计算所有的 thre_forward 和 thre_backward，确保类型一致
        thre_forward = start + torch.cumsum(torch.cat((a[:1] / 2, a[1:] / 2 + a[:-1] / 2)), dim=0).to(device, dtype=dtype)
        thre_backward = start + torch.cumsum(torch.cat((torch.tensor([0.0], device=device, dtype=dtype), a[:-1])), dim=0).to(device, dtype=dtype)

        # 避免在循环中进行不必要的操作，合并 `torch.where`
        for i in range(self.n_val):
            # 计算当前区间的 step_right
            step_right = torch.tensor((i + 1) * self.interval, device=device, dtype=dtype)

            # 批量更新 x_forward 和 x_backward，减少 torch.where 的使用
            x_forward = torch.where(x > thre_forward[i], step_right, x_forward)
            x_backward = torch.where(x > thre_backward[i], self.interval / a[i] * (x - thre_backward[i]) + step_right - self.interval, x_backward)
        
        # 量化结果应用 mask
        out = (x_forward.detach() + x_backward - x_backward.detach()) * mask
        return out
    def backward(self, ctx, g):  # g 是 L 对 y 的导数。
        x, mask = ctx.saved_tensors
        
        # 使用 G-STE 的梯度近似方法来计算输入 x 的梯度
        x_backward = x
        step_right = self.zero + 0.0

        for i in range(self.n_val):
            step_right += self.interval
            if i == 0:
                thre_backward = self.start + 0.0
                x_backward = torch.where(x > thre_backward, self.interval/a_pos[i] * (x - thre_backward) + step_right - self.interval, zero)
            else:
                thre_backward += a_pos[i-1]
                x_backward = torch.where(x > thre_backward, self.interval/a_pos[i] * (x - thre_backward) + step_right - self.interval, x_backward)
        
        # 反向传播中保留需要的梯度，并使用 mask 调整输出
        grad_input = g * mask + STE_DECAY * (1 - mask) * x_backward  # 使用 G-STE 的梯度近似
        return grad_input, None  # 只返回输入的梯度，mask 无需梯度






#  cao 注释开始
class STE(Function):
    @staticmethod
    def forward(ctx, x, mask):   # x:lora_Q(r) mask:需要保留的。
        ctx.save_for_backward(x, mask)
        return (x + 1) * mask    # 这里是不是找错了，感觉.

    @staticmethod
    def backward(ctx, g):  # g是L对y的导数，那么这个y是(x+1)*mask吗？
        x, mask = ctx.saved_tensors
        return g + STE_DECAY * (1 - mask) * x, None  # srste
    

class GSTE1(Function):
    @staticmethod
    def forward(ctx, x, mask):   
        ctx.save_for_backward(x, mask)
        return (x+1) * mask  # 保持简单的Mask稀疏化操作

    @staticmethod
    def backward(ctx, g): 
        x, mask = ctx.saved_tensors
        
        # 对被稀疏化的部分（mask为0）进行更加精细的梯度调整
        # 这里根据 x 的大小来调整梯度，而不仅仅是简单的 STE_DECAY
        # sparse_part_adjustment = (1 - mask) * (STE_DECAY * torch.abs(x))  
        a = torch.abs(x) / torch.sum(torch.abs(x))
        aa = -a * torch.log(a + 1e-8)
        sparse_part_adjustment = (1-mask) * (STE_DECAY * aa)
        
        # 对未稀疏的部分（mask为1），正常传递梯度
        return g + sparse_part_adjustment, None 
    

class STE_mi(Function):
    @staticmethod
    def forward(ctx, x, mask):   
        ctx.save_for_backward(x, mask)
        return (x+1) * mask  # 保持简单的Mask稀疏化操作

    @staticmethod
    def backward(ctx, g): 
        x, mask = ctx.saved_tensors
        
        # 对被稀疏化的部分（mask为0）进行更加精细的梯度调整
        # 这里根据 x 的大小来调整梯度，而不仅仅是简单的 STE_DECAY
        # sparse_part_adjustment = (1 - mask) * (STE_DECAY * torch.abs(x))  
        x_numpy = x.cpu().numpy()
        aa = mutual_info_score(x_numpy, x_numpy)
        print("aa_ste:", aa)

        sparse_part_adjustment = (1-mask) * (STE_DECAY * aa)
        
        # 对未稀疏的部分（mask为1），正常传递梯度
        return g + sparse_part_adjustment, None 

    


# class STE(Function):
#     @staticmethod
#     def forward(ctx, x, mask):   
#         ctx.save_for_backward(x, mask)
#         return (x + 1) * mask   

#     @staticmethod
#     def backward(ctx, g): 
#         x, mask = ctx.saved_tensors
#         return g + STE_DECAY * (1 - mask) * x, None 
# masked_weight = STE.apply(self.lora_Q, self.lora_QMask).view(1, -1)
# result = F.linear(x, self.weight, bias=self.bias)
# result += (self.lora_dropout(x) @ ((self.lora_B * masked_weight) @ self.lora_A).T) * self.scaling
# return result


    
# cao注释结束。

class LoRALayer():
    def __init__(
        self, 
        r: int, 
        lora_alpha: int, 
        lora_dropout: float,
        merge_weights: bool,
    ):
        print(f"Use official LoRA, STE_DECAY={STE_DECAY:g}..........: r={r}, alpha={lora_alpha}, lora_dropout={lora_dropout}, merge_weights={merge_weights}")
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights

class Linear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            if LORA_SPARSE == 1:
                self.lora_A = nn.Parameter(self.weight.new_zeros((r * LORA_R_SCALE, in_features)))
                self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r * LORA_R_SCALE)))
                self.scaling = self.lora_alpha / self.r
                self.lora_Q = nn.Parameter(self.weight.new_ones(r * LORA_R_SCALE))
                self.register_buffer("lora_QMask", self.weight.new_ones(r * LORA_R_SCALE))  # 缓冲区，但在优化过程中不会更新梯度
            else:
                self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
                self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
                self.scaling = self.lora_alpha / self.r

            self.weight.requires_grad = False  # 原本的参数不需要进行更新。
        self.reset_parameters()  
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))  # 也就是说A初始化不为0.
            # self.lora_A.data*=math.sqrt(self.r*LORA_R_SCALE)/math.sqrt(self.r)
            nn.init.zeros_(self.lora_B)

            if LORA_SPARSE == 1:
                nn.init.zeros_(self.lora_Q)

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)

    def forward(self, x: torch.Tensor):
        # print("=====================================")
        # print("Linear")
        # print("=============================================")
        if LORA_SPARSE == 1:
            # gste_layer = GSTE(num_bits=5)  # 创建 GSTE 实例
            # masked_weight = gste_layer(self.lora_Q + 1, self.lora_QMask).view(1, -1)


            # gste_layer1 = SimplifiedGSTE(num_bits=6)  # 创建 GSTE 实例
            # masked_weight = gste_layer1(self.lora_Q + 1, self.lora_QMask).view(1, -1)



            # masked_weight = GSTE1.apply(self.lora_Q + 1, self.lora_QMask).view(1, -1)
            masked_weight = GSTE1.apply(self.lora_Q + 1, self.lora_QMask).view(1, -1)
            # masked_weight = STE.apply(self.lora_Q, self.lora_QMask).view(1, -1) # lora_Q:r(64个), lora_QMsk:要保留的。注释掉的。
            # masked_weight = ((self.lora_Q + 1) * self.lora_QMask).view(1, -1)  # 这个是把STE取消了。
            result = F.linear(x, self.weight, bias=self.bias)
            # result += (self.lora_dropout(x) @ ((self.lora_B * masked_weight) @ (self.idct + self.lora_a @ self.lora_b)).T) * self.scaling
            # result += (self.lora_dropout(x) @ ((self.lora_B * masked_weight) @ (self.idct)).T) * self.scaling
            # masked_weight = torch.ones(1, 64)
            result += (self.lora_dropout(x) @ ((self.lora_B * masked_weight) @ self.lora_A).T) * self.scaling
            return result
        else:
            result = F.linear(x, self.weight, bias=self.bias)
            result += (self.lora_dropout(x) @ (self.lora_B @ self.lora_A).T) * self.scaling
            return result
        



class MergedLinear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        enable_lora: List[bool] = [False],
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        assert out_features % len(enable_lora) == 0, \
            'The length of enable_lora must divide out_features'
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0 and any(enable_lora):
            if LORA_SPARSE == 1:
                self.lora_A = nn.Parameter(self.weight.new_zeros((r * sum(enable_lora) * LORA_R_SCALE, in_features)))
                self.lora_B = nn.Parameter(self.weight.new_zeros((out_features // len(enable_lora) * sum(enable_lora) , r * LORA_R_SCALE)))
                self.scaling = self.lora_alpha / self.r
                self.lora_Q = nn.Parameter(self.weight.new_ones(r * LORA_R_SCALE))
                self.register_buffer("lora_QMask", self.weight.new_ones(r * LORA_R_SCALE))
            
            else:
                self.lora_A = nn.Parameter(
                    self.weight.new_zeros((r * sum(enable_lora), in_features)))
                self.lora_B = nn.Parameter(
                    self.weight.new_zeros((out_features // len(enable_lora) * sum(enable_lora), r))
                ) # weights for Conv1D with groups=sum(enable_lora)
            self.scaling = self.lora_alpha / self.r
                # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            # Compute the indices
            self.lora_ind = self.weight.new_zeros(
                (out_features, ), dtype=torch.bool
            ).view(len(enable_lora), -1)
            self.lora_ind[enable_lora, :] = True
            self.lora_ind = self.lora_ind.view(-1)
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

            if LORA_SPARSE == 1:
                nn.init.zeros_(self.lora_Q)

    def zero_pad(self, x):
        result = x.new_zeros((len(self.lora_ind), *x.shape[1:]))
        result[self.lora_ind] = x
        return result

    def merge_AB(self, masked_weight):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        delta_w = F.conv1d(
            (T(self.lora_A).unsqueeze(0)).transpose(1,2), 
            (self.lora_B*masked_weight).unsqueeze(-1), 
            groups=sum(self.enable_lora)
        ).squeeze(0)
        return T(self.zero_pad(delta_w))

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0 and any(self.enable_lora):
                    self.weight.data -= self.merge_AB() * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0 and any(self.enable_lora):
                    self.weight.data += self.merge_AB() * self.scaling
                self.merged = True        

    def forward(self, x: torch.Tensor):
        if LORA_SPARSE == 1:
            masked_weight = GSTE.apply(self.lora_Q, self.lora_QMask).view(1, -1)
            def T(w):
                return w.transpose(0, 1) if self.fan_in_fan_out else w
            if self.merged:
                return F.linear(x, T(self.weight), bias=self.bias)
            else:
                result = F.linear(x, T(self.weight), bias=self.bias)
                if self.r > 0:
                    result += self.lora_dropout(x) @ T(self.merge_AB(masked_weight=masked_weight).T) * self.scaling
                return result
        else:
            def T(w):
                return w.transpose(0, 1) if self.fan_in_fan_out else w
            if self.merged:
                return F.linear(x, T(self.weight), bias=self.bias)
            else:
                result = F.linear(x, T(self.weight), bias=self.bias)
                if self.r > 0:
                    result += self.lora_dropout(x) @ T(self.merge_AB().T) * self.scaling
                return result
        



# class MergedLinear_old(nn.Linear, LoRALayer):
#     # LoRA implemented in a dense layer
#     def __init__(
#         self, 
#         in_features: int, 
#         out_features: int, 
#         r: int = 0, 
#         lora_alpha: int = 1, 
#         lora_dropout: float = 0.,
#         enable_lora: List[bool] = [False],
#         fan_in_fan_out: bool = False,
#         merge_weights: bool = True,
#         **kwargs
#     ):
#         nn.Linear.__init__(self, in_features, out_features, **kwargs)
#         LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
#                            merge_weights=merge_weights)
#         assert out_features % len(enable_lora) == 0, \
#             'The length of enable_lora must divide out_features'
#         self.enable_lora = enable_lora
#         self.fan_in_fan_out = fan_in_fan_out
#         # Actual trainable parameters
#         if r > 0 and any(enable_lora):
#             self.lora_A = nn.Parameter(
#                 self.weight.new_zeros((r * sum(enable_lora), in_features)))
#             self.lora_B = nn.Parameter(
#                 self.weight.new_zeros((out_features // len(enable_lora) * sum(enable_lora), r))
#             ) # weights for Conv1D with groups=sum(enable_lora)
#             self.scaling = self.lora_alpha / self.r
#             # Freezing the pre-trained weight matrix
#             self.weight.requires_grad = False
#             # Compute the indices
#             self.lora_ind = self.weight.new_zeros(
#                 (out_features, ), dtype=torch.bool
#             ).view(len(enable_lora), -1)
#             self.lora_ind[enable_lora, :] = True
#             self.lora_ind = self.lora_ind.view(-1)
#         self.reset_parameters()
#         if fan_in_fan_out:
#             self.weight.data = self.weight.data.transpose(0, 1)

#     def reset_parameters(self):
#         nn.Linear.reset_parameters(self)
#         if hasattr(self, 'lora_A'):
#             # initialize A the same way as the default for nn.Linear and B to zero
#             nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
#             nn.init.zeros_(self.lora_B)

#     def zero_pad(self, x):
#         result = x.new_zeros((len(self.lora_ind), *x.shape[1:]))
#         result[self.lora_ind] = x
#         return result

#     def merge_AB(self):
#         def T(w):
#             return w.transpose(0, 1) if self.fan_in_fan_out else w
#         delta_w = F.conv1d(
#             self.lora_A.unsqueeze(0), 
#             self.lora_B.unsqueeze(-1), 
#             groups=sum(self.enable_lora)
#         ).squeeze(0)
#         return T(self.zero_pad(delta_w))

#     def train(self, mode: bool = True):
#         def T(w):
#             return w.transpose(0, 1) if self.fan_in_fan_out else w
#         nn.Linear.train(self, mode)
#         if mode:
#             if self.merge_weights and self.merged:
#                 # Make sure that the weights are not merged
#                 if self.r > 0 and any(self.enable_lora):
#                     self.weight.data -= self.merge_AB() * self.scaling
#                 self.merged = False
#         else:
#             if self.merge_weights and not self.merged:
#                 # Merge the weights and mark it
#                 if self.r > 0 and any(self.enable_lora):
#                     self.weight.data += self.merge_AB() * self.scaling
#                 self.merged = True        

#     def forward(self, x: torch.Tensor):

#         def T(w):
#             return w.transpose(0, 1) if self.fan_in_fan_out else w
#         if self.merged:
#             return F.linear(x, T(self.weight), bias=self.bias)
#         else:
#             result = F.linear(x, T(self.weight), bias=self.bias)
#             if self.r > 0:
#                 result += self.lora_dropout(x) @ T(self.merge_AB().T) * self.scaling
#             return result


