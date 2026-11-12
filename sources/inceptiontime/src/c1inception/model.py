"""Declared InceptionTime adaptation, not the original five-network ensemble."""
from __future__ import annotations
import torch
from torch import nn
from .io import require

class InceptionModule(nn.Module):
    def __init__(self, incoming, filters, bottleneck, kernels, dilation, eps, momentum):
        super().__init__()
        self.bottleneck = nn.Conv1d(incoming, bottleneck, 1, bias=False)
        self.branches = nn.ModuleList([nn.Conv1d(bottleneck, filters, k,
            padding=dilation*(k-1)//2, dilation=dilation, bias=False) for k in kernels])
        self.pool_branch = nn.Sequential(nn.MaxPool1d(3, stride=1, padding=1),
                                         nn.Conv1d(incoming, filters, 1, bias=False))
        self.norm = nn.BatchNorm1d(4*filters, eps=eps, momentum=momentum)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        b = self.bottleneck(x)
        return self.relu(self.norm(torch.cat([layer(b) for layer in self.branches]
                                             + [self.pool_branch(x)], dim=1)))

class InceptionTime1D(nn.Module):
    def __init__(self, architecture, classes=12):
        super().__init__()
        a = architecture
        require(a["name"] == "adapted_multiscale_inception1d", "Wrong architecture")
        kernels, dilations = a["kernels"], a["module_dilations"]
        require(len(kernels)==3 and all(isinstance(k,int) and k>0 and k%2 for k in kernels),
                "Three odd positive temporal kernels required")
        require(len(dilations)==6 and all(isinstance(d,int) and d>0 for d in dilations),
                "Six positive module dilations required")
        filters, bottleneck = a["filters_per_branch"], a["bottleneck_channels"]
        require(filters>0 and bottleneck>0 and classes>1, "Invalid dimensions")
        require(a["residual_every"]==3 and a["max_pool_kernel"]==3, "Residual/pooling convention changed")
        require(a["ensemble"] is False and a["initialization"]=="kaiming_normal_fan_out",
                "Single-model initialization policy changed")
        self.architecture = dict(a)
        self.inception = nn.ModuleList()
        self.shortcuts = nn.ModuleList()
        current, residual_input = 6, 6
        eps, momentum = a["batch_norm_eps"], a["batch_norm_momentum"]
        for index, dilation in enumerate(dilations):
            self.inception.append(InceptionModule(current,filters,bottleneck,kernels,dilation,eps,momentum))
            current = 4*filters
            if (index+1)%3==0:
                self.shortcuts.append(nn.Sequential(nn.Conv1d(residual_input,current,1,bias=False),
                    nn.BatchNorm1d(current,eps=eps,momentum=momentum)))
                residual_input = current
        self.relu = nn.ReLU(inplace=False)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(current,classes)
        self.receptive_field_samples = 1+sum(max((max(kernels)-1)*d,2) for d in dilations)
        for layer in self.modules():
            if isinstance(layer,nn.Conv1d):
                nn.init.kaiming_normal_(layer.weight,mode="fan_out",nonlinearity="relu")
            elif isinstance(layer,nn.BatchNorm1d):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self,x):
        if x.ndim!=3 or x.shape[1]!=6:
            raise ValueError("Expected batch-by-six-channels-by-time tensor")
        residual = x
        for index,module in enumerate(self.inception):
            x = module(x)
            if (index+1)%3==0:
                x = self.relu(x+self.shortcuts[index//3](residual))
                residual = x
        return self.classifier(self.pool(x).squeeze(-1))

    def description(self,sampling_rate_hz=40000):
        return {"name":"adapted_multiscale_inception1d","original_five_model_ensemble":False,
            "parameters":sum(p.numel() for p in self.parameters()),
            "trainable_parameters":sum(p.numel() for p in self.parameters() if p.requires_grad),
            "inception_modules":6,"residual_groups":2,
            "filters_per_branch":self.architecture["filters_per_branch"],
            "bottleneck_channels":self.architecture["bottleneck_channels"],
            "branch_kernels":self.architecture["kernels"],
            "module_dilations":self.architecture["module_dilations"],
            "convolution_layers":sum(isinstance(m,nn.Conv1d) for m in self.modules()),
            "receptive_field_samples":self.receptive_field_samples,
            "receptive_field_ms":1000*self.receptive_field_samples/sampling_rate_hz,
            "receptive_field_scope":"convolutional local support before global pooling, evaluation-mode batch norm",
            "temporal_downsampling":False,"global_average_pooling":True}
