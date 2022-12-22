import torch
from repvgg import *
import torch.onnx
import netron

# net = create_RepVGG_A1(deploy=False)
net = create_RepVGG_A1(deploy=True)
input = torch.rand(1, 3, 224, 224)
ouput = net(input)

onnx_path = "repvgg_A1_deploy.onnx"
torch.onnx.export(net, input, onnx_path)

netron.start(onnx_path)