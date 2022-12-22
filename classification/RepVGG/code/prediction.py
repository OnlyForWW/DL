import os
from PIL import Image
from repvgg import *
import torch
import time
import torchvision
from torch.utils.tensorboard import SummaryWriter

path = "pred"
filelist = os.listdir(path=path)
total_num = len(filelist)

print("图片数量为：" + str(total_num))

# 建筑、森林、冰川、山、海、街道
classes = {
    0: 'buildings',
    1: 'forest',
    2: 'glacier',
    3: 'mountain',
    4: 'sea',
    5: 'street'
}

# transformer设置
transformer = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize((150, 150)),
        torchvision.transforms.ToTensor()
    ]
)

# 加载模型
deploy = False
use_checkpoint = False
net = RepVGG(num_blocks=[2, 4, 14, 1], num_classes=6,
             width_multiplier=[1, 1, 1, 2.5], override_groups_map=None, deploy=deploy,
             use_checkpoint=use_checkpoint)
net.load_state_dict(torch.load("best.pth"))
deploy_net = repvgg_model_convert(net, save_path=None)


file = open("Result.txt", "w", encoding="UTF-8")
start_time = time.time()
for i in range(total_num):
    img_path = path + "\\" + filelist[i]
    img = Image.open(img_path)
    img = img.convert("RGB")
    img = transformer(img)
    img = torch.reshape(img, (1, 3, 150, 150))
    net.eval()
    with torch.no_grad():
        output = net(img)
    index = int(output.argmax(1))
    file.write(img_path + " " + classes[index] + "\n")
file.close()
total_time = time.time() - start_time
print('Predicting complete in {:.0f}m {:.0f}s'.format(total_time // 60, total_time % 60))


# if __name__ == "__main__":
#     with open("test.txt", "w", encoding="UTF-8") as f:
#         for i in range(10):
#             f.write(str(i) + "\n")
#     deploy = False
#     use_checkpoint = False
#     net = RepVGG(num_blocks=[2, 4, 14, 1], num_classes=6,
#                  width_multiplier=[1, 1, 1, 2.5], override_groups_map=None, deploy=deploy,
#                  use_checkpoint=use_checkpoint)
#     net.load_state_dict(torch.load("best.pth"))
#     deploy_net = repvgg_model_convert(net, save_path=None)
#     img = Image.open("pred/3.jpg")
#     img.convert("RGB")
#     img = transformer(img)
#     img = torch.reshape(img, (1, 3, 150, 150))
#     net.eval()
#     with torch.no_grad():
#         output = deploy_net(img)
#     index = int(output.argmax(1))
#     print(classes[index])
#     w = SummaryWriter("./log_test")
#     input = torch.rand(1, 3, 150, 150)
#     w.add_graph(deploy_net, input)
#     w.close()