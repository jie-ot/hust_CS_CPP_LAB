python3 --version
python3 -m venv deformable_detr
source deformable_detr/bin/activate
deactivate

pip install torch==<匹配JetPack版本> torchvision==<匹配JetPack版本>
pip install numpy pillow matplotlib opencv-python
pip install pycocotools tqdm cython scipy

pip3 install torch==2.1.0a0+41361538.nv23.6
pip3 install torchvision==0.16.0+fbb4cc5
pip3 install numpy==1.24.4
pip3 install onnx==1.15.0
pip3 install pillow matplotlib opencv-python
pip3 install pycocotools tqdm cython scipy

Reading package lists... Done
Building dependency tree       
Reading state information... Done
Some packages could not be installed. This may mean that you have
requested an impossible situation or if you are using the unstable
distribution that some required packages have not yet been created
or been moved out of Incoming.
The following information may help to resolve the situation:

The following packages have unmet dependencies:
 python3.8-venv : Depends: python3.8 (= 3.8.10-0ubuntu1~20.04.18) but 3.8.10-0ubuntu1~20.04.9 is to be installed
E: Unable to correct problems, you have held broken packages.

import torch
from models.deformable_detr import DeformableDETR
from PIL import Image
import torchvision.transforms as T

# 构建模型（默认参数）
model = DeformableDETR(num_classes=91)
model.eval()

# 加载图像并预处理
img = Image.open("test.jpg").convert("RGB")
transform = T.Compose([
    T.Resize(480),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
])
img_tensor = transform(img).unsqueeze(0)  # [1, 3, H, W]

# 构造 NestedTensor（必须）
from util.misc import nested_tensor_from_tensor_list
samples = nested_tensor_from_tensor_list([img_tensor])

# 推理
with torch.no_grad():
    outputs = model(samples)

# 打印输出形状
print("pred_logits:", outputs["pred_logits"].shape)
print("pred_boxes:", outputs["pred_boxes"].shape)


class _NewEmptyTensorOp(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, new_shape):
            ctx.shape = x.shape
            return x.new_empty(new_shape)

        @staticmethod
        def backward(ctx, grad):
            shape = ctx.shape
            return _NewEmptyTensorOp.apply(grad, shape), None

import torch
from models.deformable_detr import DeformableDETR
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 1. 构建模型并加载预训练权重
model = DeformableDETR(num_classes=91)
checkpoint = torch.load("deformable_detr_coco.pth", map_location="cpu")  # 预训练权重路径
model.load_state_dict(checkpoint["model"])
model.eval()

# 2. 图像预处理
img = Image.open("test.jpg").convert("RGB")
transform = T.Compose([
    T.Resize(480),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
])
img_tensor = transform(img).unsqueeze(0)

# 3. 构造 NestedTensor
from util.misc import nested_tensor_from_tensor_list
samples = nested_tensor_from_tensor_list([img_tensor])

# 4. 推理
with torch.no_grad():
    outputs = model(samples)

# 5. 后处理：取置信度最高的预测
logits = outputs["pred_logits"][0]  # [num_queries, num_classes+1]
boxes = outputs["pred_boxes"][0]    # [num_queries, 4]

probs = logits.softmax(-1)
scores, labels = probs.max(-1)

# 过滤掉背景和低置信度
keep = (labels != 91) & (scores > 0.5)

# 6. 可视化结果
fig, ax = plt.subplots(1, figsize=(12, 8))
ax.imshow(img)

for box, label, score in zip(boxes[keep], labels[keep], scores[keep]):
    # box 是归一化坐标 [cx, cy, w, h]
    cx, cy, w, h = box
    x = (cx - w/2) * img.width
    y = (cy - h/2) * img.height
    w = w * img.width
    h = h * img.height

    rect = patches.Rectangle((x, y), w, h, linewidth=2,
                             edgecolor='red', facecolor='none')
    ax.add_patch(rect)
    ax.text(x, y, f"{label.item()}:{score:.2f}", 
            bbox=dict(facecolor='yellow', alpha=0.5))

plt.show()

