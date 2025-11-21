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
