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
