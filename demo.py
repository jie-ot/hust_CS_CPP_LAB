import torch
from models.deformable_detr import DeformableDETR
from util.misc import nested_tensor_from_tensor_list
from PIL import Image
import torchvision.transforms as T

# ----------------------------------------------------
# 1. 构建 Deformable-DETR-R50（默认参数）
#    这是仓库里的最小模型，不会炸 Jetson 显存
# ----------------------------------------------------
def build_model():
    model = DeformableDETR(
        num_classes=91,
        num_queries=300,
        num_feature_levels=4,
        backbone="resnet50",
    )
    return model


# ----------------------------------------------------
# 2. 图像预处理
# ----------------------------------------------------
transform = T.Compose([
    T.Resize(480),          # Jetson 显存小，需要降低分辨率
    T.ToTensor(),
    T.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    )
])


# ----------------------------------------------------
# 3. 推理测试
# ----------------------------------------------------
@torch.no_grad()
def run_test(model, device):
    model.eval().to(device)

    img = Image.open("test.jpg").convert("RGB")
    img = transform(img)

    samples = nested_tensor_from_tensor_list([img.to(device)])

    print("Running inference on Jetson...")
    outputs = model(samples)

    print("Done!")
    print("pred_logits:", outputs["pred_logits"].shape)
    print("pred_boxes:", outputs["pred_boxes"].shape)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    model = build_model()
    run_test(model, device)

class _NewEmptyTensorOp(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, new_shape):
            ctx.shape = x.shape
            return x.new_empty(new_shape)

        @staticmethod
        def backward(ctx, grad):
            shape = ctx.shape
            return _NewEmptyTensorOp.apply(grad, shape), None
