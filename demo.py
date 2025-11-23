python3 --version
python3 -m venv deformable_detr
source deformable_detr/bin/activate
deactivate

pip3 install torch==2.1.0a0+41361538.nv23.6
pip3 install torchvision==0.16.0+fbb4cc5
pip3 install numpy==1.24.4
pip3 install onnx==1.15.0
pip3 install pillow matplotlib opencv-python
pip3 install pycocotools tqdm cython scipy
       
import torch
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse

from models.backbone import build_backbone
from models.deformable_transformer import build_deforamble_transformer
from models.deformable_detr import DeformableDETR
from util.misc import nested_tensor_from_tensor_list

CHECKPOINT = "/home/nx/Deformable-DETR/r50_deformable_detr_single_scale-checkpoint.pth"   
IMAGE_PATH = "/home/nx/Deformable-DETR/test2.jpg"  
OUT_PATH = "/home/nx/Deformable-DETR/infer2.jpg"
NUM_CLASSES = 91
NUM_QUERIES = 300
NUM_FEATURE_LEVELS = 1 
BACKBONE = "resnet50"
HIDDEN_DIM = 256
RESIZE_SHORT = 480
SCORE_TH = 0.5

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    args = argparse.Namespace(
    	dataset_file="coco", 
    	num_classes=91,

    	#模型结构
    	backbone="resnet50",
    	hidden_dim=256,
    	num_queries=300,
    	num_feature_levels=1,
    	dilation=False,
    	position_embedding="sine",

    	#Transformer结构
    	enc_layers=6,
    	dec_layers=6,
    	dim_feedforward=1024,
    	dropout=0.1,
    	nheads=8,

    	#设备
    	device="cuda",
	
    	#训练相关参数
    	lr=1e-4,
    	masks=False,
    	lr_backbone=1e-5,
    	batch_size=2,
    	epochs=50,
    	output_dir="exps/r50_deformable_detr_single_scale",
    	enc_n_points=4,
    	dec_n_points=4,
    	two_stage=False,
    	dec_layer_share=False,
    	pretrained=False
    )

    backbone = build_backbone(args)
    transformer = build_deforamble_transformer(args)
    
    model = DeformableDETR(
    	backbone=backbone,
    	transformer=transformer,
    	num_classes=NUM_CLASSES,
    	num_queries=NUM_QUERIES,
    	num_feature_levels=NUM_FEATURE_LEVELS
    )

    model.to(device)
    model.eval()

    # 加载 checkpoint
    ckpt = torch.load(CHECKPOINT, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt

    # 加载权重
    model.load_state_dict(state, strict=False)
    print("Loaded checkpoint:", CHECKPOINT)

    # 读图并 preprocess（Resize -> ToTensor -> Normalize）
    img = Image.open(IMAGE_PATH).convert("RGB")
    transform = T.Compose([
        T.Resize(RESIZE_SHORT),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225])
    ])
    img_t = transform(img).to(device)  # [C,H,W]

    # nested tensor
    samples = nested_tensor_from_tensor_list([img_t])

    # 推理, 启用 autocast 省显存
    with torch.no_grad():
        if device.type == "cuda":
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.float32):
                outputs = model(samples)
        else:
            outputs = model(samples)

    # 取 outputs 的 pred_logits / pred_boxes
    logits = outputs["pred_logits"][0].cpu()  # [num_queries, num_classes+1]
    boxes = outputs["pred_boxes"][0].cpu()    # [num_queries, 4] 归一化 [cx,cy,w,h] 相对 resized 输入

    probs = logits.softmax(-1)
    scores, labels = probs.max(-1)

    # background 索引通常是 num_classes
    bg_idx = NUM_CLASSES
    keep = (labels != bg_idx) & (scores > SCORE_TH)

    kept_boxes = boxes[keep].numpy()
    kept_labels = labels[keep].numpy()
    kept_scores = scores[keep].numpy()

    resized = T.Resize(RESIZE_SHORT)(Image.open(IMAGE_PATH).convert("RGB"))
    W, H = resized.size

    # 把归一化 box 映回像素
    rects = []
    for (cx, cy, w, h) in kept_boxes:
        x = (cx - w/2) * W
        y = (cy - h/2) * H
        ww = w * W
        hh = h * H
        rects.append((x, y, ww, hh))

    # 可视化并保存
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(resized)
    for (x, y, ww, hh), lab, sc in zip(rects, kept_labels, kept_scores):
        rect = patches.Rectangle((x, y), ww, hh, linewidth=2, edgecolor="r", facecolor="none")
        ax.add_patch(rect)
        ax.text(x, y, f"{int(lab)}:{sc:.2f}", bbox=dict(facecolor="yellow", alpha=0.5))
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", OUT_PATH)
    print("Detections:", len(rects))

if __name__ == "__main__":
    main()
