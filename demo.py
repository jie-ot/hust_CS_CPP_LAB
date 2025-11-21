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
       
Reading stateThe following packages were automatically installed and are no longer required:
  libcmis-0.5-5v5 libgpgmepp6 libjuh-java libjurt-java liblibreoffice-java
  libmwaw-0.3-3 libneon27-gnutls liborcus-0.15-0 libreoffice-style-tango
  libridl-java libunoloader-java libwps-0.4-4 ure-java
Use 'sudo apt autoremove' to remove them.
The following security updates require Ubuntu Pro with 'esm-infra' enabled:
  libgstreamer-plugins-base1.0-dev libsoup-gnome2.4-1 libopenjp2-7
  poppler-utils gstreamer1.0-alsa libcups2 libprotoc-dev linux-libc-dev
  xserver-common libpoppler-dev libxml2-utils libpython3.8-dev gir1.2-soup-2.4
  gstreamer1.0-plugins-base-apps openssl libblockdev-swap2 ruby2.7
  xserver-xorg-core libprotoc17 gir1.2-gdkpixbuf-2.0 libgdk-pixbuf2.0-0
  libssh-4 libpython3.8-minimal libsqlite3-dev libwbclient0 git-man
  libmysqlclient-dev libsystemd0 gcc-10-base gstreamer1.0-plugins-good libgs9
  python2.7-minimal libsqlite3-0 python3-protobuf python3-urllib3 bind9-host
  libitm1 libcgraph6 libtiff-dev sudo libpython2.7 python2.7 python3-pip
  libpython3.8 python3.8 xserver-xorg-legacy git libblockdev-crypto2 udev
  gstreamer1.0-plugins-base libblockdev-loop2 libblockdev-fs2
  libgstreamer-plugins-good1.0-dev libblockdev-part2 python3-requests
  liblab-gamut1 libgstreamer-plugins-good1.0-0 libudev1 libsoup2.4-1
  gstreamer1.0-pulseaudio systemd-timesyncd libpoppler-private-dev libgcc1
  samba-libs xserver-xephyr protobuf-compiler gstreamer1.0-gtk3 libpmix2
  libtiff5 udisks2 libtsan0 libubsan1 libruby2.7 libprotobuf-lite17
  libgfortran5 libcupsfilters1 python3.8-minimal libgstreamer-gl1.0-0
  systemd-sysv libblockdev2 libxml2-dev libpam-systemd
  libgstreamer-plugins-base1.0-0 libcdt5 xwayland gstreamer1.0-x ghostscript
  liblsan0 libpathplan4 systemd libgomp1 libgdk-pixbuf2.0-bin libssh-gcrypt-4
  gir1.2-gst-plugins-base-1.0 libssl-dev libblockdev-utils2 ghostscript-x
  libgvpr2 libgdk-pixbuf2.0-common libsmbclient libgdk-pixbuf2.0-dev
  libmysqlclient21 libnss-systemd libgs9-common libblockdev-part-err2
  libgcc-s1 libxml2 libpython2.7-minimal libpython3.8-stdlib libgnutls30
  libudisks2-0 python3.8-dev libatomic1 libssl1.1 libcc1-0 libgvc6
  libprotobuf17 libcupsimage2 libpython2.7-stdlib libpoppler-glib8 libstdc++6
  libpoppler97 python3-scipy libprotobuf-dev libopenjp2-7-dev graphviz
  python-pip-whl bind9-libs gstreamer1.0-gl libtiffxx5 libxslt1.1
Learn more about Ubuntu Pro at https://ubuntu.com/pro
The following NEW packages will be installed:
  ca-certificates-java default-jre default-jre-headless fonts-dejavu-extra
  java-common libatk-wrapper-java libatk-wrapper-java-jni liblibreoffice-java
  libreoffice-style-yaru openjdk-11-jre openjdk-11-jre-headless ure-java
The following packages have been kept back:
  gir1.2-gtk-2.0
The following packages will be upgraded:
  fonts-opensymbol libaom0 libglib2.0-tests libjuh-java libjurt-java
  libpython2.7 libpython2.7-minimal libpython2.7-stdlib librealsense2
  librealsense2-dbg librealsense2-dev librealsense2-gl librealsense2-gl-dbg
  librealsense2-gl-dev librealsense2-udev-rules librealsense2-utils
  libreoffice-base-core libreoffice-calc libreoffice-common libreoffice-core
  libreoffice-draw libreoffice-gnome libreoffice-gtk3 libreoffice-impress
  libreoffice-math libreoffice-pdfimport libreoffice-style-breeze
  libreoffice-style-colibre libreoffice-style-elementary
  libreoffice-style-tango libreoffice-writer libridl-java libtar-dev libtar0
  libuno-cppu3 libuno-cppuhelpergcc3-3 libuno-purpenvhelpergcc3-3 libuno-sal3
  libuno-salhelpergcc3-3 libunoloader-java python-pip-whl python2.7
  python2.7-minimal python3-pip python3-uno tailscale uno-libs-private ure

# detr_infer_jetson.py
import os
import torch
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from models.deformable_detr import DeformableDETR
from util.misc import nested_tensor_from_tensor_list

#CONFIG
CHECKPOINT_PATH = "/home/ubuntu/checkpoints/deformable_detr_r50.pth"
IMAGE_PATH = "/home/ubuntu/images/test.jpg" 
OUT_PATH = "/home/ubuntu/images/out_infer.jpg"
NUM_CLASSES = 91 
RESIZE_SHORT_SIDE = 480 
SCORE_THR = 0.5

def load_checkpoint_to_model(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    # 兼容不同checkpoint格式
    if isinstance(ckpt, dict):
        # 常见 key: "model", "state_dict"
        if "model" in ckpt:
            state = ckpt["model"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt
    else:
        state = ckpt
    model.load_state_dict(state, strict=False)
    return model

def preprocess_image(pil_img, resize_short=RESIZE_SHORT_SIDE):
    # 返回两个：PIL resized (用于保存/可视化) 与normalize后的tensor
    resized_pil = T.Resize(resize_short)(pil_img)  # 保持长宽比，短边为resize_short
    # 转为tensor并归一
    to_tensor = T.ToTensor()
    img_t = to_tensor(resized_pil)  # C,H,W, float32 [0,1]
    normalize = T.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    img_t = normalize(img_t)
    img_batch = img_t.unsqueeze(0)  # 1,C,H,W
    return resized_pil, img_batch

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # 构建模型
    model = DeformableDETR(num_classes=NUM_CLASSES)
    model.to(device)
    model.eval()

    # 加载checkpoint
    print("Loading checkpoint:", CHECKPOINT_PATH)
    model = load_checkpoint_to_model(model, CHECKPOINT_PATH, device)
    model.to(device)
    model.eval()

    # 读图并预处理
    img = Image.open(IMAGE_PATH).convert("RGB")
    resized_pil, img_batch = preprocess_image(img, RESIZE_SHORT_SIDE)

    # 将输入移到device
    img_batch = img_batch.to(device)

    # 构造NestedTensor
    samples = nested_tensor_from_tensor_list([img_batch])

    # 推理
    with torch.no_grad():
        # 在Jetson上启用autocast(fp16)可以显著减显存和提速
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=torch.float16):
            outputs = model(samples)

    # 后处理
    logits = outputs["pred_logits"][0].cpu()  # [num_queries, num_classes+1]
    boxes = outputs["pred_boxes"][0].cpu()    # [num_queries, 4] 归一化的 [cx,cy,w,h] 相对 resized 输入

    probs = logits.softmax(-1)
    scores, labels = probs.max(-1)

    bg_index = NUM_CLASSES
    keep = (labels != bg_index) & (scores > SCORE_THR)

    kept_boxes = boxes[keep].numpy()
    kept_labels = labels[keep].numpy()
    kept_scores = scores[keep].numpy()

    # 把 box 从归一化 [cx,cy,w,h] 映射到像素（针对 resized_pil 的尺寸）
    W, H = resized_pil.size  # PIL: (width, height)
    rects = []
    for (cx, cy, w, h) in kept_boxes:
        x = (cx - w/2) * W
        y = (cy - h/2) * H
        ww = w * W
        hh = h * H
        rects.append((x, y, ww, hh))

    # 可视化并保存
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(resized_pil)
    for (x,y,ww,hh), lab, sc in zip(rects, kept_labels, kept_scores):
        rect = patches.Rectangle((x, y), ww, hh, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y, f"{int(lab)}:{sc:.2f}", fontsize=10, bbox=dict(facecolor='yellow', alpha=0.5))
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved visualization to:", OUT_PATH)
    print("Kept detections:", len(rects))

if __name__ == "__main__":
    main()
