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

