# 模型权重路径
checkpoint: "$(find detr_detector)/models/r50_deformable_detr_single_scale-checkpoint.pth"

# 基本参数
num_classes: 91
num_queries: 300
num_feature_levels: 1
hidden_dim: 256
backbone: "resnet50"

#短边缩放到480
resize_short: 480
#置信度
score_th: 0.5

# 输入话题
input_topic: "/camera/image_raw"




std_msgs/Header header
float32[4] bbox   # [x1, y1, x2, y2]
float32 score
int32 class_id


std_msgs/Header header
int32 width
int32 height
detr_detector/Detection[] detections


import torch
import torchvision.transforms as T
from PIL import Image
from util.misc import nested_tensor_from_tensor_list

# 导入原仓库里的模块
from models.backbone import build_backbone
from models.deformable_transformer import build_deforamble_transformer
from models.deformable_detr import DeformableDETR

class DeformableDETRRunner:
    def __init__(self, cfg):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = cfg["num_classes"]
        self.num_queries = cfg["num_queries"]
        self.num_feature_levels = cfg["num_feature_levels"]
        self.hidden_dim = cfg["hidden_dim"]
        self.backbone_name = cfg["backbone"]
        self.resize_short = cfg["resize_short"]
        self.score_th = cfg["score_th"]
        self.checkpoint = cfg["checkpoint"]

        # 固定的模型结构参数直接写在这里
        import argparse
        args = argparse.Namespace(
            dataset_file="coco",
            num_classes=self.num_classes,
            backbone=self.backbone_name,
            hidden_dim=self.hidden_dim,
            num_queries=self.num_queries,
            num_feature_levels=self.num_feature_levels,
            dilation=False,
            position_embedding="sine",
            enc_layers=6, dec_layers=6, dim_feedforward=1024, dropout=0.1, nheads=8,
            device="cuda", lr=1e-4, masks=False, lr_backbone=1e-5, batch_size=2, epochs=50,
            output_dir="exps/r50_deformable_detr_single_scale",
            enc_n_points=4, dec_n_points=4, two_stage=False, dec_layer_share=False, pretrained=False
        )

        backbone = build_backbone(args)
        transformer = build_deforamble_transformer(args)
        self.model = DeformableDETR(
            backbone=backbone,
            transformer=transformer,
            num_classes=self.num_classes,
            num_queries=self.num_queries,
            num_feature_levels=self.num_feature_levels
        )
        self.model.to(self.device)
        self.model.eval()

        # 加载权重
        ckpt = torch.load(self.checkpoint, map_location=self.device)
        state = ckpt.get("model", ckpt.get("state_dict", ckpt))
        self.model.load_state_dict(state, strict=False)
        print(f"[DeformableDETRRunner] 模型已成功加载，使用设备: {self.device}")

        # 预处理
        self.transform = T.Compose([
            T.Resize(self.resize_short),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])

    def infer_image(self, pil_img):
        img_t = self.transform(pil_img).to(self.device)
        samples = nested_tensor_from_tensor_list([img_t])

        with torch.no_grad():
            outputs = self.model(samples)

        logits = outputs["pred_logits"][0].cpu()
        boxes = outputs["pred_boxes"][0].cpu()

        probs = logits.softmax(-1)
        scores, labels = probs.max(-1)
        bg_idx = self.num_classes
        keep = (labels != bg_idx) & (scores > self.score_th)

        kept_boxes = boxes[keep].numpy()
        kept_labels = labels[keep].numpy()
        kept_scores = scores[keep].numpy()

        resized = T.Resize(self.resize_short)(pil_img)
        W, H = resized.size

        dets = []
        for (cx, cy, w, h), lab, sc in zip(kept_boxes, kept_labels, kept_scores):
            x1 = (cx - w/2) * W
            y1 = (cy - h/2) * H
            x2 = (cx + w/2) * W
            y2 = (cy + h/2) * H
            dets.append([x1, y1, x2, y2, float(sc), int(lab)])
        return dets, W, H




#!/usr/bin/env python3
import rospy, yaml
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from PIL import Image as PILImage
from std_msgs.msg import Header
from detr_detector.msg import Detection
from infer_detr import DeformableDETRRunner

class DetrDetectorNode:
    def __init__(self):
        cfg_path = rospy.get_param("~config_file")
        with open(cfg_path,"r") as f:
            cfg = yaml.safe_load(f)
        self.runner = DeformableDETRRunner(cfg)
        self.bridge = CvBridge()
        self.sub = rospy.Subscriber(cfg["input_topic"], Image, self.image_cb, queue_size=1)
        self.pub = rospy.Publisher("/detections", Detection, queue_size=10)
        rospy.loginfo("detector节点启动")


    def image_cb(self,msg):
        cv_img = self.bridge.imgmsg_to_cv2(msg,"bgr8")
        pil_img = PILImage.fromarray(cv_img[:,:,::-1])
        dets = self.runner.infer_image(pil_img)
        for x1,y1,x2,y2,score,cls in dets:
            d = Detection()
            d.header = Header(stamp=msg.header.stamp, frame_id=msg.header.frame_id)
            d.bbox = [x1,y1,x2,y2]
            d.score = score
            d.class_id = cls
            self.pub.publish(d)

if __name__=="__main__":
    rospy.init_node("detr_detector")
    DetrDetectorNode()
    rospy.spin()



修改：（1）添加成功加载模型的打印信息 （2）添加detector节点启动的打印信息 （3）添加一个新的msg文件DetectionArray.msg，并记得在CMakeLists.txt中把 DetectionArray.msg 加入 add_message_files
    （4）在detector_node.py中进行如下修改：
        from detr_detector.msg import Detection, DetectionArray
        self.pub = rospy.Publisher("/detections", DetectionArray, queue_size=10)

    def image_cb(self,msg):
        cv_img = self.bridge.imgmsg_to_cv2(msg,"bgr8")
        pil_img = PILImage.fromarray(cv_img[:,:,::-1])
        dets, W, H = self.runner.infer_image(pil_img)

        arr = DetectionArray()
        arr.header = Header(stamp=msg.header.stamp, frame_id=msg.header.frame_id)
        arr.width = W
        arr.height = H

        for x1,y1,x2,y2,score,cls in dets:
            d = Detection()
            d.header = arr.header
            d.bbox = [x1,y1,x2,y2]
            d.score = score
            d.class_id = cls
            arr.detections.append(d)

        self.pub.publish(arr)
    （5）添加了infer_image的返回值，返回图像尺寸，同时回调函数中要发布尺寸




<launch>
  <arg name="config_file" default="$(find detr_detector)/config/detector.yaml"/>
  <node pkg="detr_detector" type="detector_node.py" name="detr_detector" output="screen">
    <param name="config_file" value="$(arg config_file)"/>
  </node>
</launch>
