# catkin_create_pkg hello_world roscpp rospy std_msgs
# git clone https://github.com/FoundationVision/ByteTrack

<launch>
  <arg name="config_file" default="$(find tracker)/config/tracker_config.yaml"/>
  <node pkg="tracker" type="tracker_node" name="tracker" output="screen">
    <param name="config_file" value="$(arg config_file)"/>
  </node>
</launch>


#Track.msg
std_msgs/Header header
int32 track_id
float32[4] bbox   # [x1, y1, x2, y2]
float32 score
int32 class_id

# TrackArray.msg
std_msgs/Header header
tracker/Track[] tracks



detections_topic: "/detections"
tracks_topic: "/tracks"

# ByteTracker
track_thresh: 0.5
high_thresh: 0.6
match_thresh: 0.8       
track_buffer: 30        
frame_rate: 30          

aspect_ratio_thresh: 1.6
min_box_area: 10
mot20: false



#!/usr/bin/env python3
import rospy, yaml
from std_msgs.msg import Header
from detr_detector.msg import DetectionArray
from tracker.msg import Track, TrackArray

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "third_party", "ByteTrack"))
from yolox.tracker.byte_tracker import BYTETracker
import numpy as np

class TrackerNode:
    def __init__(self):
        cfg_path = rospy.get_param("~config_file")
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)

        self.detections_topic = cfg["detections_topic"]
        self.tracks_topic = cfg["tracks_topic"]

        # 初始化ByteTrack
        self.tracker = BYTETracker(
            args=type("Args", (), {
                "track_thresh": cfg.get("track_thresh", 0.5),
                "match_thresh": cfg.get("match_thresh", 0.8),
                "track_buffer": cfg.get("track_buffer", 30),
                "aspect_ratio_thresh": cfg.get("aspect_ratio_thresh", 1.6),
                "min_box_area": cfg.get("min_box_area", 10),
                "mot20": cfg.get("mot20", False),
                "high_thresh": cfg.get("high_thresh", 0.6)
            })(),
            frame_rate=cfg.get("frame_rate", 30)
        )

        self.sub = rospy.Subscriber(self.detections_topic, DetectionArray, self.det_cb, queue_size=10)
        self.pub = rospy.Publisher(self.tracks_topic, TrackArray, queue_size=10)

        self.frame_id = 0
        rospy.loginfo("tracker节点启动")

    def det_cb(self, msg: DetectionArray):
        detections = []
        for det in msg.detections:
            x1, y1, x2, y2 = det.bbox
            detections.append([float(x1), float(y1), float(x2), float(y2), float(det.score), int(det.class_id)])

        if len(detections) > 0:
            dets_np = np.array(detections, dtype=np.float32)   # shape:(N,6)
        else:
            dets_np = np.empty((0, 6), dtype=np.float32)

        img_info = (int(msg.height), int(msg.width))
        img_size = (int(msg.height), int(msg.width))

        online_targets = self.tracker.update(dets_np, img_info, img_size)
        self.frame_id += 1

        arr = TrackArray()
        arr.header = Header(stamp=msg.header.stamp, frame_id=msg.header.frame_id)

        # 发布跟踪结果
        for i, t in enumerate(online_targets):
            bbox = t.tlbr

            track_msg = Track()
            track_msg.header = Header(stamp=msg.header.stamp, frame_id=msg.header.frame_id)
            track_msg.track_id = int(getattr(t, "track_id", -1))
            track_msg.bbox = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
            track_msg.score = float(getattr(t, "score", 0.0))
            track_msg.class_id = int(getattr(t, "cls", 0))

            arr.tracks.append(track_msg)

        self.pub.publish(arr)

if __name__ == "__main__":
    rospy.init_node("tracker")
    TrackerNode()
    rospy.spin()

