<launch>
  <arg name="config_file" default="$(find byte_tracker_ros)/config/tracker_config.yaml"/>

  <node pkg="tracker" type="tracker_node.py" name="tracker" output="screen">
    <param name="config_file" value="$(arg config_file)"/>
  </node>
</launch>



std_msgs/Header header
int32 track_id
float32[4] bbox   # [x1, y1, x2, y2]
float32 score
int32 class_id 



image_topic: "/camera/image_color"
detection_topic: "/detections"

tracks_topic: "/tracks"

# BYTETracker
track_thresh: 0.5       # High detection threshold for tracking start
match_thresh: 0.8       # Association IOU threshold
track_buffer: 30        # Frames to keep lost tracks
frame_rate: 30          # FPS of input stream

aspect_ratio_thresh: 1.6
min_box_area: 10
mot20: false

# 聚合与同步策略
aggregation_timeout_ms: 50  # 每帧检测聚合的时间窗口，按 stamp 聚合
max_queue_size: 100         # 缓存最大条目数
