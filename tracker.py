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



detections_topic: "/detections"
tracks_topic: "/tracks"

# ByteTracker
track_thresh: 0.5       
match_thresh: 0.8       
track_buffer: 30        
frame_rate: 30          

aspect_ratio_thresh: 1.6
min_box_area: 10
mot20: false

