<launch>
  <node pkg="tracker_visualizer" type="visualizer_node" name="tracker_visualizer" output="screen"/>
</launch>




#!/usr/bin/env python3
import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from tracker.msg import TrackArray
import message_filters

class VisualizerNode:
    def __init__(self):
        self.bridge = CvBridge()
        image_sub = message_filters.Subscriber("/camera/image_raw", Image, queue_size=10)
        tracks_sub = message_filters.Subscriber("/tracks", TrackArray, queue_size=10)
        ts = message_filters.ApproximateTimeSynchronizer([image_sub, tracks_sub], queue_size=10, slop=0.05)
        ts.registerCallback(self.sync_cb)
        self.image_pub = rospy.Publisher("/tracker_vis", Image, queue_size=10)

    def sync_cb(self, img_msg, tracks_msg):
        cv_img = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        orig_h, orig_w = cv_img.shape[:2]
        track_w = tracks_msg.width
        track_h = tracks_msg.height
        sx = orig_w / float(track_w)
        sy = orig_h / float(track_h)

        for t in tracks_msg.tracks:
            x1 = int(t.bbox[0] * sx)
            y1 = int(t.bbox[1] * sy)
            x2 = int(t.bbox[2] * sx)
            y2 = int(t.bbox[3] * sy)
            color = (int((t.track_id * 37) % 255), int((t.track_id * 91) % 255), int((t.track_id * 59) % 255))
            cv2.rectangle(cv_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(cv_img, f"ID:{t.track_id} cls:{t.class_id}", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        vis_msg = self.bridge.cv2_to_imgmsg(cv_img, "bgr8")
        vis_msg.header = img_msg.header
        self.image_pub.publish(vis_msg)

if __name__ == "__main__":
    rospy.init_node("tracker_visualizer")
    VisualizerNode()
    rospy.spin()
