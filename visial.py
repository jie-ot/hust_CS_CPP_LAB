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

        # 使用message_filters同步图像和跟踪结果
        image_sub = message_filters.Subscriber("/camera/image_raw", Image)
        tracks_sub = message_filters.Subscriber("/tracks", TrackArray)

        # ApproximateTimeSynchronizer可以容忍一定时间差
        ts = message_filters.ApproximateTimeSynchronizer(
            [image_sub, tracks_sub],
            queue_size=10,
            slop=0.05,   # 允许的时间差，单位秒
            allow_headerless=False
        )
        ts.registerCallback(self.sync_cb)

        self.image_pub = rospy.Publisher("/tracker_vis", Image, queue_size=1)

        rospy.loginfo("visualizer节点启动")

    def sync_cb(self, img_msg, tracks_msg):
        # 转换图像
        cv_img = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")

        # 绘制轨迹框
        for t in tracks_msg.tracks:
            x1, y1, x2, y2 = map(int, t.bbox)
            cv2.rectangle(cv_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(cv_img, f"ID:{t.track_id} Score:{t.score:.2f}",
                        (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 1)

        # 发布可视化图像
        vis_msg = self.bridge.cv2_to_imgmsg(cv_img, "bgr8")
        vis_msg.header = img_msg.header  
        self.image_pub.publish(vis_msg)

if __name__ == "__main__":
    rospy.init_node("tracker_visualizer")
    VisualizerNode()
    rospy.spin()
