#!/usr/bin/env python

import rospy
import subprocess
from sensor_msgs.msg import Image, PointCloud2
import threading
import signal
import sys
import termios
import tty

# Define global variables to store subprocesses and threads
subprocesses = []
threads = []
publishing_status = {}  # Dictionary to track publishing status for each topic
terminate_flag = False

def start_subprocess(command):
    
    process = subprocess.Popen(command)
    subprocesses.append(process)

def stop_subprocesses():
    rospy.logwarn("Stopping all subprocesses...")
    for process in subprocesses:
        process.terminate()
        process.wait()

def check_topic_publishing(topic_name, message_type, timeout):
    while not rospy.is_shutdown():
        try:
            rospy.wait_for_message(topic_name, message_type, timeout=timeout)
            publishing_status[topic_name] = True
        except rospy.ROSException:
            publishing_status[topic_name] = False
            rospy.logerr(f"Topic {topic_name} is not publishing!")

def input_thread():
    global terminate_flag
    while not terminate_flag:
        char = sys.stdin.read(1)
        if char == 'X' or char == 'x':
            stop_subprocesses()
            rospy.logwarn("Terminating MASTER")
            
            rospy.signal_shutdown("X pressed")
            break

def thread_function():
    global terminate_flag
    try:
        # Start fps.py as a subprocess
        start_subprocess(["rosrun", "dynamix", "fps.py"])

        # Wait for fps.py to start publishing to /lowfps
        while not rospy.is_shutdown():
            try:
                rospy.wait_for_message("/lowfps", Image, timeout=1)
                rospy.loginfo("Fps_dropper has started publishing ===> /lowfps")
                publishing_status["/lowfps"] = True
                break  # Exit the loop if message received
            except rospy.ROSException:
                rospy.logwarn_once("Waiting for FPS_dropper to start publishing to /lowfps...")

        # Once fps.py is publishing, start semseg.py
        start_subprocess(["rosrun", "dynamix", "semseg.py"])

        while not rospy.is_shutdown():
            try:
                rospy.wait_for_message("/zed2i/depth_masked_image", Image, timeout=1)
                rospy.loginfo("Semseg has started publishing ===> /zed2i/depth_masked_image")
                publishing_status["/zed2i/depth_masked_image"] = True
                break  # Exit the loop if message received
            except rospy.ROSException:
                rospy.logwarn_once("Waiting for Semseg to start publishing...")

        # Start the point cloud code
        start_subprocess(["rosrun", "dynamix", "pointcloud"])

        while not rospy.is_shutdown():
            try:
                rospy.wait_for_message("/obstacle_points", PointCloud2, timeout=1)
                rospy.loginfo("Pointcloud has started publishing ===> /obstacle_points ")
                publishing_status["/obstacle_points"] = True
                break  # Exit the loop if message received
            except rospy.ROSException:
                rospy.logwarn_once("Waiting for Masked_pointcloud to start publishing...")

        # Start background threads to monitor topic publishing status
        monitoring_threads = []
        for topic, _ in publishing_status.items():
            if(topic == "/obstacle_points"):
                topic_type = PointCloud2
            else :
                topic_type = Image
                _
            thread = threading.Thread(target=check_topic_publishing, args=(topic, topic_type, 2))
            thread.start()
            monitoring_threads.append(thread)

        # Start the input thread to check for 'X' key
        input_thread_thread = threading.Thread(target=input_thread)
        input_thread_thread.start()

        for thread in monitoring_threads:
            thread.join()

        # Terminate the input thread
        terminate_flag = True
        input_thread_thread.join()

    except KeyboardInterrupt:
        # Handle Ctrl+C to gracefully stop subprocesses
        rospy.loginfo("Received Ctrl+C, stopping subprocesses...")
        stop_subprocesses()

def main():
    rospy.init_node('MASTER')
    thread = threading.Thread(target=thread_function)
    thread.start()

    # Register a signal handler to stop subprocesses on Ctrl+C
    def signal_handler(sig, frame):
        rospy.loginfo("Received Ctrl+C, stopping subprocesses...")
        stop_subprocesses()
        thread.join()
        rospy.signal_shutdown("Ctrl+C")

    signal.signal(signal.SIGINT, signal_handler)

    rospy.spin()

if __name__ == '__main__':
    # Set terminal to non-blocking input mode
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    tty.setcbreak(fd)

    main()

    # Restore terminal settings on exit
    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
