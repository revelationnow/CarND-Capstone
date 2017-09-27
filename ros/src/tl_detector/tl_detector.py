#!/usr/bin/env python2
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight, NextLight
from styx_msgs.msg import Lane, Waypoint
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir + '/light_classification' )

from tl_classifier import TLClassifier
from tl_identifier import TLIdentifier
from matplotlib.pyplot import imsave
import tf
from os.path import expanduser
import cv2
import yaml
import math

STATE_COUNT_THRESHOLD = 1
MANUAL = False

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []
        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = 999999
        self.state_count = 0
        self.light_cnt = 999
        self.bridge = CvBridge()
        self.light_identifier = TLIdentifier(1,'ALL','./light_classification/traffic_light_identifier.h5')
        self.light_classifier = TLClassifier('./light_classification/traffic_light_classifier.h5')
        self.light_identifier.set_window(0,800,100,600)
        self.listener = tf.TransformListener()

        self.debug_success_counter = 0
        self.debug_fail_counter = 0

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', NextLight, queue_size=1)


        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg
        return

    def waypoints_cb(self, waypoints):
        if (waypoints):
            self.waypoints = waypoints
	return

    def traffic_cb(self, msg):
        self.lights = msg.lights
        return

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg

        light_wp, state = self.process_traffic_lights()

        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            #light_wp = light_wp if state == TrafficLight.RED else 999999
            self.last_wp = light_wp
            publish_data = NextLight()
            publish_data.state = state
            publish_data.waypoint = light_wp
            self.upcoming_red_light_pub.publish(publish_data)
        else:
            publish_data = NextLight()
            publish_data.state = self.last_state
            publish_data.waypoint = self.last_wp
            self.upcoming_red_light_pub.publish(publish_data)

        self.state_count += 1
        return

    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """

        best_dist = 9999
        best_wp = 0

        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(len(self.waypoints.waypoints)):
            dist = dl(pose.position, self.waypoints.waypoints[i].pose.pose.position)
            if dist < best_dist:
                best_dist = dist
                best_wp = i

        return best_wp

    def get_next_waypoint(self, pose):

        next_wp = self.get_closest_waypoint(pose)

        map_x = self.waypoints.waypoints[next_wp].pose.pose.position.x
        map_y = self.waypoints.waypoints[next_wp].pose.pose.position.y

        heading = math.atan2( (map_y - pose.position.y),(map_x - pose.position.x) )
        angle = abs(math.atan2(pose.position.y,pose.position.x) - heading)
        if angle > math.pi/4:
            next_wp += 1
        try:
            self.waypoints.waypoints[next_wp].pose.pose.position.x
        except:
            next_wp = 0

        return next_wp

    def project_to_image_plane(self, point_in_world):

        """Project point from 3D world coordinates to 2D camera image location
        Args:
            point_in_world (Point): 3D location of a point in the world

        Returns:
            x (int): x coordinate of target point in image
            y (int): y coordinate of target point in image

        """

        fx = self.config['camera_info']['focal_length_x']
        fy = self.config['camera_info']['focal_length_y']
        image_width = self.config['camera_info']['image_width']
        image_height = self.config['camera_info']['image_height']

        # get transform between pose of camera and world frame
        trans = None
        try:
            now = rospy.Time.now()
            self.listener.waitForTransform("/base_link",
                  "/world", now, rospy.Duration(1.0))
            (trans, rot) = self.listener.lookupTransform("/base_link",
                  "/world", now)

        except (tf.Exception, tf.LookupException, tf.ConnectivityException):
            rospy.logerr("Failed to find camera to map transform")


        #transform
        x_t = point_in_world.x - trans[0]
        y_t = point_in_world.y - trans[1]
        z_t = point_in_world.z - trans[2]

        #rotate
        x = x_t*float(rot[0])+x_t*float(rot[1])+x_t*float(rot[2])+x_t*float(rot[3])
        y = y_t*float(rot[0])+y_t*float(rot[1])+y_t*float(rot[2])+y_t*float(rot[3])
        z = z_t*float(rot[0])+z_t*float(rot[1])+z_t*float(rot[2])+z_t*float(rot[3])

        rospy.loginfo("Co-ordinates are %s %s %s",x,y,z)

        return (x*fx/z, y*fy/z)




    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        if(not self.has_image):
            self.prev_light_loc = None
            return False

	# Get Image
	self.camera_image.encoding = "rgb8"
        try:
            cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "rgb8")
        except CvBridgeError as e:
            print(e)
	#x, y = self.project_to_image_plane(light.position)
 	#cv_image =  cv_image[x-15:x+15,y-35:y+35]


	if MANUAL:
		# If manual or data gathering mode get image ..state is known

                dirname  = ""
        	state = self.lights[self.light_cnt].state
                found , light_image = self.light_identifier.process_image(cv_image)
		if state == 3:
                    state = self.light_identifier.color
		file_name = str(self.lights[self.light_cnt].header.seq)+"_"+str(self.lights[self.light_cnt].header.stamp)+"_"+str(self.state)+".jpg"

		if found is True:
                        self.debug_success_counter += 1
                        state_try = self.light_classifier.get_classification(light_image)
                        rospy.loginfo("Ground Truth Light State  %s", state)
                        rospy.loginfo("Light State Predicted  %s",state_try)

                        if state_try == state:
                            rospy.loginfo("SUCCESS Identified and Classified Traffic Light.. %f Success rate",100*self.debug_success_counter/(self.debug_success_counter+self.debug_fail_counter))
                        else:
                            rospy.loginfo(" FAILURE Traffic Light Classification Test Failed.. %f",self.debug_fail_counter)

   	                if state == TrafficLight.RED:
		            dirname = expanduser("~")+'/catkin_ws/src/CarND-Capstone/ros/src/tl_detector/light_classification/images/training/red'
		        elif state == TrafficLight.GREEN:
		            dirname = expanduser("~")+'/catkin_ws/src/CarND-Capstone/ros/src/tl_detector/light_classification/images/training/green'
		        else:
		            dirname = expanduser("~")+'/catkin_ws/src/CarND-Capstone/ros/src/tl_detector/light_classification/images/training/yellow'


                if found is False:
                        self.debug_fail_counter += 1
                        rospy.loginfo(" FAILURE Traffic Light Identification Test Failed.. %f", self.debug_fail_counter)
			dirname = expanduser("~")+'/catkin_ws/src/CarND-Capstone/ros/src/tl_detector/light_classification/images/training/'
 		imsave(os.path.join( dirname,file_name),light_image)

	else:

                #Identify Light
                found,light_image = self.light_identifier.process_image(cv_image)

		if found == True:
			#Get light classification
			state = self.light_classifier.get_classification(light_image)
                        rospy.loginfo(" SUCCESS Traffic Light Classification done as Preliminary analysis %s .. Final Classification %s",self.light_identifier.color,state)
                else:
                        state = TrafficLight.UNKNOWN
                        rospy.loginfo(" FAIL Traffic Light Identification Test Failed.. ")
	return state #self.light_classifier.get_classification(cv_image)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (999999 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        light_positions = self.config['stop_line_positions']
        light = None
        self.light_cnt = None
        best_dist = 150

        if (self.pose) and (self.waypoints) :

            # Get the closest waypoint ahead of the car. Not a compulsory step -
            # can use the car pose instead but this mkes things easier later..
            car_position = self.get_next_waypoint(self.pose.pose)

            dl = lambda a, b: math.sqrt((a[0]-b.x)**2 + (a[1]-b.y)**2 )

            # Get the closest light ahead
            for light_no in range(len(light_positions)):
               if light_positions[light_no][0] >= self.waypoints.waypoints[car_position].pose.pose.position.x:
    	            dist = dl(light_positions[light_no],self.waypoints.waypoints[car_position].pose.pose.position)
                    #rospy.loginfo("Light not %s at %s has distance %s and state %s",light_no,light_positions[light_no][0],dist,self.lights[light_no].state)

	            if dist < best_dist:
		        best_dist = dist
                        self.light_cnt = light_no


        # get the state
        if self.light_cnt is not None:

            rospy.loginfo("Closes Light Ahead detected (from closest wayoint ahead) )%s",self.light_cnt)
            rospy.loginfo("Distance to closes Waypoint from Light ahead is %s",best_dist)
            rospy.loginfo("LIGHT PROCESSING START: Approaching Light ahead... %s,%s",light_positions[self.light_cnt][0],light_positions[self.light_cnt][1])

   	    # create the light pose
            pose = Pose()
	    pose.position.x = light_positions[self.light_cnt][0]
	    pose.position.y = light_positions[self.light_cnt][1]
	    light = pose

            light_wp = self.get_closest_waypoint(light)
            '''
            if best_dist >  75:
                self.light_identifier.counter = 1
            elif 75 < best_dist  < 50 :
                self.light_identifier.counter = 2
            else:
                self.light_identifier.counter = 3
            '''
            state = self.get_light_state(light)
            GET_GROUND_TRUTH = False
            if(GET_GROUND_TRUTH == True):
                state = self.lights[self.light_cnt].state
            return light_wp, state

        return 999999, TrafficLight.UNKNOWN


if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
