#!/usr/bin/env python2

import rospy
from geometry_msgs.msg import PoseStamped, Point
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        rospy.Subscriber('/obstacle_waypoint', Waypoint, self.obstacle_cb)


        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        self._waypoints = Lane()
        self._pose = PoseStamped()
        self._pose_initialized = False
        self._waypoints_initialized = False
        self._base_vel = 10
        self._debug_print = False
        self._stop_line_position = Point();

        self.go()

        rospy.spin()

    def go(self):
        rate = rospy.Rate(5)
        while not rospy.is_shutdown():
            if(True == self._pose_initialized):
                if(True == self._waypoints_initialized):
                    closest_wp = self.closest_waypoint()
                    rospy.loginfo("Closest waypoint : " + str(self._waypoints.waypoints[closest_wp].pose.pose.position.x)
                            + ", " + str(self._waypoints.waypoints[closest_wp].pose.pose.position.y))
                    final_wps = []
                    num_wps = len(self._waypoints.waypoints)
                    out_msg = Lane()
                    rospy.loginfo("Orientation : " + str(self._pose.pose.orientation.x) + ", " + str(self._pose.pose.orientation.y)
                            + ", " + str(self._pose.pose.orientation.z) + ", " + str(self._pose.pose.orientation.w))
                    for i in range(LOOKAHEAD_WPS ):
                        if(i < LOOKAHEAD_WPS - 1):
                            self.eval_waypoint_vel( (closest_wp + i) % num_wps, (closest_wp + i + 1) % num_wps,
                                    (closest_wp + i + 2) % num_wps)
                        final_wps.append(self._waypoints.waypoints[(closest_wp + i)%num_wps])

                    if self._debug_print == True:
                    #   self._debug_print = False
                        for i in range(len(final_wps)):
                            rospy.logerr(str(final_wps[i].pose.pose.position.x) + ", " + str(final_wps[i].pose.pose.position.y))

                    out_msg.waypoints = final_wps
                    self.final_waypoints_pub.publish(out_msg)

            rate.sleep()



    def pose_cb(self, msg):
        self._pose_initialized = True
        self._pose = msg

        rospy.loginfo("Got new pose : " + str(msg.pose.position.x) + ", "
                      + str(msg.pose.position.y) + ", " + str(msg.pose.position.z))

    def eval_waypoint_vel(self, cur_waypoint, next_waypoint, next_next_waypoint):
        '''
        v_tot = self._base_vel
        wp1_x = self._waypoints.waypoints[cur_waypoint].pose.pose.position.x
        wp1_y = self._waypoints.waypoints[cur_waypoint].pose.pose.position.y

        wp2_x = self._waypoints.waypoints[next_waypoint].pose.pose.position.x
        wp2_y = self._waypoints.waypoints[next_waypoint].pose.pose.position.y
        theta = 0
        if wp2_x != wp1_x:
            theta = (wp2_y - wp1_y) / (wp2_x - wp1_x)
        v_x = v_tot * math.sin(theta)
        v_y = v_tot * math.cos(theta)
        self._waypoints.waypoints[cur_waypoint].twist.twist.linear.x = v_x
        self._waypoints.waypoints[cur_waypoint].twist.twist.linear.y = v_y
        self._waypoints.waypoints[cur_waypoint].twist.twist.linear.z = 0
        self._waypoints.waypoints[cur_waypoint].twist.twist.angular.x = 1
        '''
        self._waypoints.waypoints[cur_waypoint].twist.twist.linear.x = self._base_vel
        self._waypoints.waypoints[cur_waypoint].twist.twist.angular.z = 50
        self._waypoints.waypoints[cur_waypoint].twist.twist.angular.y = 50
        self._waypoints.waypoints[cur_waypoint].twist.twist.angular.x = 50




    def waypoints_cb(self, waypoints):
        self._waypoints = waypoints
        self._waypoints_initialized = True

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        self._traffic_waypoint_id = msg.data
        self._stop_line_position.x = self._waypoints.waypoints[self._traffic_waypoint_id].pose.pose.position.x-20.991  #according to site_traffic_light_config.yaml
        self._stop_line_position.y = self._waypoints.waypoints[self._traffic_waypoint_id].pose.pose.position.y-22.837     #according to site_traffic_light_config.yaml
        self.get_stop_waypoint()
        self.set_waypoints_to_stop()

    def get_stop_waypoint(self):
        min_dist = 99999999
        stop_waypoint_id = 99999999
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  )
        i = self._traffic_waypoint_id
        for waypoint in self._waypoints.waypoints:
            dist = dl(self._stop_line_position, waypoint.pose.pose.position)
            if (dist < min_dist):
                min_dist = dist
                stop_waypoint_id = i
            i = i - 1
            stop_waypoint_id = stop_waypoint_id - 2   # margin N: stop about N points timeslots before the stop line
            self._stop_waypoint_id = stop_waypoint_id

    def set_waypoints_to_stop(self):
        out_msg = Lane()
        final_wps = []
        total_distance = 0
        total_delta_t = 0

        #self._waypoints.waypoints[0].twist.twist.linear.x = self._base_vel
        for i in range(self._stop_waypoint_id, 0, -1):
            total_distance += distance(self._waypoints.waypoints,i, i-1)
            total_delta_t = self._waypoints.waypoints[i].twist.header.stamp -
                                     self._waypoints.waypoints[i-1].twist.header.stamp

        deaceleration = math.sqrt(2.0 * total_distance/total_delta_t/total_delta_t)

        if(deaceleration>10):
            rospy.logerr("the deaceleration is higher than 10 m2/s")

        for i in range(LOOKAHEAD_WPS-1 ):
            if(i >= self._stop_waypoint_id):
                self._waypoints.waypoints[i].twist.twist.linear.x = 0
            else:
                dealta_t = self._waypoints.waypoints[i+1].twist.header.stamp-self._waypoints.waypoints[i].twist.header.stamp
                self._waypoints.waypoints[i].twist.twist.linear.x = self._base_vel - deaceleration * dealta_t

            final_wps.append(self._waypoints.waypoints[i])

        out_msg.waypoints = final_wps
        self.final_waypoints_pub.publish(out_msg)




    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def closest_waypoint(self):
        min_dist = 99999999
        min_dist_id = 99999999
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        i = 0
        for waypoint in self._waypoints.waypoints:
            dist = dl(self._pose.pose.position, waypoint.pose.pose.position)
            if (dist < min_dist):
                min_dist = dist
                min_dist_id = i
            i = i + 1
        return min_dist_id


    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
