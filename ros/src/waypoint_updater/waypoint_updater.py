#!/usr/bin/env python2

import rospy
from geometry_msgs.msg import PoseStamped, Point, TwistStamped, Twist
from styx_msgs.msg import Lane, Waypoint, NextLight
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
INVALID_WP_ID = 999999

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.velocity_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        rospy.Subscriber('/obstacle_waypoint', Waypoint, self.obstacle_cb)
        rospy.Subscriber('/traffic_waypoint', NextLight, self.upcoming_red_light_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        self._waypoints = Lane()
        self._pose = PoseStamped()
        self._pose_initialized = False
        self._waypoints_initialized = False
        #self._base_vel = 90/2.25
        self._base_vel = rospy.get_param('/waypoint_loader/velocity')/3.6
        self._debug_print = False
        self._stop_line_position = Point()
        self._light_waypoint = INVALID_WP_ID
        self._red_light_ahead = False
        self._closest_wp = Waypoint()
        self._velocity = Twist()
        self._velocity_initialized = False
        self._dl  = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  )
        self._keep_vel_flag = False
        self._keep_vel_val = 0

        self.go()

        rospy.spin()

    def go(self):
        rate = rospy.Rate(5)
        while not rospy.is_shutdown():
            if(True == self._pose_initialized):
                if(True == self._waypoints_initialized):
                    closest_wp = self.closest_waypoint()
                    self._closest_wp = closest_wp
                    rospy.loginfo("Closest waypoint : " + str(closest_wp) +" : "
                            + str(self._waypoints.waypoints[closest_wp].pose.pose.position.x)
                            + ", " + str(self._waypoints.waypoints[closest_wp].pose.pose.position.y))

                    self.setup_trajectory(closest_wp, self._light_waypoint)

            rate.sleep()

    def get_steady_trajectory(self, base_vel, curr_wp):
        self._keep_vel_flag = False
        num_wp = len(self._waypoints.waypoints)
        for i in range(LOOKAHEAD_WPS):
            check_wp = (curr_wp + i + 1) % num_wp
            #rospy.loginfo("Steady Traj: Waypoint : %d set to velocity : %d",check_wp,
            #        self._waypoints.waypoints[i].twist.twist.linear.x)
            self._waypoints.waypoints[check_wp].twist.twist.linear.x = base_vel
            self._waypoints.waypoints[check_wp].twist.twist.angular.x = 0
            self._waypoints.waypoints[check_wp].twist.twist.angular.y = 0
            self._waypoints.waypoints[check_wp].twist.twist.angular.z = 0
        return self._waypoints.waypoints

    def get_stop_trajectory(self, light_wp, curr_vel, curr_wp):
        # Find the waypoint to start decelerating from
        # s = (v^2 - u^2)/2a
        # here v = 0, u = current velocity, a = 5m/s2
        if(self._keep_vel_flag == True):
            v_local = self._keep_vel_val
        else:
            v_local = curr_vel
            self._keep_vel_flag = True
            self._keep_vel_val = curr_vel

        a = 7
        s = (v_local * v_local)/(2 * a)
        num_wp = len(self._waypoints.waypoints)
        first_dec_wp = self.get_wp_at_distance_before(s + 15, light_wp)
        rospy.loginfo("Curr_vel = %d, s = %d",curr_vel, s)
        rospy.loginfo("First waypoint to decrease speed : %d",first_dec_wp)
        rospy.loginfo("Light waypoint : %d, num_waypoints = %d",
                      light_wp,
                      len(self._waypoints.waypoints))

        v_exp = math.sqrt(2 * a * self.distance(self._waypoints.waypoints, curr_wp, light_wp))

        danger = False

        if( v_exp < v_local):
            danger = True

        if( danger == True):
            a = a + 10 * (v_local - v_exp)



        for i in range(LOOKAHEAD_WPS):
            check_wp = (curr_wp + i + 1) % num_wp
            prev_wp  = (curr_wp + i) % num_wp
            if(self.is_inbetween( check_wp, first_dec_wp, light_wp)):
                u = self._waypoints.waypoints[check_wp].twist.twist.linear.x
                s_local = self._dl( self._waypoints.waypoints[check_wp].pose.pose.position,
                                    self._waypoints.waypoints[prev_wp].pose.pose.position)

                v_2 = u*u -  2 * a * s_local;
                v = 0
                if v_2 > 0:
                    v = min(math.sqrt(v_2), self._base_vel)

                self._waypoints.waypoints[check_wp].twist.twist.linear.x = v
            elif(self.is_after(check_wp,light_wp)):
                self._waypoints.waypoints[check_wp].twist.twist.linear.x = 0
            else:
                self._waypoints.waypoints[check_wp].twist.twist.linear.x = v_local

            rospy.loginfo("Stop Traj : Waypoint : %d set to velocity : %d",check_wp,
                    self._waypoints.waypoints[check_wp].twist.twist.linear.x)
            self._waypoints.waypoints[check_wp].twist.twist.angular.x = 0
            self._waypoints.waypoints[check_wp].twist.twist.angular.y = 0
            self._waypoints.waypoints[check_wp].twist.twist.angular.z = 0
        return self._waypoints.waypoints



    def get_wp_at_distance_before(self, distance, source_wp):
        num_wp = len(self._waypoints.waypoints)

        start_wp = (int)(source_wp - (num_wp/2))
        found = False
        found_wp = start_wp
        for i in range(num_wp/2):
            check_wp = (source_wp - i) % num_wp
            check_dist = self.distance(self._waypoints.waypoints, check_wp, source_wp)
            rospy.loginfo("Distance between %d and source(%d) is : %d", check_wp, source_wp,check_dist)
            if(distance < check_dist):
                found_wp = check_wp
                found = True
                break

        return found_wp

    def setup_trajectory(self, first_wp, light_wp):
        out_msg = Lane()
        final_wps = []
        if((self._red_light_ahead == False) or (self.is_earlier(light_wp, first_wp)))  :
            final_wps = self.get_steady_trajectory(self._base_vel, first_wp)
        else:
            final_wps = self.get_stop_trajectory(light_wp, self._velocity.linear.x, first_wp)
        out_msg.waypoints = final_wps[first_wp + 1:first_wp + 1 + LOOKAHEAD_WPS]
        if(first_wp + 1 + LOOKAHEAD_WPS > len(final_wps)):
            remaining_wps = LOOKAHEAD_WPS - (len(final_wps) - first_wp - 1)
            out_msg.waypoints += final_wps[0 : remaining_wps]
        rospy.loginfo("First WP : %d, Light WP : %d, num_wp sent : %d", first_wp, light_wp, len(out_msg.waypoints))
        self.final_waypoints_pub.publish(out_msg)


    def upcoming_red_light_cb(self, msg):
        if self._waypoints_initialized == True:
            if msg.state != NextLight.GREEN and msg.state != NextLight.UNKNOWN and msg.waypoint != INVALID_WP_ID:
                rospy.loginfo("Red light upcoming at waypoint : %d, current waypoint : %d",
                        msg.waypoint,
                        self._closest_wp
                        )
                self._stop_line_position = self._waypoints.waypoints[msg.waypoint].pose.pose.position
                self._light_waypoint = msg.waypoint
                self._red_light_ahead = True
            else:
                rospy.loginfo("Light is GREEN")
                self._red_light_ahead = False

    def velocity_cb(self, msg):
        self._velocity = msg.twist
        self._velocity_initialized = True


    def pose_cb(self, msg):
        self._pose_initialized = True
        self._pose = msg

        rospy.loginfo("Got new pose : " + str(msg.pose.position.x) + ", "
                      + str(msg.pose.position.y) + ", " + str(msg.pose.position.z))

    def is_inbetween(self, check_wp, wp1, wp2):
        if(self.is_earlier(wp1, wp2)):
            if(self.is_earlier(check_wp, wp2)):
                return self.is_after(check_wp, wp1)
            else:
                return False
        else:
            if(self.is_earlier(check_wp, wp1)):
                return self.is_after(check_wp, wp2)
            else:
                return False



    def is_after(self, wp1, wp2):
        return not self.is_earlier(wp1, wp2)



    def is_earlier(self, wp1, wp2):
        num_wp = len(self._waypoints.waypoints)
        if(wp1 > wp2):
            if((wp1 - wp2) < num_wp/2 ):
                return False
            else:
                return True
        else:
            if((wp2 - wp1) < num_wp/2 ):
                return True
            else:
                return False

    def eval_waypoint_vel(self, cur_waypoint, next_waypoint, next_next_waypoint, first_waypoint, light_waypoint, first_point_vel, next_point_vel):
        if self._red_light_ahead == False:
            self._waypoints.waypoints[cur_waypoint].twist.twist.linear.x = self._base_vel
            self._waypoints.waypoints[cur_waypoint].twist.twist.angular.z = 0
            self._waypoints.waypoints[cur_waypoint].twist.twist.angular.y = 0
            self._waypoints.waypoints[cur_waypoint].twist.twist.angular.x = 0
        else:
            self._stop_waypoint_id = light_waypoint

            if(self.is_earlier(light_waypoint, cur_waypoint)):
                self._waypoints.waypoints[cur_waypoint].twist.twist.linear.x = 0
                next_point_vel = 0
            else:
                slope = 0
                if(first_waypoint != light_waypoint):
                    slope = (first_point_vel - 0)/(first_waypoint - light_waypoint)
                intercept = first_point_vel + (slope * first_waypoint)
                next_point_vel = first_point_vel + slope * (cur_waypoint - first_waypoint)
                self._waypoints.waypoints[cur_waypoint].twist.twist.linear.x = next_point_vel
            self._waypoints.waypoints[cur_waypoint].twist.twist.angular.z = 0
            self._waypoints.waypoints[cur_waypoint].twist.twist.angular.y = 0
            self._waypoints.waypoints[cur_waypoint].twist.twist.angular.x = 0








    def waypoints_cb(self, waypoints):
        self._waypoints = waypoints
        self._waypoints_initialized = True
        rospy.loginfo("Received waypoints : length : %d",len(waypoints.waypoints))

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        pass
        #self._traffic_waypoint_id = msg.data
        #self._stop_line_position.x = self._waypoints.waypoints[self._traffic_waypoint_id].pose.pose.position.x-20.991  #according to site_traffic_light_config.yaml
        #self._stop_line_position.y = self._waypoints.waypoints[self._traffic_waypoint_id].pose.pose.position.y-22.837     #according to site_traffic_light_config.yaml
        #self.get_stop_waypoint()
        #self.set_waypoints_to_stop()

    def get_stop_waypoint(self):
        min_dist = 99999999
        stop_waypoint_id = INVALID_WP_ID
        i = self._traffic_waypoint_id
        for waypoint in self._waypoints.waypoints:
            dist = self._dl(self._stop_line_position, waypoint.pose.pose.position)
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
        total_delta_t = 0.05

        for i in range(self._stop_waypoint_id, 0, -1):
            total_distance += distance(self._waypoints.waypoints,i, i-1)
            total_delta_t = 0.05
            #self._waypoints.waypoints[i].twist.header.stamp - self._waypoints.waypoints[i-1].twist.header.stamp

        rospy.loginfo("total_delta_t = %d, total_distance = %d",total_distance, total_delta_t)
        deceleration = math.sqrt(2.0 * total_distance/(total_delta_t* total_delta_t))

        if(deceleration>10):
            rospy.logerr("the deceleration is higher than 10 m2/s")

        for i in range(LOOKAHEAD_WPS-1 ):
            if(i >= self._stop_waypoint_id):
                self._waypoints.waypoints[i].twist.twist.linear.x = 0
            else:
                delta_t = self._waypoints.waypoints[i+1].twist.header.stamp-self._waypoints.waypoints[i].twist.header.stamp
                self._waypoints.waypoints[i].twist.twist.linear.x = self._base_vel - deceleration * delta_t

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
        min_dist_id = INVALID_WP_ID
        i = 0
        for waypoint in self._waypoints.waypoints:
            dist = self._dl(self._pose.pose.position, waypoint.pose.pose.position)
            if (dist < min_dist):
                min_dist = dist
                min_dist_id = i
            i = i + 1
        return min_dist_id


    def distance(self, waypoints, wp1, wp2):
        dist = 0
        for i in range(wp1, wp2+1):
            dist += self._dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
