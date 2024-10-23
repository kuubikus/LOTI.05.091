#! /usr/bin/env python3

# TODO
# local minima check
# only look at obstacle that are towards me

from cmath import inf
from visualization_msgs.msg import Marker,MarkerArray
import roslib, sys, rospy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry,Path
from geometry_msgs.msg import Point,PoseStamped,PointStamped
import numpy as np
import math
import time
import tf
from cvxopt import solvers
import cvxopt
from scipy.interpolate import CubicSpline
#------------------------------------------

class potential_field:
    def __init__(self):
        rospy.init_node("potential_field")

        self.sample_rate = rospy.get_param("~sample_rate", 10)

        # Subscribe to the global planner using the move base package. The global plan is the path that the robot would ideally follow if 
        # there are no unknown/dynamic obstacles. In the videos this is highlighted by green color.
        self.global_path_sub = rospy.Subscriber("/move_base/NavfnROS/plan", Path, self.handle_global_path)

        self.laser_sub = rospy.Subscriber("/scan", LaserScan, self.handle_laser)
        self.odom_sub = rospy.Subscriber("/odom", Odometry, self.handle_odom)
        
        # Subscribe to the goal topic to get the goal position given using rviz's 2D Navigation Goal option.
        self.goal_sub = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.handle_goal)

        # Publish the potential field vector topic which will be subscribed by the command_velocity node in order to
        # compute velocities.
        self.potential_field_pub = rospy.Publisher("potential_field_vector", Point,queue_size=10)

        # We store the path data gotten from the global planner above and display it. We have written a custom publisher 
        # in order to get more flexibility while displaying the paths.
        self.global_path_pub = rospy.Publisher("global_path",Path,queue_size=10)

        # This is a publisher to publish the robot path. In the videos it is highlighted by red color.
        self.robot_path_pub = rospy.Publisher("robot_path",Path,queue_size=10)

        # For CS transforms
        self.CS_listener= tf.TransformListener()
        
        self.path_robot = Path()
        self.path_robot.header.frame_id = 'map'

        ## TODO Choose suitable values
        self.q_star = 2 # threshold distance for obstacles
        self.Kp = 10
        self.Kd = self.Kp/15
        self.dt = 0.1
       
        self.laser = None
        self.odom = None
        self.goal = None

        self.path_data = Path()
        self.path_data.header.frame_id = 'map'
        
        self.position_x = []
        self.position_y = []
        self.position_all = []
        self.temporary_obstacles = []
        self.prev_positions = []
        # to reset path upon adding goal
        self.last_index = 0
        
        # Boolean variables used for proper display of robot path and global path
        self.bool_goal = False
        self.bool_path = False

        self.max_force = 1
#------------------------------------------
    def start(self):
        x_path_data = np.load('/home/marten/catkin_ws/src/APF_package/potential_field/scripts/sine.npy')[0]
        y_path_data = np.load('/home/marten/catkin_ws/src/APF_package/potential_field/scripts/sine.npy')[1]
        rate = rospy.Rate(self.sample_rate)
        while not rospy.is_shutdown():
            if(self.path_data):
                self.global_path_pub.publish(self.path_data)

            self.robot_path_publish()
            if self.goal:
                # Not at goal
                if not self.at_point((x_path_data[-1],y_path_data[-1])):
                    """if self.at_local_minima():
                        self.add_obstacle()"""
                    cs_x_path, cs_y_path, cs_phi_path, arc_length, arc_vec, cs_xdot, cs_ydot = self.path_spline(x_path_data, y_path_data)
                    net_force = self.cbf(x_path_data,y_path_data,arc_vec,cs_x_path, cs_y_path,cs_phi_path, arc_length,cs_xdot, cs_ydot,alpha=0.8)
                    print("net force ", net_force)
                    self.publish_sum(net_force[0],net_force[1])
                else:
                    self.temporary_obstacles = []
                    self.publish_sum(0,0)
            rate.sleep()

#------------------------------------------
    def robot_path_publish(self):
        if(self.odom):
            odom_data = self.odom
            if(self.bool_path == True):
                self.bool_path = False
                self.path_robot = Path()
                self.path_robot.header.frame_id = 'map'
            pose = PoseStamped()
            pose.header = odom_data.header
            pose.pose = odom_data.pose.pose
            self.path_robot.poses.append(pose)
            self.robot_path_pub.publish(self.path_robot)

#------------------------------------------
    def path_spline(self, x_path, y_path):
        x_diff = np.diff(x_path)
        y_diff = np.diff(y_path)
        phi = np.unwrap(np.arctan2(y_diff, x_diff))
        phi_init = phi[0]
        phi = np.hstack(( phi_init, phi  ))
        arc = np.cumsum( np.sqrt( x_diff**2+y_diff**2 )   )
        arc_length = arc[-1]
        arc_vec = np.linspace(0, arc_length, np.shape(x_path)[0])
        cs_x_path = CubicSpline(arc_vec, x_path)
        cs_y_path = CubicSpline(arc_vec, y_path)
        cs_phi_path = CubicSpline(arc_vec, phi)
        # xd_d0t, yd_dot
        cs_xdot = cs_x_path.derivative()
        cs_ydot = cs_y_path.derivative()
        # x(s), y(s), ...
        return cs_x_path, cs_y_path, cs_phi_path, arc_length, arc_vec, cs_xdot, cs_ydot
        
    # Gives the closest waypoint
    def waypoint_generator(self, x_global_init, y_global_init, x_path_data, y_path_data, arc_vec, cs_x_path, cs_y_path, cs_phi_path, arc_length,cs_xdot,cs_ydot,v,dt):
        #index of the closest point
        idx = np.argmin( np.sqrt((x_global_init-x_path_data)**2+(y_global_init-y_path_data)**2))
        arc_curr = arc_vec[idx] #s0
        arc_pred = arc_curr + v*dt #so+dt*sdot
        x_waypoints = cs_x_path(arc_pred) # x_d(so+dt*sdot)
        y_waypoints =  cs_y_path(arc_pred)
        phi_Waypoints = cs_phi_path(arc_pred)
        xd_dot = cs_xdot(arc_pred)*v
        yd_dot = cs_ydot(arc_pred)*v
        return x_waypoints, y_waypoints, phi_Waypoints,xd_dot,yd_dot # returns x_d,y_d,_ from slides
    
    # PD controller implementation
    def PD_controller(self, x_path_data, y_path_data, arc_vec, cs_x_path, cs_y_path, cs_phi_path, arc_length, cs_xdot, cs_ydot):
        # current velocity
        dt = self.dt
        vel_vec = (self.odom.twist.twist.linear.x,self.odom.twist.twist.linear.y)
        v = np.linalg.norm(vel_vec)
        # current pos
        x_global_init = self.odom.pose.pose.position.x
        y_global_init = self.odom.pose.pose.position.y
        # Call the waypoint generator to get desired waypoints
        x_waypoints, y_waypoints, phi_waypoints, xd_dot, yd_dot = self.waypoint_generator(
            x_global_init, y_global_init, x_path_data, y_path_data, arc_vec, 
            cs_x_path, cs_y_path, cs_phi_path, arc_length, cs_xdot, cs_ydot, v, dt)
        
        # Calculate errors (distance between the current position and the desired position)
        
        # Proportional term: multiply errors by Kp
        u_p_x = self.Kp * (x_waypoints - x_global_init)
        u_p_y = self.Kp * (y_waypoints - y_global_init)

        # Derivative term: error in velocity (difference in predicted velocities)
        u_d_x = self.Kd * (xd_dot - vel_vec[0])
        u_d_y = self.Kd * (yd_dot - vel_vec[1])

        # Total control input for x and y
        u_x = u_p_x + u_d_x
        u_y = u_p_y + u_d_y
        # Output the desired control values
        return np.array((u_x, u_y))

#------------------------------------------
    def cbf(self,x_path_data,y_path_data,arc_vec,cs_x_path, cs_y_path,cs_phi_path, arc_length,cs_xdot, cs_ydot,alpha=0.6):
        if self.laser == None or self.odom == None or self.goal == None:
            return np.array([0,0])

        # laser set up
        laser_data = self.laser
        ranges = np.array(laser_data.ranges)
        angle = laser_data.angle_min
        angle_increment = laser_data.angle_increment

        # solver set up
        num_dim = 2
        odom_data = self.odom
        x = odom_data.pose.pose.position.x
        y = odom_data.pose.pose.position.y
        
        
        v_des = self.PD_controller(x_path_data,y_path_data,arc_vec,cs_x_path, cs_y_path,cs_phi_path, arc_length,cs_xdot, cs_ydot)

        # Initialize constraint matrices for multiple obstacles
        Q = np.identity(num_dim)
        q = -v_des
        A_in_list = []
        b_in_list = []

        R = 0.3 # Robot radius
        # Iterate through each obstacle from laser
        for distance in ranges:
            if distance < self.q_star and distance > R: 
                dist_obs = distance - R  # h(x)
                grad_hx = np.hstack((np.cos(angle),np.sin(angle))).reshape(1, num_dim)
                A_in_list.append(grad_hx)
                b_in_list.append(alpha*dist_obs)
            angle += angle_increment

        # Iterate through each obstacle from virtual ones
        for point in self.temporary_obstacles:
            x_obs = point[0]
            y_obs = point[1]
            norm_factor = np.sqrt((x - x_obs)**2 + (y - y_obs)**2)
            dist_obs = np.sqrt((x - x_obs)**2 + (y - y_obs)**2) - R  # h(x)
            grad_hx = np.hstack(((x - x_obs) / norm_factor, (y - y_obs) / norm_factor)).reshape(1, num_dim)

            A_in_list.append(-grad_hx)
            b_in_list.append(alpha * dist_obs)

        # if any obstacles detected
        if A_in_list:
            A_in = np.vstack(A_in_list)
            b_in = np.array(b_in_list)
            sol_data = solvers.qp(cvxopt.matrix(Q, tc='d'), cvxopt.matrix(q, tc='d'), cvxopt.matrix(A_in, tc='d'), cvxopt.matrix(b_in, tc='d'), None, None)
            sol = np.asarray(sol_data['x'])
            return (sol[0][0],sol[1][0])
        # no obstacles within range
        else:
            return v_des


    def cbf_with_path(self,x_path_data,y_path_data, alpha=0.6):
        if self.laser == None or self.odom == None or self.goal == None:
            return np.array([0,0])

        # laser set up
        laser_data = self.laser
        ranges = np.array(laser_data.ranges)
        angle = laser_data.angle_min
        angle_increment = laser_data.angle_increment

        # solver set up
        num_dim = 2
        odom_data = self.odom
        x = odom_data.pose.pose.position.x
        y = odom_data.pose.pose.position.y

        cs_x_path, cs_y_path, cs_phi_path, arc_length, arc_vec, cs_xdot, cs_ydot = self.path_spline(x_path_data, y_path_data)
        v_des = self.PD_controller(x_path_data,y_path_data,arc_vec,cs_x_path, cs_y_path,cs_phi_path, arc_length,cs_xdot, cs_ydot)
        v_des = np.hstack((v_des[0],v_des[1]))

        # Initialize constraint matrices for multiple obstacles
        Q = np.identity(num_dim)
        q = -v_des
        A_in_list = []
        b_in_list = []

        R = 0.3 # Robot radius
        # Iterate through each obstacle from laser
        for distance in ranges:
            if distance < self.q_star and distance > R: 
                dist_obs = distance - R  # h(x)
                grad_hx = np.hstack((np.cos(angle),np.sin(angle))).reshape(1, num_dim)
                A_in_list.append(grad_hx)
                b_in_list.append(alpha*dist_obs)
            angle += angle_increment

        # Iterate through each obstacle from virtual ones
        for point in self.temporary_obstacles:
            x_obs = point[0]
            y_obs = point[1]
            norm_factor = np.sqrt((x - x_obs)**2 + (y - y_obs)**2)
            dist_obs = np.sqrt((x - x_obs)**2 + (y - y_obs)**2) - R  # h(x)
            grad_hx = np.hstack(((x - x_obs) / norm_factor, (y - y_obs) / norm_factor)).reshape(1, num_dim)

            A_in_list.append(-grad_hx)
            b_in_list.append(alpha * dist_obs)

        # if any obstacles detected
        if A_in_list:
            A_in = np.vstack(A_in_list)
            b_in = np.array(b_in_list)
            sol_data = solvers.qp(cvxopt.matrix(Q, tc='d'), cvxopt.matrix(q, tc='d'), cvxopt.matrix(A_in, tc='d'), cvxopt.matrix(b_in, tc='d'), None, None)
            sol = np.asarray(sol_data['x'])
            return (sol[0][0],sol[1][0])
        # no obstacles within range
        else:
            return v_des

#------------------------------------------
    
    def at_point(self,point):
        odom_data = self.odom
        pos_x = odom_data.pose.pose.position.x
        pos_y = odom_data.pose.pose.position.y
        if abs(pos_x - point[0]) <= 0.05 and abs(pos_y - point[1]) <= 0.05:
            return True
        else:
            return False
        
    def at_local_minima(self):
        if not self.at_point((self.goal.pose.position.x,self.goal.pose.position.y)):
            # Check if the robot is oscillating
            vel_vec = (self.odom.twist.twist.linear.x,self.odom.twist.twist.linear.y)
            if np.linalg.norm(vel_vec) < 0.1:
                # Basically stopped
                print("MINIMA")
                return True
            else:
                return False
        
    def add_obstacle(self):
        print("Adding obstacle")
        # No more than 3 obstacles
        if len(self.temporary_obstacles) > 3:
            self.temporary_obstacles.pop(0)
        pos_x = self.odom.pose.pose.position.x + 0.1
        pos_y = self.odom.pose.pose.position.y + 0.1
        self.temporary_obstacles.append([pos_x,pos_y])

#------------------------------------------
    def closest_waypoint(self,point, points):
        i=0
        pt=[]
        dist = math.inf
        for p in points:
            if(math.dist(p,point)<dist):
                dist = math.dist(p,point)
                pt = p
                i = points.index(pt)
        return [i,pt]
#------------------------------------------

    def handle_laser(self, laser_data):
        self.laser = laser_data
        
#------------------------------------------
    
    def handle_odom(self, odom_data):
        self.odom = odom_data
        pos = (self.odom.pose.pose.position.x,self.odom.pose.pose.position.y)
        self.prev_positions.append(pos)
#------------------------------------------
    
    def handle_goal(self, goal_data):
        self.bool_goal = True
        self.bool_path = True
        self.goal = goal_data
#------------------------------------------
    def publish_sum(self, x, y):
        vector = Point(x, y, 0)
        self.potential_field_pub.publish(vector)

#------------------------------------------
    def publish_dist_to_goal(self, dist):
        dist_to_goal = Float32(dist)
        self.dist_to_goal_pub.publish(dist_to_goal)
#------------------------------------------

    def handle_global_path(self, path_data):
        if(self.bool_goal == True):
            self.bool_goal = False
            self.path_data = path_data
            i=0
            while(i <= len(self.path_data.poses)-1):
                self.position_x.append(self.path_data.poses[i].pose.position.x)
                self.position_y.append(self.path_data.poses[i].pose.position.y)
                i=i+1
            print("update pos all")
            print("len of pos all: ", len(self.position_all))
            self.position_all = [list(double) for double in zip(self.position_x,self.position_y)]
            # reset waypoints
            self.position_x = []
            self.position_y = []

if __name__ == "__main__":
    pf = potential_field()
    pf.start()