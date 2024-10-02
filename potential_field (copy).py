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
        self.eta = 0.55# scaling factor for repulsive force
        self.zeta = 2.2 # scaling factor for attractive force
        self.q_star = 2.2 # threshold distance for obstacles
        self.d_star = 1.1 # threshold distance for goal

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
        rate = rospy.Rate(self.sample_rate)
        while not rospy.is_shutdown():
            if(self.path_data):
                self.global_path_pub.publish(self.path_data)

            self.robot_path_publish()
            if self.goal:
                # Not at goal
                if not self.at_point((self.goal.pose.position.x,self.goal.pose.position.y)):
                    net_force = self.compute_attractive_force() + self.compute_repulsive_force(self.eta) + self.compute_tangential_force(self.eta*2)
                    self.publish_sum(net_force[0],net_force[1])
                else:
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

    def convert_to_global(self,point, sensor_frame,global_frame="map"):
        # Create a PointStamped message for the point in the sensor frame
        point_in_sensor_frame = PointStamped()
        point_in_sensor_frame.header.frame_id = sensor_frame
        point_in_sensor_frame.header.stamp = rospy.Time(0)  # Use the latest available transform
        point_in_sensor_frame.point.x = point[0]
        point_in_sensor_frame.point.y = point[1]
        point_in_sensor_frame.point.z = 0

        try:
            # Wait for the transform to be available
            self.listener.waitForTransform(global_frame, sensor_frame, rospy.Time(0), rospy.Duration(4.0))
            # Transform the point to the global frame
            point_in_global_frame = self.listener.transformPoint(global_frame, point_in_sensor_frame)

            return point_in_global_frame.point.x, point_in_global_frame.point.y
        
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logerr("Transformation failed: {}".format(e))
            return None
        
    def at_point(self,point):
        odom_data = self.odom
        pos_x = odom_data.pose.pose.position.x
        pos_y = odom_data.pose.pose.position.y
        if abs(pos_x - point[0]) <= 0.05 and abs(pos_y - point[1]) <= 0.05:
            return True
        else:
            return False
        
    def is_oscillating(self):
        """Check if the robot's position is fluctuating within a small range."""
        # Calculate the distance between the first and last positions in the list
        first_position = self.prev_positions[0]
        last_position = self.prev_positions[-1]
        middel_position = self.prev_positions[5]
        distance = math.sqrt(
            (last_position[0] - first_position[0]) ** 2 +
            (middel_position[0] - middel_position[0]) ** + 
            (last_position[1] - first_position[1]) ** 2
        )
        # If the distance is below a threshold, consider it oscillation
        print(distance)
        if distance < 1:
            print("OSCILLATING /")
            return True  # Adjust threshold based on your robot's speed
        else: return False
        
    def at_local_minima(self):
        if not self.at_point((self.goal.pose.position.x,self.goal.pose.position.y)):
            # Check if the robot is oscillating
            vel_vec = (self.odom.twist.twist.linear.x,self.odom.twist.twist.linear.y)
            if len(self.prev_positions) > 10:  # Adjust the number of positions tracked
                self.prev_positions.pop(0)  # Keep the list size manageable
            if self.is_oscillating():
                return True
            else: 
                return False
        else:
            return False
        
    def add_obstacle(self):
        pos_x = self.odom.pose.pose.position.x
        pos_y = self.odom.pose.pose.position.y
        self.temporary_obstacles.append([pos_x,pos_y])

    
    def determine_rotation_direction(self,point1, point2):
        angle1 = np.arctan2(point1[1], point1[0])  
        angle2 = np.arctan2(point2[1], point2[0])  

        delta_angle = angle2 - angle1
        
        delta_angle = np.arctan2(np.sin(delta_angle), np.cos(delta_angle))

        # Determine whether to rotate clockwise or counterclockwise
        if delta_angle > 0:
            return False
        else:
            return True

    def compute_tangential_force(self,eta,vector=None,theta=np.deg2rad(90)):
        
        if vector is None:
            vector = self.compute_repulsive_force(eta)
        
        R = [[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]]
        return np.dot(R,vector)
        
    def compute_repulsive_force(self,eta):
        if self.laser == None or self.odom == None or self.goal == None:
            return np.array([0,0])
    
        laser_data = self.laser
        ranges = np.array(laser_data.ranges)
        c = 0
        angle = laser_data.angle_min
        resolution = laser_data.angle_increment
        vector_sum = np.array([0.0,0.0])

        # The lidar outputs 360 degree data and therefore we get data of 360 points within the range of the sensor.
        # Each obstacle detected by the sensor will occupy some points out of these 360 points. We treat each single point as 
        # an individual obstacle.
        odom_data = self.odom
        pos_x = odom_data.pose.pose.position.x
        pos_y = odom_data.pose.pose.position.y
        dist_to_goal = np.sqrt((pos_x - self.goal.pose.position.x)**2 + (pos_y - self.goal.pose.position.y)**2)
        current_angle = odom_data.pose.pose.orientation.x
        #print("current angle ", current_angle)
        # virtual obstacles
        for temp_obs in self.temporary_obstacles.copy():
            x,y = temp_obs
            distance = np.linalg.norm((x-pos_x,y-pos_y))
            if distance < self.q_star and distance < dist_to_goal and distance > 0:
                mag = 2*eta*(1/distance - 1/self.q_star)**2/distance**3
                vector = mag*np.array((x,y)) ## What should be the magnitude of the repulsive force generated by each obstacle point?
                c += 1
            else:
                vector = np.array((0,0))
                self.temporary_obstacles.remove(temp_obs)
            vector_sum += vector
            
        # measured obstacles
        for i, distance in enumerate(ranges):
            # if obstacle too far then don't care. Also obstacle needs to be closer than goal for me to care
            if distance < self.q_star and distance < dist_to_goal:      
                x,y = (distance*np.cos(angle),distance*np.sin(angle))
                mag = 2*eta*(1/distance - 1/self.q_star)**2/distance**3
                vector = -mag*np.array((x,y)) ## What should be the magnitude of the repulsive force generated by each obstacle point?
                c += 1
            else:
                vector = np.array((0,0)) ## What should be the magnitude of repulsive force outside the region of "influence"?
            vector_sum += vector ## You need to add the effect of all obstacle points
            angle += resolution
        
        # Normalization of the repulsive forces
        if c > 0:
            return vector_sum/c
        else:
            return vector_sum
#------------------------------------------

    def compute_attractive_force(self):
        if self.odom == None or self.goal == None:
            return np.array([0,0])

        odom_data = self.odom
        pos_x = odom_data.pose.pose.position.x
        pos_y = odom_data.pose.pose.position.y
        pos = []
        pos.append(pos_x)
        pos.append(pos_y)

        closest_waypoint = []
        

        while(not closest_waypoint or closest_waypoint is None or not closest_waypoint[1]):
            closest_waypoint = self.closest_waypoint(pos, self.position_all)

        dist_to_goal = np.sqrt((pos_x - self.goal.pose.position.x)**2 + (pos_y - self.goal.pose.position.y)**2)
        
        vec_q = np.array([pos_x - self.goal.pose.position.x, pos_y - self.goal.pose.position.y])
        if dist_to_goal <= self.d_star:
            vector = self.zeta*vec_q
        else:
            vector = self.d_star*self.zeta*vec_q/dist_to_goal


        return -vector
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
            self.position_all = [list(double) for double in zip(self.position_x,self.position_y)]
            # reset waypoints
            self.position_x = []
            self.position_y = []

if __name__ == "__main__":
    pf = potential_field()
    pf.start()