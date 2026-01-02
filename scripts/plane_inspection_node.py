#!/usr/bin/env python3
import rospy
import numpy as np
import actionlib
from enum import Enum
from geometry_msgs.msg import Twist
import tf2_ros
import tf2_geometry_msgs
import math

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Trigger, TriggerResponse
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from cola2_msgs.msg import NavSts, BodyVelocityReq, GoalDescriptor

import tf.transformations as tf

# Dynamic reconfigure
from dynamic_reconfigure.server import Server
from girona500_inspection_trajectories.cfg import InspectionPlannerConfig, InspectionControllerConfig

# Service (trigger-only)
from girona500_inspection_trajectories.srv import (
    PlanInspectionPath,
    PlanInspectionPathResponse
)

from girona500_inspection_trajectories.msg import (
    ExecutePlaneInspectionAction,
    ExecutePlaneInspectionFeedback,
    ExecutePlaneInspectionResult
)
from girona500_inspection_trajectories.msg import ExecutePlaneInspectionGoal


# ============================================================
# State machine
# ============================================================

class InspectionState(Enum):
    IDLE = 0
    PREVIEW = 1
    PLANNED = 2
    EXECUTING = 3


# ============================================================
# Plane Inspection Node
# ============================================================

class PlaneInspectionNode:

    def __init__(self):
        rospy.loginfo("[PlaneInspection] Initializing")

        self.state = InspectionState.IDLE

        self.preview_path = None
        self.planned_path = None

        # Execution state
        self.current_pose = None          # PoseStamped
        self._stop_requested = False
        self._active_index = 0

        # ==============================
        # Controller defaults (overwritten by dyn-reconf)
        # ==============================
        self.exec_rate_hz = 10.0
        self.waypoint_tol = 0.5
        self.exec_timeout = 120.0

        self.v_nominal = 0.2
        self.kp_yaw    = 0.3

        self.max_vx = 0.3
        self.max_wz = 0.3

        self.kp_z   = 0.3
        self.max_vz = 0.2

        # Frame / identity (COLA2)
        self.frame = rospy.get_param("~frame", "world_ned")
        self.name  = rospy.get_param("~requester_name", rospy.get_name())

        # TF
        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(30.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Frames
        self.world_frame = rospy.get_param("~world_frame", "world_ned")
        self.robot_frame = rospy.get_param("~robot_frame", "girona500/base_link")

        # Current pose (from TF)
        self.current_pose = None

        # Publishers
        self.vel_publisher = rospy.Publisher("plane_inspection/cmd_vel",BodyVelocityReq,queue_size=1)

        self.preview_pub = rospy.Publisher(
            "plane_inspection/path_preview",
            Path,
            queue_size=1,
            latch=True
        )
        self.path_marker_pub = rospy.Publisher(
            "plane_inspection/path_markers",
            MarkerArray,
            queue_size=1,
            latch=True
        )

        # Services
        self.stop_srv = rospy.Service("plane_inspection/stop", Trigger, self.stop_cb)
        self.start_srv = rospy.Service("plane_inspection/start", Trigger, self.start_cb)
        self.plan_srv = rospy.Service(
            "plane_inspection/plan_path",
            PlanInspectionPath,
            self.plan_path_cb
        )

        # Dynamic reconfigure
        self.dyn_server = Server(
            InspectionPlannerConfig,
            self.dynamic_reconfigure_cb,
            namespace="planner"
        )

        # Dynamic reconfigure - CONTROLLER (execution)
        self.controller_dyn_server = Server(
            InspectionControllerConfig,
            self.controller_reconfigure_cb,
            namespace="controller"
        )

        self.action_server = actionlib.SimpleActionServer(
            "plane_inspection/execute",
            ExecutePlaneInspectionAction,
            execute_cb=self.execute_cb,
            auto_start=False
        )
        self.action_server.register_preempt_callback(self.preempt_cb)
        self.action_server.start()

        # Internal action client (for start service)
        self.action_client = actionlib.SimpleActionClient(
            "plane_inspection/execute",
            ExecutePlaneInspectionAction
        )

        rospy.loginfo("[PlaneInspection] Waiting for action server...")
        self.action_client.wait_for_server()
        rospy.loginfo("[PlaneInspection] Action server ready")

        rospy.loginfo("[PlaneInspection] Ready")

    # ========================================================
    # Utilities
    # ========================================================

    @staticmethod
    def normalize(v):
        n = np.linalg.norm(v)
        if n < 1e-8:
            raise ValueError("Zero-length vector")
        return v / n

    @staticmethod
    def project_onto_plane(v, n):
        return v - np.dot(v, n) * n
    
    @staticmethod
    def pose_to_np(pose):
        """
        Convert geometry_msgs/Pose to numpy array [x, y, z]
        """
        return np.array([
            pose.position.x,
            pose.position.y,
            pose.position.z
        ], dtype=float)
    
    @staticmethod
    def quat_to_yaw(q):
        """
        Convert quaternion to yaw angle [rad]
        """
        _, _, yaw = tf.euler_from_quaternion([q.x, q.y, q.z, q.w])
        return yaw
    
    @staticmethod
    def wrap_pi(angle):
        """
        Wrap angle to [-pi, pi]
        """
        return (angle + math.pi) % (2.0 * math.pi) - math.pi
    
    def stop_cb(self, req):
        self._stop_requested = True
        self.publish_body_velocity_req(0.0, 0.0, 0.0, 0.0)  # frena
        return TriggerResponse(success=True, message="Stop requested")
    
    def start_cb(self, req):
        """
        Service callback to START plane inspection execution.
        Internally sends a goal to the action server.
        """
        if self.state == InspectionState.EXECUTING:
            return TriggerResponse(
                success=False,
                message="Inspection already executing"
            )

        if self.planned_path is None or len(self.planned_path.poses) == 0:
            return TriggerResponse(
                success=False,
                message="Cannot start inspection: no planned path available"
            )
            

        goal = ExecutePlaneInspectionGoal()
        goal.start = True

        self._stop_requested = False

        self.action_client.send_goal(goal)

        rospy.loginfo("[PlaneInspection] START service called -> action goal sent")

        return TriggerResponse(
            success=True,
            message="Inspection execution started"
        )
    
    def preempt_cb(self):
        rospy.logwarn("[PlaneInspection] Preempt requested")
        self._stop_requested = True
        self.publish_body_velocity_req(0.0, 0.0, 0.0, 0.0)

    def update_current_pose_from_tf(self):
        """
        Update self.current_pose using TF: world_frame -> robot_frame
        """
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                self.world_frame,
                self.robot_frame,
                rospy.Time(0),
                rospy.Duration(0.2)
            )

            ps = PoseStamped()
            ps.header = tf_msg.header
            ps.pose.position.x = tf_msg.transform.translation.x
            ps.pose.position.y = tf_msg.transform.translation.y
            ps.pose.position.z = tf_msg.transform.translation.z
            ps.pose.orientation = tf_msg.transform.rotation

            self.current_pose = ps
            return True

        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            rospy.logwarn_throttle(
                1.0,
                "[PlaneInspection] TF not available (%s -> %s): %s",
                self.world_frame,
                self.robot_frame,
                str(e)
            )
            return False

    def publish_body_velocity_req(self, vx, vy, vz, wz):
        """
        Publica un BodyVelocityReq (COLA2).
        Velocidades expresadas en el body frame.
        """
        bvr = BodyVelocityReq()
        bvr.header.frame_id = self.frame
        bvr.header.stamp = rospy.Time.now()

        bvr.goal.priority = GoalDescriptor.PRIORITY_NORMAL
        bvr.goal.requester = self.name

        # Ejes deshabilitados (según tu configuración)
        bvr.disable_axis.x = False
        bvr.disable_axis.y = False
        bvr.disable_axis.z = False
        bvr.disable_axis.roll  = True
        bvr.disable_axis.pitch = True
        bvr.disable_axis.yaw   = False

        bvr.twist.linear.x = vx
        bvr.twist.linear.y = vy
        bvr.twist.linear.z = vz
        bvr.twist.angular.z = wz

        self.vel_publisher.publish(bvr)

    def compute_control(self, p_cur, yaw_cur, p_wp):
        """
        - Surge: v_nominal * cos(|e_yaw|)
        - Yaw:   P sobre e_yaw
        - Z:     P desacoplado sobre error vertical
        Salidas en BODY frame: vx, vy, vz, wz
        """

        # Error en mundo
        e = p_wp - p_cur
        dist_xy = float(np.linalg.norm(e[:2]))

        # Yaw deseado hacia el waypoint
        yaw_des = math.atan2(e[1], e[0])
        e_yaw = self.wrap_pi(yaw_des - yaw_cur)

        # --- SURGE MODULADO POR |ERROR DE YAW| ---
        vx = self.v_nominal * math.cos(abs(e_yaw))
        vx = max(0.0, vx)                 # evita cos negativo si |e_yaw| > 90°
        vx = min(vx, self.max_vx)

        # --- YAW P ---
        wz = self.kp_yaw * e_yaw
        wz = float(np.clip(wz, -self.max_wz, self.max_wz))

        # --- Z P DESACOPLADO ---
        ez = e[2]                          # (z_wp - z_cur) en world
        vz = self.kp_z * ez
        vz = float(np.clip(vz, -self.max_vz, self.max_vz))

        # Sin sway
        vy = 0.0

        return vx, vy, vz, wz, e_yaw, dist_xy, ez


    # ========================================================
    # Execute Action server
    # ========================================================

    def execute_cb(self, goal):
        # 1) Stop by goal.start = false
        if not goal.start:
            self._stop_requested = True
            self.publish_body_velocity_req(0.0, 0.0, 0.0, 0.0)
            res = ExecutePlaneInspectionResult(success=True, message="Stop received (start=false)")
            self.action_server.set_succeeded(res)
            return

        # 2) Preconditions
        if self.planned_path is None or len(self.planned_path.poses) == 0:
            res = ExecutePlaneInspectionResult(success=False, message="No planned path available")
            self.action_server.set_aborted(res)
            return

        if not self.update_current_pose_from_tf():
            res = ExecutePlaneInspectionResult(
                success=False,
                message="TF not available (cannot get robot pose)"
            )
            self.action_server.set_aborted(res)
            return

        # 3) Init execution
        self.state = InspectionState.EXECUTING
        self._stop_requested = False
        self._active_index = 0

        total = len(self.planned_path.poses)
        start_time = rospy.Time.now()
        rate = rospy.Rate(self.exec_rate_hz)

        rospy.loginfo("[PlaneInspection] EXECUTING (%d points)", total)

        while not rospy.is_shutdown():
            if not self.update_current_pose_from_tf():
                rate.sleep()
                continue

            # Stop / cancel / preempt
            if self.action_server.is_preempt_requested() or self._stop_requested:
                self.publish_body_velocity_req(0.0, 0.0, 0.0, 0.0)
                self.state = InspectionState.PLANNED
                res = ExecutePlaneInspectionResult(success=False, message="Stopped (preempt/stop)")
                self.action_server.set_preempted(res)
                return

            # Timeout
            elapsed = (rospy.Time.now() - start_time).to_sec()
            remaining = max(0.0, self.exec_timeout - elapsed)
            if elapsed > self.exec_timeout:
                self.publish_body_velocity_req(0.0, 0.0, 0.0, 0.0)
                self.state = InspectionState.PLANNED
                res = ExecutePlaneInspectionResult(success=False, message="Timeout")
                self.action_server.set_aborted(res)
                return

            # Current waypoint
            wp = self.planned_path.poses[self._active_index]

            p_cur = self.pose_to_np(self.current_pose.pose)
            p_wp  = self.pose_to_np(wp.pose)
            dist = float(np.linalg.norm(p_wp - p_cur))

            # Yaw error to direction-to-waypoint (plane heading proxy)
            yaw_cur = self.quat_to_yaw(self.current_pose.pose.orientation)
            yaw_des = math.atan2((p_wp[1] - p_cur[1]), (p_wp[0] - p_cur[0]))
            yaw_err = self.wrap_pi(yaw_des - yaw_cur)
            yaw_err_deg = math.degrees(yaw_err)

            # Markers
            self.publish_path_markers(self.planned_path, active_index=self._active_index)

            vx, vy, vz, wz, yaw_err, dist_xy, ez = self.compute_control(
                p_cur,
                yaw_cur,
                p_wp
            )
            self.publish_body_velocity_req(vx, vy, vz, wz)


            # Feedback (Twist para log/GUI, aunque el envío real sea BVR)
            cmd_twist = Twist()
            cmd_twist.linear.x = vx
            cmd_twist.linear.y = vy
            cmd_twist.linear.z = vz
            cmd_twist.angular.z = wz

            fb = ExecutePlaneInspectionFeedback()
            fb.current_index = int(self._active_index)
            fb.total_points = int(total)
            fb.distance_to_waypoint = dist
            fb.yaw_plane_error = yaw_err_deg
            fb.current_pose = self.current_pose
            fb.next_waypoint_pose = wp
            fb.time_remaining = remaining
            fb.yaw_plane_error = math.degrees(yaw_err)
            fb.distance_to_waypoint = dist_xy
            fb.body_cmd_vel = cmd_twist
            self.action_server.publish_feedback(fb)

            # Waypoint reached?
            if dist <= self.waypoint_tol:
                if self._active_index >= total - 1:
                    self.publish_body_velocity_req(0.0, 0.0, 0.0, 0.0)
                    self.state = InspectionState.PLANNED
                    res = ExecutePlaneInspectionResult(success=True, message="Finished")
                    self.action_server.set_succeeded(res)
                    return
                else:
                    self._active_index += 1

            rate.sleep()
    
    # ========================================================
    # Publish Marker arrays (for RViz visualization)
    # ========================================================
    
    def publish_path_markers(self, path, active_index=0):
        ma = MarkerArray()

        for i, pose in enumerate(path.poses):
            m = Marker()

            m.header.frame_id = path.header.frame_id
            m.header.stamp = rospy.Time.now()

            m.ns = "plane_inspection_path"
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD

            m.pose.position.x = pose.pose.position.x
            m.pose.position.y = pose.pose.position.y
            m.pose.position.z = pose.pose.position.z
            m.pose.orientation.w = 1.0

            # Tamaño
            m.scale.x = 0.18
            m.scale.y = 0.18
            m.scale.z = 0.18

            # 🎨 COLORES SEGÚN ESTADO
            if i == active_index:
                if self.state == InspectionState.PREVIEW:
                    # 🔵 Azul
                    m.color.r = 0.0
                    m.color.g = 0.0
                    m.color.b = 1.0
                elif self.state == InspectionState.PLANNED:
                    # 🟡 Amarillo
                    m.color.r = 1.0
                    m.color.g = 1.0
                    m.color.b = 0.0
                elif self.state == InspectionState.EXECUTING:
                    # 🔴 Rojo
                    m.color.r = 1.0
                    m.color.g = 0.0
                    m.color.b = 0.0
                else:
                    # fallback
                    m.color.r = 1.0
                    m.color.g = 1.0
                    m.color.b = 1.0

                m.color.a = 1.0

            else:
                # 🟢 Resto de waypoints
                m.color.r = 0.0
                m.color.g = 1.0
                m.color.b = 0.0
                m.color.a = 0.6

            ma.markers.append(m)

        self.path_marker_pub.publish(ma)

    def controller_reconfigure_cb(self, cfg, level):
        rospy.loginfo("[PlaneInspection] Controller reconfigure update")

        # Execution
        self.exec_rate_hz = cfg.exec_rate_hz
        self.waypoint_tol = cfg.waypoint_tolerance
        self.exec_timeout = cfg.exec_timeout

        # Surge + yaw
        self.v_nominal = cfg.v_nominal
        self.kp_yaw    = cfg.kp_yaw
        self.max_vx    = cfg.max_vx
        self.max_wz    = cfg.max_wz

        # Z
        self.kp_z   = cfg.kp_z
        self.max_vz = cfg.max_vz

        return cfg
    # ========================================================
    # Dynamic reconfigure callback (PREVIEW MODE)
    # ========================================================

    def dynamic_reconfigure_cb(self, cfg, level):
        self.state = InspectionState.PREVIEW
        rospy.loginfo("[PlaneInspection] Dynamic reconfigure update")

        try:
            self.preview_path = self.generate_path_from_cfg(cfg)
            self.preview_pub.publish(self.preview_path)

            if self.preview_path.poses:
                self.publish_path_markers(self.preview_path, active_index=0)

            rospy.loginfo(
                "[PlaneInspection] Preview path published (%d poses)",
                len(self.preview_path.poses)
            )

        except Exception as e:
            rospy.logerr("[PlaneInspection] Preview failed: %s", str(e))

        return cfg

    # ========================================================
    # Trigger PLAN service (commit preview → planned)
    # ========================================================
    def plan_path_cb(self, req):
        self.state = InspectionState.PLANNED
        rospy.loginfo("[PlaneInspection] Plan trigger received")

        if self.preview_path is None:
            return PlanInspectionPathResponse(
                success=False,
                message="No preview path available"
            )

        self.planned_path = self.preview_path
        self.preview_pub.publish(self.planned_path)

        if self.planned_path.poses:
            self.publish_path_markers(self.planned_path, active_index=0)

        rospy.loginfo(
            "[PlaneInspection] Path committed (%d poses)",
            len(self.planned_path.poses)
        )

        return PlanInspectionPathResponse(
            success=True,
            message="Path planned successfully"
        )

    # ========================================================
    # Path generation
    # ========================================================

    def generate_path_from_cfg(self, cfg):
        """
        Generate a raster inspection path on a plane using
        dynamic reconfigure parameters.
        """

        # --- Plane definition ---
        n = self.normalize(np.array([cfg.normal_x,
                                     cfg.normal_y,
                                     cfg.normal_z]))

        u_raw = np.array([cfg.u_axis_x,
                          cfg.u_axis_y,
                          cfg.u_axis_z])

        u = self.normalize(self.project_onto_plane(u_raw, n))
        v = np.cross(n, u)

        origin = np.array([cfg.origin_x,
                           cfg.origin_y,
                           cfg.origin_z])

        # --- Discretization ---
        nu = cfg.number_steps
        nv = cfg.number_steps

        u_vals = np.linspace(0.0, cfg.u_amplitude, nu)
        v_vals = np.linspace(0.0, cfg.v_amplitude, nv)

        # --- Path message ---
        path = Path()
        path.header.frame_id = "world_ned"
        path.header.stamp = rospy.Time.now()

        zigzag = False
        for v_i in v_vals:
            row = u_vals if not zigzag else u_vals[::-1]
            zigzag = not zigzag

            for u_i in row:
                p = (origin
                     + u * u_i
                     + v * v_i
                     + n * cfg.normal_offset)

                pose = PoseStamped()
                pose.header = path.header
                pose.pose.position.x = p[0]
                pose.pose.position.y = p[1]
                pose.pose.position.z = p[2]

                # Orientation: align yaw with u-axis projection
                yaw = np.arctan2(u[1], u[0])
                q = tf.quaternion_from_euler(0.0, 0.0, yaw)

                pose.pose.orientation.x = q[0]
                pose.pose.orientation.y = q[1]
                pose.pose.orientation.z = q[2]
                pose.pose.orientation.w = q[3]

                path.poses.append(pose)

        return path
# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    rospy.init_node("plane_inspection_node")
    PlaneInspectionNode()
    rospy.spin()
