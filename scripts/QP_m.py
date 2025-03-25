#!/usr/bin/env python3

import rospy

import torch
import torch.nn as nn
from std_msgs.msg import String
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose
import numpy as np
import ctypes

from franka_msgs.msg import FrankaState

class subscriber_QP:
    def __init__(self):
        self.sub_sca = rospy.Subscriber("sca_params", Float32MultiArray, self.QP_callback_sub_sca, queue_size=10)
        self.sub_joint = rospy.Subscriber("/franka_state_controller/joint_states", JointState, self.QP_callback_sub_joint, queue_size=10)
        self.sub_ee = rospy.Subscriber("/franka_state_controller/ee_pose", Pose, self.QP_callback_sub_ee, queue_size=10)
        self.sub_state = rospy.Subscriber("/franka_state_controller/franka_model", Float32MultiArray, self.QP_callback_sub_state, queue_size=10)

        ### Get Franka States --> Needed for ext torques ###
        self.sub_franka_state = rospy.Subscriber("/franka_state_controller/franka_states", FrankaState, self.QP_callback_sub_franka_state, queue_size=10)


        self.message_sca = [0, 0, 0, 0, 0, 0, 0, 0]
        self.message_joint_position = [0, 0, 0, 0, 0, 0, 0]
        self.message_joint_velocity = [0, 0, 0, 0, 0, 0, 0]
        self.message_ee = [0, 0, 0]
        self.message_ee_ori = [0, 0, 0, 0]
        self.message_state = list(np.zeros(105))
        self.message_tau_ext = [0, 0, 0, 0, 0, 0, 0]

    def QP_callback_sub_sca(self, msg):
        self.message_sca = msg.data

    def QP_callback_sub_joint(self, msg):
        self.message_joint_position = msg.position
        self.message_joint_velocity = msg.velocity

    def QP_callback_sub_ee(self, msg):
        self.message_ee = [msg.position.x, msg.position.y, msg.position.z]
        self.message_ee_ori = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]

    def QP_callback_sub_state(self, msg):
        self.message_state = msg.data

    ### Define callback to get external torque measurements from Franka States ###
    def QP_callback_sub_franka_state(self, msg):
        self.message_tau_ext = msg.tau_ext_hat_filtered
    
    def return_message(self):
        return self.message_sca, self.message_joint_position, self.message_joint_velocity, self.message_ee, self.message_state, self.message_tau_ext, self.message_ee_ori


def CVXsolver_in_python(J_in, fc_in, M_in, b_in, JL1_in, JL2_in):
    my_c_library = ctypes.CDLL('/home/tianyu/zhiquan_ws/src/constr_passive/scripts/libcvxgensolver_joint_lim.so')
    my_c_library.CVXsolver.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
    my_c_library.CVXsolver.restype = None

    c_J_in = (ctypes.c_double * len(J_in))(*J_in)
    c_fc_in = (ctypes.c_double * len(fc_in))(*fc_in)
    c_M_in = (ctypes.c_double * len(M_in))(*M_in)
    c_b_in = (ctypes.c_double * len(b_in))(*b_in)


    c_j_l_in = (ctypes.c_double * len(JL1_in))(*JL1_in)
    c_j_u_in = (ctypes.c_double * len(JL2_in))(*JL2_in)
    opt_x1 = ctypes.c_double()
    opt_x2 = ctypes.c_double()
    opt_x3 = ctypes.c_double()
    opt_x4 = ctypes.c_double()
    opt_x5 = ctypes.c_double()
    opt_x6 = ctypes.c_double()
    opt_x7 = ctypes.c_double()

    my_c_library.CVXsolver(c_J_in, c_fc_in, c_M_in, c_b_in, c_j_l_in, c_j_u_in, ctypes.byref(opt_x1), ctypes.byref(opt_x2), ctypes.byref(opt_x3), ctypes.byref(opt_x4), ctypes.byref(opt_x5), ctypes.byref(opt_x6), ctypes.byref(opt_x7))

    result_x = [opt_x1.value, opt_x2.value, opt_x3.value, opt_x4.value, opt_x5.value, opt_x6.value, opt_x7.value]
    return result_x

if __name__ == "__main__":

    rospy.init_node("QP")
    rospy.logwarn("Node initialized!")
    r = rospy.Rate(100)
    
    ##### Subscribing Message #####
    
    Subscriber_QP = subscriber_QP()

    ##### Publishing Message #####
    pub = rospy.Publisher("/joint_gravity_compensation_controller/Control_signals", Float32MultiArray, queue_size = 10)

    while not rospy.is_shutdown():


        ##### Parameter Need #####
        q = Subscriber_QP.return_message()[1]
        
        q_dot = Subscriber_QP.return_message()[2]
        end_pos = Subscriber_QP.return_message()[3]
        end_ori = Subscriber_QP.return_message()[6]

        tau_ext = Subscriber_QP.return_message()[5]
        
        Jacobian = np.reshape(np.array(Subscriber_QP.return_message()[4][63: ]), (6, 7), order="F")
        Jacobian_all = Jacobian
        Jacobian = Jacobian_all[0:3, :].T
        Jacobian_ori = Jacobian_all[3:6, :].T


        xdot = Jacobian.T @ np.array(q_dot)

        omega = Jacobian_ori.T @ np.array(q_dot)
        

        MassMatrix = np.array(Subscriber_QP.return_message()[4][14:63]).reshape((7, 7), order="F")
        Coriolis = np.array(Subscriber_QP.return_message()[4][0:7])
        # Gravity = np.array(Subscriber_QP.return_message()[4][7:14])

        ##### Parameter Definition #####
        
        ### Old Parameters from QP code ###
        #lambda1 = 10
        #lambda2 = 10
        #lambda3 = 10
        ### Parameters from QP without ECA ###
        lambda1 = 20
        lambda2 = 10
        lambda3 = 10

        q_min = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
        q_max = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])

    
        ### Old Parameters from QP code ###
        #alpha1_JL = 1000
        #alpha2_JL = 1000
        ### Parameters from QP without ECA ###  
        alpha1_JL = 30
        alpha2_JL = 50
        
        epsilon_joint_limit = 0.00

        target = np.array([0.5, 0.0, 0.5])
        fx = [-1.5 * (end_pos[0] - target[0]), -1.5 * (end_pos[1] - target[1]), -5.0 * (end_pos[2] - target[2])]
        
        ##### Computing Damping Matrix #####
        e1 = np.array(fx)/np.linalg.norm(fx)
        e2_0 = np.array([1, 0, 0])
        e3_0 = np.array([0, 1, 0])
        e2 = e2_0 - np.dot(e2_0, (e1/np.linalg.norm(e1)))*(e1/np.linalg.norm(e1))
        e2 = e2/np.linalg.norm(e2)
        e3 = e3_0 - np.dot(e3_0, (e1/np.linalg.norm(e1)))*(e1/np.linalg.norm(e1)) - np.dot(e3_0, (e2 / np.linalg.norm(e2))) * (e2 / np.linalg.norm(e2))
        e3 = e3/np.linalg.norm(e3)
        Q = np.zeros([3, 3])
        Q[:, 0] = np.transpose(e1)
        Q[:, 1] = np.transpose(e2)
        Q[:, 2] = np.transpose(e3)

        Lambda = np.diag([lambda1, lambda2, lambda3])

        D = Q @ Lambda @ np.transpose(Q)

        ##### Compute desired target force #####
        fc = -D @ (np.transpose(np.array(xdot)) - np.transpose(fx))




        #### Compute target Wrench ####
        omega_current = omega

        ## Compute f_ori ##
        ori = end_ori
        # print(ori)
        ori_des = [1, 0, 0, 0]
        s1 = ori[3]
        s2 = ori_des[3]
        u1 = np.array([ori[0], ori[1], ori[2]])
        u2 = np.array([ori_des[0], ori_des[1], ori_des[2]])
        Su1 = np.array([[0, -u1[2], u1[1]], [u1[2], 0, -u1[0]], [-u1[1], u1[0], 0]])
        temp = -s1*u2 + s2*u1 - Su1@u2
        dori = np.array([s1*s2+u1@u2.T, temp[0], temp[1], temp[2]])
        logdq = np.arccos(dori[0])*(np.array([dori[1], dori[2], dori[3]]) / np.linalg.norm(np.array([dori[1], dori[2], dori[3]])))
        # omega_num = 2*logdq/stepsize
        f_ori = -15*logdq

        ## Compute fc_wrench ##
        e1 = np.array(f_ori)/np.linalg.norm(f_ori)
        e2_0 = np.array([1, 0, 0])
        e3_0 = np.array([0, 1, 0])
        e2 = e2_0 - np.dot(e2_0, (e1/np.linalg.norm(e1)))*(e1/np.linalg.norm(e1))
        e2 = e2/np.linalg.norm(e2)
        e3 = e3_0 - np.dot(e3_0, (e1/np.linalg.norm(e1)))*(e1/np.linalg.norm(e1)) - np.dot(e3_0, (e2 / np.linalg.norm(e2))) * (e2 / np.linalg.norm(e2))
        e3 = e3/np.linalg.norm(e3)
        Q = np.zeros([3, 3])
        Q[:, 0] = np.transpose(e1)
        Q[:, 1] = np.transpose(e2)
        Q[:, 2] = np.transpose(e3)

        Lambda = np.diag([3, 3, 3])

        D = Q @ Lambda @ np.transpose(Q)
        fc_wrench = -D @ (np.transpose(np.array(omega_current)) - np.transpose(f_ori))



        f_all = np.concatenate((fc, fc_wrench))


        ##### Dynamic Constraints #####
        # b = -message_state[0:7]
        ##### SDF Constraints #####
        # dTau = Subscriber_QP.return_message()[0][0:7]
        # b_SCA = Subscriber_QP.return_message()[0][7]

        ##### Joint Limits Constraints #####
        JL1 = -alpha1_JL*(np.array(q) - q_min - epsilon_joint_limit) - alpha2_JL*np.array(q_dot)
        JL2 = -alpha1_JL*(np.array(q) - q_max + epsilon_joint_limit) - alpha2_JL*np.array(q_dot)
        
        ##### QP Solver #####
        J_in = list(np.linalg.pinv(Jacobian_all.T).flatten('F'))
        fc_in = list(f_all)
        M_in = list(np.array(MassMatrix).flatten('F'))
        b_in = list(-np.array(Coriolis))
        # b_in = [0, 0, 0, 0, 0, 0, 0]
        #dTau_in = list(dTau)
        #b_SCA_in = list(np.array([b_SCA]))
        JL1_in = list(JL1)
        JL2_in = list(JL2)

        result = CVXsolver_in_python(J_in, fc_in, M_in, b_in, JL1_in, JL2_in)

        target_torque = [result[0], result[1], result[2], result[3], result[4], result[5], result[6]]
        target_torque = np.clip(target_torque, [-87, -87, -87, -87, -12, -12, -12], [87, 87, 87, 87, 12, 12, 12])
        print(target_torque)
        # target_torque = list(Jacobian @ fc)
        # print(target_torque)
        msg_torque = Float32MultiArray()
        msg_torque.data = target_torque
        # msg_torque.data = [0, 0, 0, 0, 0, 0, 0]
        pub.publish(msg_torque)
        # rospy.spin()