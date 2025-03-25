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

class subscriber_QP:
    def __init__(self):
        self.sub_sca = rospy.Subscriber("sca_params", Float32MultiArray, self.QP_callback_sub_sca, queue_size=10)
        self.sub_joint = rospy.Subscriber("/franka_state_controller/joint_states", JointState, self.QP_callback_sub_joint, queue_size=10)
        self.sub_ee = rospy.Subscriber("/franka_state_controller/ee_pose", Pose, self.QP_callback_sub_ee, queue_size=10)
        self.sub_state = rospy.Subscriber("/franka_state_controller/franka_model", Float32MultiArray, self.QP_callback_sub_state, queue_size=10)

        self.message_sca = [0, 0, 0, 0, 0, 0, 0, 0]
        self.message_joint_position = [0, 0, 0, 0, 0, 0, 0]
        self.message_joint_velocity = [0, 0, 0, 0, 0, 0, 0]
        self.message_ee = [0, 0, 0]
        self.message_state = list(np.zeros(105))

    def QP_callback_sub_sca(self, msg):
        self.message_sca = msg.data

    def QP_callback_sub_joint(self, msg):
        self.message_joint_position = msg.position
        self.message_joint_velocity = msg.velocity

    def QP_callback_sub_ee(self, msg):
        self.message_ee = [msg.position.x, msg.position.y, msg.position.z]
 
    def QP_callback_sub_state(self, msg):
        self.message_state = msg.data
    
    def return_message(self):
        return self.message_sca, self.message_joint_position, self.message_joint_velocity, self.message_ee, self.message_state


def CVXsolver_in_python(J_in, fc_in, M_in, b_in, dTau_in, b_SCA_in, JL1_in, JL2_in):
    my_c_library = ctypes.CDLL('/home/zhiquan/franka_ws/src/constr_passive/scripts/libcvxgen.so')
    my_c_library.CVXsolver.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                    ctypes.POINTER(ctypes.c_double)]
    my_c_library.CVXsolver.restype = None

    c_J_in = (ctypes.c_double * len(J_in))(*J_in)
    c_fc_in = (ctypes.c_double * len(fc_in))(*fc_in)
    c_M_in = (ctypes.c_double * len(M_in))(*M_in)
    c_b_in = (ctypes.c_double * len(b_in))(*b_in)
    c_dTau_in = (ctypes.c_double * len(dTau_in))(*dTau_in)
    c_b_SCA_in = (ctypes.c_double * len(b_SCA_in))(*b_SCA_in)
    c_JL1_in = (ctypes.c_double * len(JL1_in))(*JL1_in)
    c_JL2_in = (ctypes.c_double * len(JL2_in))(*JL2_in)
    opt_x1 = ctypes.c_double()
    opt_x2 = ctypes.c_double()
    opt_x3 = ctypes.c_double()
    opt_x4 = ctypes.c_double()
    opt_x5 = ctypes.c_double()
    opt_x6 = ctypes.c_double()
    opt_x7 = ctypes.c_double()
    opt_x7 = ctypes.c_double()

    my_c_library.CVXsolver(c_J_in, c_fc_in, c_M_in, c_b_in, c_dTau_in, c_b_SCA_in, c_JL1_in, c_JL2_in, ctypes.byref(opt_x1), ctypes.byref(opt_x2),
                           ctypes.byref(opt_x3), ctypes.byref(opt_x4), ctypes.byref(opt_x5), ctypes.byref(opt_x6), ctypes.byref(opt_x7))

    result_x = [opt_x1.value, opt_x2.value, opt_x3.value, opt_x4.value, opt_x5.value, opt_x6.value, opt_x7.value]
    return result_x

if __name__ == "__main__":

    rospy.init_node("QP")
    rospy.logwarn("Node initialized!")
    r = rospy.Rate(200)
    
    ##### Subscribing Message #####
    
    Subscriber_QP = subscriber_QP()

    ##### Publishing Message #####
    pub = rospy.Publisher("/joint_gravity_compensation_controller/Control_signals", Float32MultiArray, queue_size = 10)

    while not rospy.is_shutdown():
        
        ##### Parameter Need #####
        q = Subscriber_QP.return_message()[1]
        
        q_dot = Subscriber_QP.return_message()[2]
        end_pos = Subscriber_QP.return_message()[3]
    
        
        Jacobian = np.reshape(np.array(Subscriber_QP.return_message()[4][63: ]), (6, 7), order="F")
        Jacobian = Jacobian[0:3, :].T

        xdot = Jacobian.T @ np.array(q_dot)

        MassMatrix = np.array(Subscriber_QP.return_message()[4][14:63]).reshape((7, 7), order="F")
        Coriolis = np.array(Subscriber_QP.return_message()[4][0:7])
        # Gravity = np.array(Subscriber_QP.return_message()[4][7:14])

        # print('q: ', q)
        # print('Jacobian: ', Jacobian)
        # print('MassMatrix: ', MassMatrix)
        ##### Parameter Definition #####
        lambda1 = 10
        lambda2 = 10
        lambda3 = 10

        q_min = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
        q_max = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])

        # alpha1_SCA = 10
        # alpha2_SCA = 100
        # epsilon_SCA = 10

        # alpha1_JL = 1000
        # alpha2_JL = 1000
        # epsilon_joint_limit = 0.00

        alpha1_SCA = 10
        alpha2_SCA = 100
        epsilon_SCA = 10

        alpha1_JL = 1000
        alpha2_JL = 1000
        epsilon_joint_limit = 0.00

        fx = [-0.1 * (end_pos[0] - 0.), -0.1 * (end_pos[1] + 0.5), -0.1 * (end_pos[2] - 0.3)]
        
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

        ##### Dynamic Constraints #####
        # b = -message_state[0:7]
        ##### SDF Constraints #####
        dTau = Subscriber_QP.return_message()[0][0:7]
        b_SCA = Subscriber_QP.return_message()[0][7]

        ##### Joint Limits Constraints #####
        JL1 = -alpha1_JL*(np.array(q) - q_min - epsilon_joint_limit) - alpha2_JL*np.array(q_dot)
        JL2 = -alpha1_JL*(np.array(q) - q_max + epsilon_joint_limit) - alpha2_JL*np.array(q_dot)
        
        ##### QP Solver #####
        J_in = list(np.linalg.pinv(Jacobian).flatten('F'))
        fc_in = list(fc)
        M_in = list(np.array(MassMatrix).flatten('F'))
        b_in = list(-np.array(Coriolis))
        # b_in = [0, 0, 0, 0, 0, 0, 0]
        dTau_in = list(dTau)
        b_SCA_in = list(np.array([b_SCA]))
        JL1_in = list(JL1)
        JL2_in = list(JL2)

        result = CVXsolver_in_python(J_in, fc_in, M_in, b_in, dTau_in, b_SCA_in, JL1_in, JL2_in)

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
        r.sleep()



# q:  (-0.019449076101273555, -0.2247923844410824, 0.018610246309024525, -2.096102759231723, 0.002234242317879165, 1.8863527956604287, 0.8025995226267915)
# Jacobian:  [[-2.95867940e-04  4.90805805e-01  0.00000000e+00]
#             [ 1.85853601e-01 -3.61513649e-03 -4.90707248e-01]
#             [ 5.17404289e-04  5.19884825e-01 -2.19358248e-03]
#             [ 1.40550211e-01  1.81072077e-03  4.80801910e-01]
#             [ 1.46460603e-04  7.44060501e-02 -1.49852858e-04]
#             [ 1.05667554e-01 -2.75512557e-05  8.95955786e-02]
#             [ 5.42101086e-20  1.64798730e-17 -3.30342849e-20]]
# MassMatrix:  [[ 8.89820516e-01 -4.21661846e-02  9.57559049e-01 -1.30570633e-03 -7.28655257e-04  1.79774140e-03 -8.64851661e-03]
#               [-4.21661846e-02  1.82086396e+00 -3.47757936e-02 -7.96689570e-01 -1.67717244e-02 -8.28963295e-02  6.25336717e-04]
#               [ 9.57559049e-01 -3.47757936e-02  1.10796905e+00 -9.13086999e-03 -6.74638525e-03  1.21765991e-03 -8.77156574e-03]
#               [-1.30570633e-03 -7.96689570e-01 -9.13086999e-03  8.08008552e-01  2.28721369e-02  1.02818839e-01 -1.74086378e-03]
#               [-7.28655257e-04 -1.67717244e-02 -6.74638525e-03  2.28721369e-02  2.46446468e-02  2.60146451e-04  1.00206654e-03]
#               [ 1.79774140e-03 -8.28963295e-02  1.21765991e-03  1.02818839e-01  2.60146451e-04  3.25885750e-02 -1.58381218e-03]
#               [-8.64851661e-03  6.25336717e-04 -8.77156574e-03 -1.74086378e-03  1.00206654e-03 -1.58381218e-03  4.90965182e-03]]

# Jacobian:  ((-0.00029586793336647347, 0.18585359348738056, 0.0005174042829728595, 0.14055021793174902, 0.00014646061010289627, 0.10566755073849579, 0.0), 
#             (0.49080581385811645, -0.003615136520943636, 0.5198848437888052, 0.0018107207918782748, 0.07440604956241437, -2.7551256189872492e-05, -0.0), 
#             (2.168404344971009e-19, -0.4907072350753251, -0.002193582371399228, 0.4808019106440025, -0.0001498528585632503, 0.0895955800352657, 0.0))
# MassMatrix:  ((1.1011239541422713, -0.01444631047417716, 1.1820227295658279, 0.00395346682652323, 0.025776716240619533, 2.7882351970285725e-06, -0.0011646735901551146),
#               (-0.01444631047417716, 2.052898630250725, -0.009358288490296851, -0.95557354943255, 0.00013469129133201156, -0.04771243804292176, -2.0397388324155624e-06), 
#               (1.1820227295658279, -0.009358288490296851, 1.3272713060307113, -2.627877945021765e-05, 0.024079511718964934, -0.0002078762777554104, -0.0011392756492088108), 
#               (0.00395346682652323, -0.95557354943255, -2.627877945021765e-05, 1.0300306139923512, 6.992560133975741e-05, 0.08078815929269387, -2.4739596957912084e-06), 
#               (0.025776716240619533, 0.00013469129133201156, 0.024079511718964934, 6.992560133975741e-05, 0.022026005762753566, -5.214933810752028e-07, 0.0003614926586841016), 
#               (2.7882351970285725e-06, -0.04771243804292176, -0.0002078762777554104, 0.08078815929269387, -5.214933810752028e-07, 0.028119410448157363, 5.7036395142184674e-15), 
#               (-0.0011646735901551146, -2.0397388324155624e-06, -0.0011392756492088108, -2.4739596957912084e-06, 0.0003614926586841016, 5.7036395142184674e-15, 0.0011648071098918457))
