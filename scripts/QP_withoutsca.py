#!/usr/bin/env python3

import rospy
import torch
import torch.nn as nn
from std_msgs.msg import String
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, WrenchStamped
import numpy as np
from torch.autograd.functional import hessian
import ctypes
from scipy.io import savemat

class subscriber_QP:
    def __init__(self):
        self.sub_joint = rospy.Subscriber("/franka_state_controller/joint_states", JointState, self.QP_callback_sub_joint, queue_size=10)
        self.sub_ee = rospy.Subscriber("/franka_state_controller/ee_pose", Pose, self.QP_callback_sub_ee, queue_size=10)
        self.sub_state = rospy.Subscriber("/franka_state_controller/franka_model", Float32MultiArray, self.QP_callback_sub_state, queue_size=10)
        self.sub_fext = rospy.Subscriber("/franka_state_controller/F_ext", WrenchStamped, self.QP_callback_sub_fext, queue_size=10)

        self.message_joint_position = [0, 0, 0, 0, 0, 0, 0]
        self.message_joint_velocity = [0, 0, 0, 0, 0, 0, 0]
        self.message_ee = [0, 0, 0]
        self.message_state = list(np.zeros(105))
        self.message_fext = [0, 0, 0]

    def QP_callback_sub_joint(self, msg):
        self.message_joint_position = msg.position
        self.message_joint_velocity = msg.velocity

    def QP_callback_sub_ee(self, msg):
        self.message_ee = [msg.position.x, msg.position.y, msg.position.z]
 
    def QP_callback_sub_state(self, msg):
        self.message_state = msg.data

    def QP_callback_sub_fext(self, msg):
        self.message_fext = [msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z]
    
    def return_message(self):
        return self.message_joint_position, self.message_joint_velocity, self.message_ee, self.message_state, self.message_fext

def f(inputs):
    return model(inputs)[1] - model(inputs)[0]
    
class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.linear1 = nn.Linear(7, 80)
        self.linear2 = nn.Linear(80, 50)
        self.linear3 = nn.Linear(50, 30)
        self.linear4 = nn.Linear(30, 10)
        self.linear5 = nn.Linear(10, 2)
        self.leakyrelu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.linear1(x)
        out = self.tanh(out)
        out = self.linear2(out)
        out = self.tanh(out)
        out = self.linear3(out)
        out = self.tanh(out)
        out = self.linear4(out)
        out = self.tanh(out)
        out = self.linear5(out)
        return out
    

def CVXsolver_in_python(J_in, fc_in, M_in, b_in, dTau_in, b_SCA_in, JL1_in, JL2_in):
    my_c_library = ctypes.CDLL('/home/tianyu/zhiquan_ws/src/constr_passive/scripts/libcvxgen.so')
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

    ##### Torch Settings #####
    device = torch.device('cpu')
    m_state_dict = torch.load('/home/tianyu/zhiquan_ws/src/constr_passive/scripts/SCA_model_state.pt')
    model = BinaryClassifier()
    model.load_state_dict(m_state_dict)

    f_ext_init = Subscriber_QP.return_message()[4]
    # print(f_ext_init)
    loop = 0

    q_save = []
    q_dot_save = []
    end_pos_save = []
    xdot_save = []
    f_ext_save = []

    while not rospy.is_shutdown():
        loop = loop + 1
        if loop == 50:
            f_ext_init = Subscriber_QP.return_message()[4]
        

        ##### Parameter Need #####
        q = Subscriber_QP.return_message()[0]
        q_dot = Subscriber_QP.return_message()[1]
        end_pos = Subscriber_QP.return_message()[2]
        
        Jacobian = np.reshape(np.array(Subscriber_QP.return_message()[3][63: ]), (6, 7), order="F")
        Jacobian = Jacobian[0:3, :].T

        xdot = Jacobian.T @ np.array(q_dot)

        MassMatrix = np.array(Subscriber_QP.return_message()[3][14:63]).reshape((7, 7), order="F")
        Coriolis = np.array(Subscriber_QP.return_message()[3][0:7])
        # Gravity = np.array(Subscriber_QP.return_message()[4][7:14])
        f_ext = Subscriber_QP.return_message()[4]

        q_save.append(q)
        q_dot_save.append(q_dot)
        end_pos_save.append(end_pos)
        xdot_save.append(list(xdot))
        f_ext_save.append(f_ext)

        # print(np.array(f_ext) - np.array(f_ext_init))

        ##### Parameter Definition #####
        lambda1 = 30
        lambda2 = 10
        lambda3 = 10

        q_min = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
        q_max = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])

        alpha1_SCA = 50
        alpha2_SCA = 100
        epsilon_SCA = 0

        alpha1_JL = 30
        alpha2_JL = 50
        epsilon_joint_limit = 0.00

        fx = [-0.5 * (end_pos[0] - 0.1), -0.5 * (end_pos[1] + 0.5), -1 * (end_pos[2] - 0.3)]
        # fx = [-1 * (end_pos[0] - 0.4), -1 * (end_pos[1] + 0.), -1 * (end_pos[2] - 0.4)]

        # print(np.array(f_ext) - np.array(f_ext_init))

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

        ##### SCA Constraints #####
        inputs = torch.tensor(q, dtype=torch.float32, requires_grad=True).to(device)
        output = model(inputs)[1] - model(inputs)[0]
        dTau = torch.autograd.grad(output, inputs, retain_graph=True, create_graph=True)[0].cpu().detach().numpy()
        hTau = hessian(f,inputs).cpu().detach().numpy()
        b_SCA = -alpha1_SCA * (output.cpu().detach().numpy() - epsilon_SCA) - alpha2_SCA * dTau @ np.array(q_dot) - np.array(
            q_dot) @ hTau @ np.transpose(np.array(q_dot))

        ##### Joint Limits Constraints #####
        JL1 = -alpha1_JL*(np.array(q) - q_min - epsilon_joint_limit) - alpha2_JL*np.array(q_dot)
        JL2 = -alpha1_JL*(np.array(q) - q_max + epsilon_joint_limit) - alpha2_JL*np.array(q_dot)
        
        ##### QP Solver #####
        J_in = list(np.linalg.pinv(Jacobian).flatten('F'))
        fc_in = list(fc)
        M_in = list(np.array(MassMatrix).flatten('F'))
        # b_in = list(-np.array(Coriolis))
        b_in = [0, 0, 0, 0, 0, 0, 0]
        # b_in = list(Jacobian@(np.array(f_ext) - np.array(f_ext_init)))
        
        dTau_in = list(dTau)
        b_SCA_in = list(np.array([b_SCA]))
        JL1_in = list(JL1)
        JL2_in = list(JL2)

        result = CVXsolver_in_python(J_in, fc_in, M_in, b_in, dTau_in, b_SCA_in, JL1_in, JL2_in)

        target_torque = [result[0], result[1], result[2], result[3], result[4], result[5], result[6]]
        target_torque = np.clip(target_torque, [-87, -87, -87, -87, -12, -12, -12], [87, 87, 87, 87, 12, 12, 12])
        # print(target_torque)
        target_torque = list(Jacobian @ fc)
        # print(target_torque)
        msg_torque = Float32MultiArray()
        msg_torque.data = target_torque
        # msg_torque.data = [0, 0, 0, 0, 0, 0, 0]
        pub.publish(msg_torque)
        # print(loop)
        # save data
        # if (loop == 15000):
        #     mdic = {"q": np.array(q_save), "q_dot": np.array(q_dot_save), "end_pos": np.array(end_pos_save), "xdot": np.array(xdot_save), 
        #             "f_ext": np.array(f_ext_save), "label": "experiment"}
        #     savemat("/home/zhiquan/franka_ws/src/constr_passive/scripts/sca_fext.mat", mdic)

        # rospy.spin()
        r.sleep()

