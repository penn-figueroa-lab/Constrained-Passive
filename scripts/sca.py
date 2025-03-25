import sys
import torch
import torch.nn as nn

import rospy
from std_msgs.msg import String
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import JointState
import numpy as np
from torch.autograd.functional import hessian
import ctypes


class subscriber_sca:
    def __init__(self):
        self.sub_state = rospy.Subscriber("/franka_state_controller/joint_states", JointState, self.QP_callback_sub_joint, queue_size=10)
        self.message_joint_position_sca = [0, 0, 0, 0, 0, 0, 0]
        self.message_joint_velocity_sca = [0, 0, 0, 0, 0, 0, 0]

    def QP_callback_sub_joint(self, msg):
        self.message_joint_position_sca = msg.position
        self.message_joint_velocity_sca = msg.velocity

    def return_message(self):
        return self.message_joint_position_sca, self.message_joint_velocity_sca


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

if __name__ == '__main__':
    rospy.init_node("sca")
    rospy.logwarn("Node initialized!")

    ##### Torch Settings #####
    device = torch.device('cpu')
    m_state_dict = torch.load('/home/zhiquan/franka_ws/src/constr_passive/scripts/SCA_model_state.pt')
    model = BinaryClassifier()
    model.load_state_dict(m_state_dict)

    ##### Subscribing Robot States#####
    Subscriber_sca = subscriber_sca()
    ##### Publishing Message #####
    pub = rospy.Publisher("sca_params", Float32MultiArray, queue_size=10)
    rate = rospy.Rate(200)

    ##### Parameters Definition #####
    alpha1_SCA = 100
    alpha2_SCA = 1000
    epsilon_SCA = 10

    alpha1_JL = 1000
    alpha2_JL = 1000
    epsilon_joint_limit = 0.00
    

    q_min = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
    q_max = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])
    
    while not rospy.is_shutdown():
        ##### Calculating #####
        q = Subscriber_sca.return_message()[0]
        q_dot = Subscriber_sca.return_message()[1]

        inputs = torch.tensor(q, dtype=torch.float32, requires_grad=True).to(device)
        output = model(inputs)[1] - model(inputs)[0]
        dTau = torch.autograd.grad(output, inputs, retain_graph=True, create_graph=True)[0].cpu().detach().numpy()
        hTau = hessian(f,inputs).cpu().detach().numpy()
        righthandside = -alpha1_SCA * (output.cpu().detach().numpy() - epsilon_SCA) - alpha2_SCA * dTau @ np.array(q_dot) - np.array(
            q_dot) @ hTau @ np.transpose(np.array(q_dot))
        
        message = [dTau[0], dTau[1], dTau[2], dTau[3], dTau[4], dTau[5], dTau[6], righthandside]
        msg = Float32MultiArray()
        msg.data = message
        pub.publish(msg)
        # rospy.spin()
        rate.sleep()