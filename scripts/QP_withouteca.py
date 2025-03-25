#!/usr/bin/env python3

import rospy
import torch
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, ELU, ReLU6, Tanh
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, ReLU6, ELU, Dropout, BatchNorm1d as BN, LayerNorm as LN, Tanh
from functorch import vmap, jacrev
from functorch.compile import aot_function
from std_msgs.msg import String
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, WrenchStamped
import numpy as np
from torch.autograd.functional import hessian
import ctypes
import matplotlib.pyplot as plt
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

def xavier(param):
    """ initialize weights with xavier.

    Args:
        param (network params): params to initialize.
    """    
    nn.init.xavier_uniform(param)
def he_init(param):
    """initialize weights with he.

    Args:
        param (network params): params to initialize.
    """    
    nn.init.kaiming_uniform_(param,nonlinearity='relu')
    nn.init.normal(param)
def weights_init(m):
    """Function to initialize weights of a nn.

    Args:
        m (network params): pass in model.parameters()
    """    
    fn = he_init
    if isinstance(m, nn.Conv2d):
        fn(m.weight.data)
        m.bias.data.zero_()
    elif isinstance(m, nn.Conv3d):
        fn(m.weight.data)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        fn(m.weight.data)
        if(m.bias is not None):
            m.bias.data.zero_()
def MLP(channels, act_fn=ReLU, islast = False):
    """Automatic generation of mlp given some

    Args:
        channels (int): number of channels in input
        dropout_ratio (float, optional): dropout used after every layer. Defaults to 0.0.
        batch_norm (bool, optional): batch norm after every layer. Defaults to False.
        act_fn ([type], optional): activation function after every layer. Defaults to ReLU.
        layer_norm (bool, optional): layer norm after every layer. Defaults to False.
        nerf (bool, optional): use positional encoding (x->[sin(x),cos(x)]). Defaults to True.

    Returns:
        nn sequential layers
    """
    if not islast:
        layers = [Seq(Lin(channels[i - 1], channels[i]), act_fn())
                  for i in range(1, len(channels))]
    else:
        layers = [Seq(Lin(channels[i - 1], channels[i]), act_fn())
                  for i in range(1, len(channels)-1)]
        layers.append(Seq(Lin(channels[-2], channels[-1])))
    
    layers = Seq(*layers)

    return layers
class MLPRegression(nn.Module):
    def __init__(self, input_dims=10, output_dims=1, mlp_layers=[128, 128, 128, 128, 128],skips=[2], act_fn=ReLU, nerf=True):
        """Create an instance of mlp nn model

        Args:
            input_dims (int): number of channels
            output_dims (int): output channel size
            mlp_layers (list, optional): perceptrons in each layer. Defaults to [256, 128, 128].
            dropout_ratio (float, optional): dropout after every layer. Defaults to 0.0.
            batch_norm (bool, optional): batch norm after every layer. Defaults to False.
            scale_mlp_units (float, optional): Quick way to scale up and down the number of perceptrons, as this gets multiplied with values in mlp_layers. Defaults to 1.0.
            act_fn ([type], optional): activation function after every layer. Defaults to ELU.
            layer_norm (bool, optional): layer norm after every layer. Defaults to False.
            nerf (bool, optional): use positional encoding (x->[sin(x),cos(x)]). Defaults to False.
        """        
        super(MLPRegression, self).__init__()

        mlp_arr = []
        if (nerf):
            input_dims = 3*input_dims
        if len(skips)>0:
            mlp_arr.append(mlp_layers[0:skips[0]])
            mlp_arr[0][-1]-=input_dims
            for s in range(1,len(skips)):
                mlp_arr.append(mlp_layers[skips[s-1]:skips[s]])
                mlp_arr[-1][-1]-=input_dims
            mlp_arr.append(mlp_layers[skips[-1]:])
        else:
            mlp_arr.append(mlp_layers)

        mlp_arr[-1].append(output_dims)

        mlp_arr[0].insert(0,input_dims)
        self.layers = nn.ModuleList()
        for arr in mlp_arr[0:-1]:
            self.layers.append(MLP(arr,act_fn=act_fn, islast=False))
        self.layers.append(MLP(mlp_arr[-1],act_fn=act_fn, islast=True))

        self.nerf = nerf
    
    def forward(self, x):
        """forward pass on network."""        
        if(self.nerf):
            x_nerf = torch.cat((x, torch.sin(x), torch.cos(x)), dim=-1)
        else:
            x_nerf = x
        y = self.layers[0](x_nerf)
        for layer in self.layers[1:]:
            y = layer(torch.cat((y, x_nerf), dim=1))
        return y
    def reset_parameters(self):
        """Use this function to initialize weights. Doesn't help much for mlp.
        """        
        self.apply(weights_init)
def scale_to_base(data, norm_dict, key):
    """Scale the tensor back to the orginal units.  

    Args:
        data (tensor): input tensor to scale
        norm_dict (Dict): normalization dictionary of the form dict={key:{'mean':,'std':}}
        key (str): key of the data

    Returns:
        tensor : output scaled tensor
    """    
    scaled_data = torch.mul(data,norm_dict[key]['std']) + norm_dict[key]['mean']
    return scaled_data   
def scale_to_net(data, norm_dict, key):
    """Scale the tensor network range

    Args:
        data (tensor): input tensor to scale
        norm_dict (Dict): normalization dictionary of the form dict={key:{'mean':,'std':}}
        key (str): key of the data

    Returns:
        tensor : output scaled tensor
    """    
    
    scaled_data = torch.div(data - norm_dict[key]['mean'],norm_dict[key]['std'])
    scaled_data[scaled_data != scaled_data] = 0.0
    return scaled_data
class RobotSdfCollisionNet():
    """This class loads a network to predict the signed distance given a robot joint config."""
    def __init__(self, in_channels, out_channels, skips, layers):

        super().__init__()
        act_fn = ReLU
        in_channels = in_channels
        self.out_channels = out_channels
        dropout_ratio = 0
        mlp_layers = layers
        self.model = MLPRegression(in_channels, self.out_channels, mlp_layers, skips, act_fn=act_fn, nerf=True)
        self.m = torch.zeros((500, 1)).to('cpu:0')
        self.m[:, 0] = 1
        self.order = list(range(out_channels))

    def set_link_order(self, order):
        self.order = order

    def load_weights(self, f_name, tensor_args):
        """Loads pretrained network weights if available.

        Args:
            f_name (str): file name, this is relative to weights folder in this repo.
            tensor_args (Dict): device and dtype for pytorch tensors
        """
        try:
            chk = torch.load(f_name, map_location=torch.device('cpu'))
            self.model.load_state_dict(chk["model_state_dict"])
            self.norm_dict = chk["norm"]
            for k in self.norm_dict.keys():
                self.norm_dict[k]['mean'] = self.norm_dict[k]['mean'].to(**tensor_args)
                self.norm_dict[k]['std'] = self.norm_dict[k]['std'].to(**tensor_args)
            print('Weights loaded!')
        except Exception as E:
            print('WARNING: Weights not loaded')
            print(E)
        self.model = self.model.to(**tensor_args)
        self.tensor_args = tensor_args
        self.model.eval()

    def compute_signed_distance(self, q):
        """Compute the signed distance given the joint config.

        Args:
            q (tensor): input batch of joint configs [b, n_joints]

        Returns:
            [tensor]: largest signed distance between any two non-consecutive links of the robot.
        """
        with torch.no_grad():
            q_scale = scale_to_net(q, self.norm_dict, 'x')
            dist = self.model.forward(q_scale)
            dist_scale = scale_to_base(dist, self.norm_dict, 'y')
        return dist_scale[:, self.order].detach()

    def compute_signed_distance_wgrad(self, q, idx = 'all'):
        minidxMask = torch.zeros(q.shape[0])
        if idx == 'all':
            idx = list(range(self.out_channels))
        if self.out_channels == 1:
            with torch.enable_grad():
                q.requires_grad = True
                q.grad = None
                q_scale = scale_to_net(q, self.norm_dict, 'x')
                dist = self.model.forward(q_scale)
                dist_scale = scale_to_base(dist, self.norm_dict, 'y').detach()
                m = torch.zeros((q.shape[0], dist.shape[1])).to(q.device)
                m[:, 0] = 1
                dist.backward(m)
                grads = q.grad.detach()
                # jac = torch.autograd.functional.jacobian(self.model, q_scale)
        else:
            with torch.enable_grad():
                #https://discuss.pytorch.org/t/derivative-of-model-outputs-w-r-t-input-features/95814/2
                q.requires_grad = True
                q.grad = None
                #removed scaling as we don't use it
                dist_scale = self.model.forward(q)
                dist_scale = dist_scale[:, self.order]
                minidxMask = torch.argmin(dist_scale, dim=1)
                grd = torch.zeros((q.shape[0], self.out_channels), device = q.device, dtype = q.dtype) # same shape as preds
                if type(idx) == list:
                    grads = torch.zeros((q.shape[0], q.shape[1], len(idx)))
                    for k, i in enumerate(idx):
                        grd *= 0
                        grd[:, i] = 1  # column of Jacobian to compute
                        dist_scale.backward(gradient=grd, retain_graph=True)
                        grads[:, :, k] = q.grad  # fill in one column of Jacobian
                        q.grad.zero_()  # .backward() accumulates gradients, so reset to zero
                else:
                    grads = torch.zeros((q.shape[0], q.shape[1], 1))
                    grd[list(range(q.shape[0])), minidxMask] = 1
                    dist_scale.backward(gradient=grd, retain_graph=False)
                    grads[:, :, 0] = q.grad  # fill in one column of Jacobian
                    #q.grad.zero_()  # .backward() accumulates gradients, so reset to zero
                    for param in self.model.parameters():
                        param.grad = None
        return dist_scale.detach(), grads.detach(), minidxMask.detach()

    def functorch_jacobian(self, points):
        """calculate a jacobian tensor along a batch of inputs. returns something of size
        `batch_size` x `output_dim` x `input_dim`"""
        return vmap(jacrev(self.model))(points)

    def pytorch_jacobian(self, points):
        """calculate a jacobian tensor along a batch of inputs. returns something of size
        `batch_size` x `output_dim` x `input_dim`"""
        def _func_sum(points):
            return self.model(points).sum(dim=0)
        return torch.autograd.functional.jacobian(_func_sum, points, create_graph=True, vectorize=True).permute(1, 0, 2)

    def functorch_jacobian2(self, points):
        """calculate a jacobian tensor along a batch of inputs. returns something of size
        `batch_size` x `output_dim` x `input_dim`"""
        def _func_sum(points):
            return self.model(points).sum(dim=0)
        return jacrev(_func_sum)(points).permute(1, 0, 2)

    def ts_compile(self, fx_g, inps):
        print("compiling")
        f = torch.jit.script(fx_g)
        f = torch.jit.freeze(f.eval())
        return f

    def ts_compiler(self, f):
        return aot_function(f, self.ts_compile, self.ts_compile)

def f_4(inputs):
    return model(inputs.double())[4]



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
    r = rospy.Rate(100)
    
    ##### Subscribing Message #####
    
    Subscriber_QP = subscriber_QP()

    ##### Publishing Message #####
    pub = rospy.Publisher("/joint_gravity_compensation_controller/Control_signals", Float32MultiArray, queue_size = 10)

    ##### Torch Settings #####
    device = torch.device('cpu', 0)
    tensor_args = {'device': device, 'dtype': torch.double}
    dof = 10
    s = 256
    n_layers = 5
    skips = []
    if skips == []:
        n_layers-=1
    nn_model = RobotSdfCollisionNet(in_channels=dof, out_channels=9, layers=[s] * n_layers, skips=skips)
    nn_model.load_weights('/home/tianyu/zhiquan_ws/src/constr_passive/scripts/franka_collision_model.pt', tensor_args)
    model = nn_model.model
    model = model.double()

    ## plot
    loop = 0

    q_save = []
    q_dot_save = []
    end_pos_save = []
    xdot_save = []
    f_ext_save = []

    while not rospy.is_shutdown():
        loop = loop + 1
        ##### Parameter Need #####
        q = Subscriber_QP.return_message()[0]
        q_dot = Subscriber_QP.return_message()[1]
        end_pos = Subscriber_QP.return_message()[2]
        Jacobian = np.reshape(np.array(Subscriber_QP.return_message()[3][63: ]), (6, 7), order="F")
        Jacobian = Jacobian[0:3, :].T
        xdot = Jacobian.T @ np.array(q_dot)
        MassMatrix = np.array(Subscriber_QP.return_message()[3][14:63]).reshape((7, 7), order="F")
        Coriolis = np.array(Subscriber_QP.return_message()[3][0:7])
        Gravity = np.array(Subscriber_QP.return_message()[3][7:14])
        f_ext = Subscriber_QP.return_message()[4]

        q_save.append(q)
        q_dot_save.append(q_dot)
        end_pos_save.append(end_pos)
        xdot_save.append(list(xdot))
        f_ext_save.append(f_ext)

        ##### Parameter Definition #####
        lambda1 = 20
        lambda2 = 10
        lambda3 = 10

        q_min = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
        q_max = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])

        alpha1_eca = 1
        alpha2_eca = 100
        rs = 10
        end_pos_ball = [0.25, -0.25, 0.6]

        alpha1_JL = 30
        alpha2_JL = 50
        epsilon_joint_limit = 0.00

        # fx = [-0.1 * (end_pos[0] - 0.1), -0.1 * (end_pos[1] + 0.5), -0.1 * (end_pos[2] - 0.2)]
        # fx = [-1 * (end_pos[0] - 0.4), -1 * (end_pos[1] + 0.), -1 * (end_pos[2] - 0.4)]
        # fx = [-0.1 * (end_pos[0] - 0.5), -0.1 * (end_pos[1] + 0.5), -0.1 * (end_pos[2] - 0.5)]
        # fx = [-0.1 * (end_pos[0] - 0.5), -0.1 * (end_pos[1] + 0), -0.1 * (end_pos[2] - 0.5)]
        # fx = [-1 * (end_pos[0] - 0.1), -1 * (end_pos[1] + 0.5), -1 * (end_pos[2] - 0.2)]
        fx = [-2.5 * (end_pos[0] - 0.7), -2.5 * (end_pos[1] + 0), -2.5 * (end_pos[2] - 0.5)]


        # print(end_pos)
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
        
        ##### ECA Constraints #####
        inputs = torch.tensor(list(np.hstack((np.array(q), np.array(end_pos_ball)))), dtype=torch.float32, requires_grad=True).to(device)
        output = model(inputs.double())

        dTau = torch.autograd.grad(output[4], inputs, retain_graph=True, create_graph=True)[0].detach().numpy()[0:7]
        hTau = hessian(f_4, inputs).detach().numpy()[0:7, 0:7]
        b_sdf1 = -alpha1_eca * (output[4].detach().numpy() - rs) - alpha2_eca * dTau @ np.transpose(np.array(q_dot)) - np.transpose(np.array(q_dot)) @ hTau @ np.array(q_dot)

        ##### Joint Limits Constraints #####
        JL1 = -alpha1_JL*(np.array(q) - q_min - epsilon_joint_limit) - alpha2_JL*np.array(q_dot)
        JL2 = -alpha1_JL*(np.array(q) - q_max + epsilon_joint_limit) - alpha2_JL*np.array(q_dot)
        
        ##### QP Solver #####
        J_in = list(np.linalg.pinv(Jacobian).flatten('F'))
        fc_in = list(fc)
        M_in = list(np.array(MassMatrix).flatten('F'))
        # b_in = list(-np.array(Coriolis))
        # b_in = [0, 0, 0, 0, 0, 0, 0]
        b_in = list(-Jacobian@(np.array(f_ext)))
        dTau_in = list(dTau)
        b_SCA_in = list(np.array([b_sdf1]))
        JL1_in = list(JL1)
        JL2_in = list(JL2)

        result = CVXsolver_in_python(J_in, fc_in, M_in, b_in, dTau_in, b_SCA_in, JL1_in, JL2_in)

        target_torque = [result[0], result[1], result[2], result[3], result[4], result[5], result[6]]
        target_torque = np.clip(target_torque, [-87, -87, -87, -87, -12, -12, -12], [87, 87, 87, 87, 12, 12, 12])
        
        # print(target_torque)
        # target_torque = list(np.linalg.pinv(Jacobian.T) @ fc)
        target_torque = list((Jacobian) @ fc)
        # print(target_torque)
        msg_torque = Float32MultiArray()
        msg_torque.data = target_torque
        # msg_torque.data = [0, 0, 0, 0, 0, 0, 0]
        # msg_torque.data = -Gravity
        pub.publish(msg_torque)

        # # save data
        # if (loop == 3000):
        #     mdic = {"q": np.array(q_save), "q_dot": np.array(q_dot_save), "end_pos": np.array(end_pos_save), "xdot": np.array(xdot_save), 
        #             "f_ext": np.array(f_ext_save), "label": "experiment"}
        #     savemat("/home/zhiquan/franka_ws/src/constr_passive/scripts/eca_fext.mat", mdic)

        # rospy.spin()
        r.sleep()

