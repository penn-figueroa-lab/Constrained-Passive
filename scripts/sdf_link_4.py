#!/usr/bin/env python3

import torch
from torch import nn
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, ELU, ReLU6, Tanh
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, ReLU6, ELU, Dropout, BatchNorm1d as BN, LayerNorm as LN, Tanh
from functorch import vmap, jacrev
from functorch.compile import aot_function

import rospy
from std_msgs.msg import String
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose
import numpy as np

from torch.autograd.functional import hessian
# from sdf.robot_sdf import RobotSdfCollisionNet

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

# message_joint_position_4 = [0, 0, 0, 0, 0, 0, 0]
# message_joint_velocity_4 = [0, 0, 0, 0, 0, 0, 0]

class subscriber_link_4:
    def __init__(self):
        self.sub_state = rospy.Subscriber("/franka_state_controller/joint_states", JointState, self.QP_callback_sub_joint, queue_size=10)
        self.message_joint_position_4 = [0, 0, 0, 0, 0, 0, 0]
        self.message_joint_velocity_4 = [0, 0, 0, 0, 0, 0, 0]

    def QP_callback_sub_joint(self, msg):
        # rospy.loginfo(msg.data)
        self.message_joint_position_4 = msg.position
        self.message_joint_velocity_4 = msg.velocity
        # print(msg.position)
        # print("QP_callback_sub_joint is receiving")

    def return_message(self):
        return self.message_joint_position_4, self.message_joint_velocity_4

def main():
    # def f_4(inputs):
    #     return model(inputs.double())[4]

    # def QP_callback_sub_joint(msg):
    #     # rospy.loginfo(msg.data)
    #     global message_joint_position_4
    #     global message_joint_velocity_4
    #     message_joint_position_4 = msg.position
    #     message_joint_velocity_4 = msg.velocity
    #     # print(msg.position)
    #     # print("QP_callback_sub_joint is receiving")

    rospy.init_node("sdf_link_4")
    rospy.logwarn("Node initialized!")

    ##### Subscribing Robot States#####
    # sub_state = rospy.Subscriber("/franka_state_controller/joint_states", JointState, QP_callback_sub_joint, queue_size=10)
    Subscriber_link_4 = subscriber_link_4()
    ##### Publishing Message #####
    pub = rospy.Publisher("link_4_params", Float32MultiArray, queue_size = 10)
    rate = rospy.Rate(100)
    
    ##### Parameters Definition #####
    alpha1 = 10
    alpha2 = 1000
    rs = 5
    end_pos_ball = [0.35, -0.15, 0.7]
    q_min = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
    q_max = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])

    ##### Torch Settings #####
    # device = torch.device('cpu', 0)
    # tensor_args = {'device': device, 'dtype': torch.double}
    # dof = 10
    # s = 256
    # n_layers = 5
    # skips = []
    # if skips == []:
    #     n_layers-=1
    # nn_model = RobotSdfCollisionNet(in_channels=dof, out_channels=9, layers=[s] * n_layers, skips=skips)
    # nn_model.load_weights('/home/zhiquan/franka_ws/src/constr_passive/scripts/franka_collision_model.pt', tensor_args)

    # model = nn_model.model
    # model = model.double()
    while not rospy.is_shutdown():
        ##### Calculating #####
        # q = message_joint_position_4
        # q_dot = message_joint_velocity_4

        q = Subscriber_link_4.return_message()[0]
        q_dot = Subscriber_link_4.return_message()[1]

        # print("q: ", q)
        # print("q_dot: ", q_dot)
        # inputs = torch.tensor(list(np.hstack((np.array(q), np.array(end_pos_ball)))), dtype=torch.float32, requires_grad=True).to(device)
        # output = model(inputs.double())

        # dTau = torch.autograd.grad(output[4], inputs, retain_graph=True, create_graph=True)[0].detach().numpy()[0:7]
        # hTau = hessian(f_4, inputs).detach().numpy()[0:7, 0:7]
        # b_sdf1 = -alpha1 * (output[4].detach().numpy() - rs) - alpha2 * dTau @ np.transpose(np.array(q_dot)) - np.transpose(np.array(q_dot)) @ hTau @ np.array(q_dot)
        message = [0, 0, 0, 0, 0, 0, 0, 0]
        # message = [dTau[0], dTau[1], dTau[2], dTau[3], dTau[4], dTau[5], dTau[6], b_sdf1]
        # print("message: ", message)
        # rospy.loginfo("sdf_link_4_node Start Publishing!")
        msg = Float32MultiArray()
        msg.data = message
        pub.publish(msg)
        # rospy.spin()
        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass