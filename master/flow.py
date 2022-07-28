import torch
import torch.nn as nn
from torch import autograd
import torch.nn.functional as F
import numpy as np


class NaturalGradientDescentVelNet(nn.Module):
	"""
	taskmap_fcn: map to a latent space
	grad_taskmap_fcn: jacobian of the map
	grad_potential_fcn: gradient of a potential fcn defined on the mapped space
	n_dim_x: observed (input) space dimensions
	n_dim_y: latent (output) space dimentions
	origin (optional): shifted origin of the input space (this is the goal usually)
	scale_vel (optional): if set to true, learns a scalar velocity multiplier
	is_diffeomorphism (optional): if set to True, use the inverse of the jacobian itself rather than pseudo-inverse
	"""
	def __init__(self, taskmap_fcn, grad_potential_fcn, n_dim_x, n_dim_y,
				  eps=1e-12, device='cpu',dt = 0.01):

		super(NaturalGradientDescentVelNet, self).__init__()
		self.taskmap_fcn = taskmap_fcn
		# self.grad_potential_fcn = grad_potential_fcn
		self.n_dim_x = n_dim_x
		self.n_dim_y = n_dim_y
		self.eps = eps
		self.linear = nn.Linear(1,1,bias=None)
		self.dt = dt




	def forward(self, x,via_point=None):

		y_hat = self.taskmap_fcn(x)
		y_hat = y_hat.reshape(-1,2)

		v_now = torch.bmm(y_hat.unsqueeze(1),y_hat.unsqueeze(2)).squeeze(1).squeeze(1)
	
		if via_point is not None:
			via_point = via_point.unsqueeze(0)
			via_point_plane = self.taskmap_fcn(via_point)
			k_plane = via_point_plane[:,1]/via_point_plane[:,0]

			wg = torch.sqrt(k_plane**2+1)

			v_via_point = torch.mm(via_point_plane,via_point_plane.transpose(0,1)).squeeze(1)
			if v_now >= v_via_point/2:
				x_plane_now = torch.sqrt(v_now)/wg

				flg = F.relu(-via_point_plane[0][0]) / (via_point_plane[0][0] + 1e-8)  # 正为0 负为-1

				x_plane_now = x_plane_now + 2 * flg * x_plane_now
				y_plane_now = x_plane_now*k_plane

				x_plane_now = x_plane_now.unsqueeze(1)
				y_plane_now = y_plane_now.unsqueeze(1)
				xy_plane_now = torch.cat((x_plane_now,y_plane_now),1)
				xy_plane_now_v = -F.normalize(xy_plane_now)

				xy_plane_next = xy_plane_now+0.2*xy_plane_now_v

				yd_hat = F.normalize(xy_plane_next-y_hat)
				y_hat_now = y_hat + 0.01 * yd_hat

				y_hat_now1 = y_hat_now.unsqueeze(2)
				y_hat1 = y_hat.unsqueeze(2)
				y_hat_flag = -torch.bmm(y_hat_now1.transpose(1, 2), y_hat1).squeeze(1)  # 800 1 1
				y_hat_flag_relu = F.relu(y_hat_flag) / (y_hat_flag + 1e-8)
				y_hat_flag_reluu = y_hat_flag_relu.repeat(1, y_hat_flag_relu.shape[1])
				y_hat_now = y_hat_now - y_hat_flag_reluu * y_hat_now

				x_hat_now = self.taskmap_fcn(y_hat_now, mode="inverse")
				xd = (x_hat_now - x) / self.dt
				# yy=self.taskmap_fcn(x_hat_now)


				return xd, y_hat
			else:
				yd_hat = -F.normalize(y_hat)
				y_hat_now = y_hat + 0.01 * yd_hat

				y_hat_now1 = y_hat_now.unsqueeze(2)
				y_hat1 = y_hat.unsqueeze(2)
				y_hat_flag = -torch.bmm(y_hat_now1.transpose(1, 2), y_hat1).squeeze(1)  # 800 1 1
				y_hat_flag_relu = F.relu(y_hat_flag) / (y_hat_flag + 1e-8)
				y_hat_flag_reluu = y_hat_flag_relu.repeat(1, y_hat_flag_relu.shape[1])
				y_hat_now = y_hat_now - y_hat_flag_reluu * y_hat_now

				x_hat_now = self.taskmap_fcn(y_hat_now, mode="inverse")
				# print(self.taskmap_fcn(x_hat_now)-y_hat_now)

				xd = (x_hat_now - x) / self.dt
				# yy = self.taskmap_fcn(x_hat_now)
				return xd, y_hat

		#

		#
		else:
			yd_hat = -F.normalize(y_hat,dim=1)  # negative gradient of potential

			y_hat_now = y_hat + 0.01 * yd_hat
			# print(F.normalize(y_hat_now)+yd_hat)

			y_hat_now1 = y_hat_now.unsqueeze(2)
			y_hat1 = y_hat.unsqueeze(2)
			y_hat_flag = -torch.bmm(y_hat_now1.transpose(1,2),y_hat1).squeeze(1)# 800 1 1
			y_hat_flag_relu = F.relu(y_hat_flag)/(y_hat_flag+1e-8)
			y_hat_flag_reluu = y_hat_flag_relu.repeat(1,y_hat_flag_relu.shape[1])
			y_hat_now = y_hat_now - y_hat_flag_reluu*y_hat_now

			x_hat_now = self.taskmap_fcn(y_hat_now,mode="inverse")
			xd = (x_hat_now-x)/self.dt
			# yy = self.taskmap_fcn(x_hat_now)
			# print(yy)

			return xd,y_hat


class BijectionNet(nn.Sequential):
	"""
	A sequential container of flows based on coupling layers.
	"""
	def __init__(self, num_dims, num_blocks, num_hidden, s_act=None, t_act=None, sigma=None,
				 coupling_network_type='fcnn'):
		self.num_dims = num_dims
		modules = []
		# print('Using the {} for coupling layer'.format(coupling_network_type))
		mask = torch.arange(0, num_dims) % 2  # alternating inputs
		mask = mask.float()
		# mask = mask.to(device).float()
		for _ in range(num_blocks):
			modules += [
				CouplingLayer(
					num_inputs=num_dims, num_hidden=num_hidden, mask=mask,
					s_act=s_act, t_act=t_act, sigma=sigma, base_network=coupling_network_type),
			]
			mask = 1 - mask  # flipping mask
		super(BijectionNet, self).__init__(*modules)



	def forward(self, inputs, mode='direct'):
		""" Performs a forward or backward pass for flow modules.
		Args:
			inputs: a tuple of inputs and logdets
			mode: to run direct computation or inverse
		"""
		assert mode in ['direct', 'inverse']
		# batch_size = inputs.size(0)
		# J = torch.eye(self.num_dims, device=inputs.device).unsqueeze(0).repeat(batch_size, 1, 1)

		if mode == 'direct':
			for module in self._modules.values():
				# J_module = module.jacobian(inputs)
				# J = torch.matmul(J_module, J)
				inputs = module(inputs, mode)
		else:
			for module in reversed(self._modules.values()):
				# J_module = module.jacobian(inputs)
				# J = torch.matmul(J_module, J)
				inputs = module(inputs, mode)
		return inputs



class CouplingLayer(nn.Module):
	""" An implementation of a coupling layer
	from RealNVP (https://arxiv.org/abs/1605.08803).
	"""

	def __init__(self, num_inputs, num_hidden, mask,
				 base_network='rffn', s_act='elu', t_act='elu', sigma=0.45):
		super(CouplingLayer, self).__init__()

		self.num_inputs = num_inputs
		self.mask = mask

		if base_network == 'fcnn':
			self.scale_net = FCNN(in_dim=2, out_dim=2, hidden_dim=num_hidden, act=s_act)
			self.translate_net = FCNN(in_dim=2, out_dim=2, hidden_dim=num_hidden, act=t_act)
			# print('Using neural network initialized with identity map!')

			nn.init.zeros_(self.translate_net.network[-1].weight.data)
			nn.init.zeros_(self.translate_net.network[-1].bias.data)


		elif base_network == 'rffn':
			print('Using random fouier feature with bandwidth = {}.'.format(sigma))
			self.scale_net = RFFN(in_dim=num_inputs, out_dim=num_inputs, nfeat=num_hidden, sigma=sigma)
			self.translate_net = RFFN1(in_dim=num_inputs, out_dim=num_inputs, nfeat=num_hidden, sigma=sigma)

			print('Initializing coupling layers as identity!')
			nn.init.zeros_(self.translate_net.network[-1].weight.data)
			nn.init.zeros_(self.scale_net.network[-1].weight.data)
			# nn.init.zeros_(self.translate_net.network[-1].weight.data)
			# nn.init.zeros_(self.scale_net.network[-1].weight.data)
		else:
			raise TypeError('The network type has not been defined')

	def forward(self, inputs, mode='direct'):
		mask = self.mask
		masked_inputs = inputs * mask
		# masked_inputs.requires_grad_(True)

		log_s =self.scale_net(masked_inputs) * (1 - mask)
		t =self.translate_net(masked_inputs)* (1 - mask)

		if mode == 'direct':
			s = torch.exp(log_s)
			return inputs * s + t
		else:
			s = torch.exp(-log_s)
			return (inputs - t) * s

class RFFN(nn.Module):
	"""
	Random Fourier features network.
	"""

	def __init__(self, in_dim, out_dim, nfeat, sigma=10.): # 2 2 100
		super(RFFN, self).__init__()
		self.sigma = np.ones(in_dim) * sigma #
		self.coeff = np.random.normal(0.0, 2, (nfeat, in_dim))
		self.coeff = self.coeff / self.sigma.reshape(1, len(self.sigma))
		self.offset = 2.0 * np.pi * np.random.rand(1, nfeat)

		self.network = nn.Sequential(
			LinearClamped(in_dim, nfeat, self.coeff, self.offset),
			Cos(),
			nn.Linear(nfeat, out_dim, bias=True),
			# nn.ELU(),
			# nn.Linear(out_dim, out_dim, bias=True),

		)

	def forward(self, x):
		return self.network(x)


class RFFN1(nn.Module):
	"""
	Random Fourier features network.
	"""

	def __init__(self, in_dim, out_dim, nfeat, sigma=10.): # 2 2 100
		super(RFFN1, self).__init__()
		self.sigma = np.ones(in_dim) * sigma #
		self.coeff = np.random.normal(0.0, 2, (nfeat, in_dim))
		self.coeff = self.coeff / self.sigma.reshape(1, len(self.sigma))
		self.offset = 2.0 * np.pi * np.random.rand(1, nfeat)

		self.network = nn.Sequential(
			LinearClamped1(in_dim, nfeat, self.coeff, self.offset),
			Sin(),
			nn.Linear(nfeat, out_dim, bias=False),
			# nn.ELU(),
			# nn.Linear(out_dim, out_dim, bias=False),
		)

	def forward(self, x):
		return self.network(x)


class FCNN(nn.Module):
	'''
	2-layer fully connected neural network
	'''

	def __init__(self, in_dim, out_dim, hidden_dim, act='tanh'):
		super(FCNN, self).__init__()
		activations = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh, 'leaky_relu': nn.LeakyReLU,
					   'elu': nn.ELU, 'prelu': nn.PReLU, 'softplus': nn.Softplus,'rrelu':nn.RReLU}

		act_func = activations[act]
		self.network = nn.Sequential(
			nn.Linear(in_dim, hidden_dim,bias=False), act_func(),
			nn.Linear(hidden_dim, hidden_dim,bias=False),act_func(),
			nn.Linear(hidden_dim, out_dim,bias=False),
		)

	def forward(self, x):
		return self.network(x)


class LinearClamped(nn.Module):
	'''
	Linear layer with user-specified parameters (not to be learrned!)
	'''

	__constants__ = ['bias', 'in_features', 'out_features']

	def __init__(self, in_features, out_features, weights, bias_values, bias=True):
		super(LinearClamped, self).__init__()
		self.in_features = in_features
		self.out_features = out_features

		self.register_buffer('weight', torch.Tensor(weights))
		if bias:
			self.register_buffer('bias', torch.Tensor(bias_values))

	def forward(self, input):
		if input.dim() == 1:
			return F.linear(input.view(1, -1), self.weight,self.bias)
		return F.linear(input, self.weight,self.bias)

	def extra_repr(self):
		return 'in_features={}, out_features={}, bias={}'.format(
			self.in_features, self.out_features, self.bias is not None
		)

class LinearClamped1(nn.Module):
	'''
	Linear layer with user-specified parameters (not to be learrned!)
	'''

	__constants__ = ['bias', 'in_features', 'out_features']

	def __init__(self, in_features, out_features, weights, bias_values, bias=False):
		super(LinearClamped1, self).__init__()
		self.in_features = in_features
		self.out_features = out_features

		self.register_buffer('weight', torch.Tensor(weights))
		if bias:
			self.register_buffer('bias', torch.Tensor(bias_values))

	def forward(self, input):
		if input.dim() == 1:
			return F.linear(input.view(1, -1), self.weight)
		return F.linear(input, self.weight)

	def extra_repr(self):
		return 'in_features={}, out_features={}, bias={}'.format(
			self.in_features, self.out_features, self.bias is not None
		)

class Cos(nn.Module):
	"""
	Applies the cosine element-wise function
	"""

	def forward(self, inputs):
		return torch.cos(inputs)


class Sin(nn.Module):
	"""
	Applies the cosine element-wise function
	"""

	def forward(self, inputs):
		return inputs * torch.sigmoid(inputs)




