import numpy as np
import matplotlib.pyplot as plt
import time
import param
import copy

def scan2pc(scan,robot_pose):
	theta = robot_pose[2]+0.08
	angles = np.linspace(theta-np.pi*1.5/2,theta+np.pi*1.5/2,len(scan), endpoint=False)
	pc = np.zeros((len(scan),2))
	pc[:,0] = scan * np.cos(angles)+robot_pose[0]
	pc[:,1] = scan * np.sin(angles)+robot_pose[1]
	return pc
#map
class Lidar:
	def __init__(self, num_particles=61, fov = np.pi * 1.5, max_dist=10000, dyn_obs=None):
		self.num_particles =num_particles
		# [Lx, Ly, Ux, Uy]
		self.obstacles = np.array([[0., 3280., 1000., 3480.], # B9
					          	   [7080., 1000., 8080., 1200.], # B1
							  	   [1500., 0., 1700., 1000.], # B7
							  	   [6380., 3480., 6580., 4480.], # B3
							  	   [1500., 2140., 2300., 2340.], # B8
							  	   [5780., 2140., 6580., 2340.], # B2
							  	   [3540., 3345., 4540., 3545.], # B6
							  	   [3540., 935., 4540., 1135.], # B4
								   [0, 0 , 8080, 4480]]) # Wall
		if not (dyn_obs is None):
			self.obstacles = np.vstack([self.obstacles, dyn_obs])
		self.fov = fov
		self.max_dist = max_dist
		self.obstacles_segment = self.get_all_obstacles_segments(self.obstacles)
		center_obj_segment = np.array([[3863,2240,4040,2417],
									   [4040,2417,4217,2240],
									   [4217,2240,4040,2063],
									   [4040,2063,3863,2240]]) # B5
		self.obstacles_segment = np.vstack([self.obstacles_segment, center_obj_segment])

	def add_dynamic_obstacles(self, dyna_obs):
		self.obstacles = np.vstack([self.obstacles, dyna_obs])
		self.obstacles_segment = self.get_all_obstacles_segments(self.obstacles)

	def get_all_obstacles_segments(self, obs):
		"""
		obs: obstacles array holding left_bottom and top_right coordinate
		return: segments (x1,y1,x2,y2)
		
		"""
		# a:left bottom x
		left_bottom = obs[:,0:2]
		right_top = obs[:,2:4]
		left_top = np.vstack((obs[:,0],obs[:,3])).T
		right_bottom = np.vstack((obs[:,2],obs[:,1])).T
		left_vertical = np.concatenate((left_bottom,left_top),axis=1)
		right_vertical = np.concatenate((right_bottom,right_top),axis=1)
		top_horizontal = np.concatenate((left_top,right_top),axis=1)
		bottom_horizontal = np.concatenate((left_bottom,right_bottom),axis=1)
		all_segments = np.concatenate((left_vertical,right_vertical,top_horizontal,bottom_horizontal))
		return all_segments

	def get_intersection(self, a1, a2, b1, b2) :
		"""
		:param a1: (x1,y1) line segment 1 - starting position
		:param a2: (x1',y1') line segment 1 - ending position
		:param b1: (x2,y2) line segment 2 - starting position
		:param b2: (x2',y2') line segment 2 - ending position
		:return: point of intersection, if intersect; None, if do not intersect
		#adopted from https://github.com/LinguList/TreBor/blob/master/polygon.py
		"""
		def perp(a) :
			b = np.empty_like(a)
			b[0] = -a[1]
			b[1] = a[0]
			return b
		
		da = a2-a1
		db = b2-b1
		dp = a1-b1
		dap = perp(da)
		denom = np.dot( dap, db)
		num = np.dot( dap, dp )
		
		if denom == 0:
			return None
		else:
			intersct = np.array((num/denom)*db + b1) #TODO: check divide by zero!
		delta = 1e-3
		condx_a = min(a1[0], a2[0])-delta <= intersct[0] and max(a1[0], a2[0])+delta >= intersct[0] #within line segment a1_x-a2_x
		condx_b = min(b1[0], b2[0])-delta <= intersct[0] and max(b1[0], b2[0])+delta >= intersct[0] #within line segment b1_x-b2_x
		condy_a = min(a1[1], a2[1])-delta <= intersct[1] and max(a1[1], a2[1])+delta >= intersct[1] #within line segment a1_y-b1_y
		condy_b = min(b1[1], b2[1])-delta <= intersct[1] and max(b1[1], b2[1])+delta >= intersct[1] #within line segment a2_y-b2_y
		if not (condx_a and condy_a and condx_b and condy_b):
			intersct = None #line segments do not intercept i.e. interception is away from from the line segments
			
		return intersct

	def get_laser_ref(self, robot_pose=np.array([4.04, 2.0, 0])):
		"""
		:param
			robot_pose: robot's position in the global coordinate system in meter and rad
		:return: 1xn_reflections array indicating the laser end point
		"""
		xy_robot = robot_pose[:2] * 1000 #robot position from meter to mm
		theta_robot = robot_pose[2] + 0.08 #robot angle in rad
		
		angles = np.linspace(theta_robot - self.fov/2, theta_robot + self.fov/2, self.num_particles, endpoint=False)
		dist_theta = self.max_dist*np.ones(self.num_particles) # set all laser reflections to max_dist
		point_theta = np.zeros((self.num_particles, 2))
		delta_vec = np.array([0.15 * np.cos(robot_pose[2]), 0.15 * np.sin(robot_pose[2])])
		
		for seg_i in self.obstacles_segment:
			xy_i_start, xy_i_end = np.array(seg_i[:2]), np.array(seg_i[2:]) #starting and ending points of each segment
			for j, theta in enumerate(angles):
				xy_ij_max = xy_robot + np.array([self.max_dist*np.cos(theta), self.max_dist*np.sin(theta)]) + delta_vec # max possible distance
				intersection = self.get_intersection(xy_i_start, xy_i_end, xy_robot, xy_ij_max)

				if intersection is not None: #if the line segments intersect
					r = np.sqrt(np.sum((intersection-xy_robot)**2)) #radius

					if r < dist_theta[j]:
						dist_theta[j] = r
						point_theta[j] = intersection / 1000

		return dist_theta/1000, point_theta

	def paint(self, pointcloud, robot_pose):
		figure, axes = plt.subplots()
		for segment in self.obstacles_segment:
			plt.plot([segment[0]/1000, segment[2]/1000],[segment[1]/1000, segment[3]/1000], color="black")
		
		plt.plot(pointcloud[:,0], pointcloud[:,1], "bo")
		cc = plt.Circle(robot_pose[:2], 0.1, color="red")
		axes.add_artist(cc)
		axes.set_aspect(1)
		plt.xlim(0, 8.080)
		plt.ylim(0, 4.480)

class MotionModel:
	def __init__(self, init_pose, weight=0.5):
		self.init_pose = init_pose
		self.current_pose = init_pose.copy()
		self.weight = weight

	def map_velocity(self, action, theta):
		action = copy.deepcopy(action)
		vx = action[0] * np.cos(theta) - action[1] * np.sin(theta)
		vy = action[0] * np.sin(theta) + action[1] * np.cos(theta)
		action[0] = vx
		action[1] = vy
		return action

	def update(self, action, s_obs):
		action = self.map_velocity(action, self.current_pose[2])
		s_update = self.current_pose + action[:3] * param.TS
		w=np.array([self.weight, self.weight, 1])
		s = w * s_obs + (1-w) * s_update
		self.current_pose = s
		return s

class Filter:
	def __init__(self, init_obs, init_pose, dyn_obs=None, samples=2000): # dyn_obs: [5, 2]
		self.init_pos = np.array([float(init_pose[0]),float(init_pose[1])])
		self.init_theta = init_pose[2]
		self.init_obs = init_obs
		self.lidar = Lidar()
		if not (dyn_obs is None):
			dynamic_obs = np.hstack([dyn_obs-0.15, dyn_obs+0.15]) * 1000
			print(dynamic_obs)
			self.lidar.add_dynamic_obstacles(dynamic_obs)
		self.samples = samples
		#self.offset = self.debias()
		self.offset = self.debias_hierarchical()
		self.current_pose = np.array([float(init_pose[0])+float(self.offset[0]), 
									  float(init_pose[1])+float(self.offset[1]),init_pose[2]])
		self.model = MotionModel(self.current_pose)

	def diff(self, scan):
		dist = np.linalg.norm(scan-self.init_obs,ord=2)
		return dist
	
	def debias(self):
		rand_coord = np.tile(self.init_pos, self.samples).reshape(-1, 2)
		E = np.random.uniform(-0.5, 0.5, size=(self.samples, 2))
		rand_coord += E
		dist = np.inf
		idx = 0
		for i in range(self.samples):
			scan, pc = self.lidar.get_laser_ref(np.array([rand_coord[i,0],rand_coord[i,1],self.init_theta]))
			_dist = self.diff(scan)
			if _dist < dist:
				dist = _dist
				idx = i
		return rand_coord[idx] - self.init_pos

	def debias_hierarchical(self, search_depth = 3):
		current_guess = self.init_pos.copy()
		for i in range(search_depth):
			rand_coord = np.tile(current_guess, self.samples//search_depth).reshape(-1, 2)
			#print(rand_coord.shape)
			E = np.random.uniform(-0.5/(i+1), 0.5/(i+1), size=(self.samples//search_depth,2))
			rand_coord += E
			dists = np.zeros(self.samples//search_depth)
			for j in range(self.samples//search_depth):
				scan, _ = self.lidar.get_laser_ref(np.array([rand_coord[j,0], rand_coord[j,1],self.init_theta]))
				_dist = self.diff(scan)
				dists[j] = _dist
			order = np.argsort(dists)
			current_guess = np.mean(rand_coord[order][:3], axis=0) # Only select top 3 as candidate.
		return current_guess - self.init_pos
			

	def filter_obs(self, obs_vec, action):
		obs_pose = obs_vec[0][:3]
		obs_pose[0] += self.offset[0]
		obs_pose[1] += self.offset[1]
		updated_pose = self.model.update(action, obs_pose)
		return updated_pose

if __name__ == "__main__":
	lidar = Lidar(dyn_obs=np.load("dyna_obs.npy"))
	robot_pose = np.load("robot_pose.npy") 
	dist, pc = lidar.get_laser_ref(robot_pose)
	pc2 = scan2pc(np.load("laser.npy")[::-1],robot_pose)
	#pc = 
	print(pc.shape)
	lidar.paint(pc,robot_pose)
	plt.show()
	lidar.paint(pc2,robot_pose)
	plt.show()
	exit()
	dist += np.random.random(61) * 0.05 - 0.025
	init_guess = [3.9, 1.4, 0]
	filter = Filter(dist, [3.9, 1.4, 0], samples = 200)
	result = filter.debias()
	print(result)
	init_guess[0] += result[0]
	init_guess[1] += result[1]
	print(init_guess)
	dist2, pc2 = lidar.get_laser_ref(np.array(init_guess))
	lidar.paint(pc2, init_guess)
	plt.show()

	

