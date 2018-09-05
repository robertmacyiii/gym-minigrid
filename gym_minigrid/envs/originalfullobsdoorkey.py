from copy import deepcopy 

from gym_minigrid.minigrid import *
from gym_minigrid.register import register

class OriginalFullObsDoorKeyEnv(MiniGridEnv):
    """
    Environment with a door and key, sparse reward
    """

    def __init__(self, size=8):
        super().__init__(
            grid_size=size,
            max_steps=10*size*size
        )

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(size, size, 3),
            dtype='uint8'
        )
        self.observation_space = spaces.Dict({
            'image': self.observation_space
        })

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal in the bottom-right corner
        self.grid.set(width - 2, height - 2, Goal())

        # Create a vertical splitting wall
        splitIdx = self._rand_int(2, width-2)
        self.grid.vert_wall(splitIdx, 0)

        # Place the agent at a random position and orientation
        # on the left side of the splitting wall
        self.start_pos = self.place_agent(size=(splitIdx, height))

        # Place a door in the wall
        doorIdx = self._rand_int(1, width-2)
        self.grid.set(splitIdx, doorIdx, LockedDoor('yellow'))

        # Place a yellow key on the left side
        self.place_obj(
            obj=Key('yellow'),
            top=(0, 0),
            size=(splitIdx, height)
        )

        self.mission = "use the key to open the door and then get to the goal"

    def reset(self):
        obs = super().reset()
        grid = deepcopy(self.grid)
        grid.set(self.agent_pos[0], self.agent_pos[1], Box('red'))
        for i in range(self.agent_dir + 1):
            grid = grid.rotate_left()

        obs['image'] = grid.encode()
        return obs

    def step(self, action):
        preCarrying = self.carrying

        obs, reward, done, info = super().step(action)

#        import pdb; pdb.set_trace()
        grid = deepcopy(self.grid)
        grid.set(self.agent_pos[0], self.agent_pos[1], Box('red'))
        for i in range(self.agent_dir + 1):
            grid = grid.rotate_left()
        obs['image'] = grid.encode()
        #print('obs[\'image\'].shape: ', obs['image'].shape)
        #print('observation space: ', self.observation_space)
        #if self.old_obs_image is not None:
        #    print('obs[\'image\'] == self.old_obs_image: ', obs['image'] == self.old_obs_image)
        #self.old_obs_image = obs['image']
        #obs['agent_position'] = self.agent_pos
        #print('agent position: ', obs['agent_position'])
        #if obs['image'][:,:,2].sum() > 0:
        #    import pdb; pdb.set_trace()
        #obs['image'][self.agent_pos[0], self.agent_pos[1], 2] = 8
        if preCarrying is None and self.carrying is not None:
            if self.carrying.type == 'key':
                info['get_key'] = True
        u, v = self.dir_vec
        ox, oy = (self.agent_pos[0] + u, self.agent_pos[1] + v)
        if action == self.actions.toggle and preCarrying and self.grid.get(ox, oy) is not None:
            if preCarrying.type == 'key' and (self.grid.get(ox, oy).type == 'locked_door'):
                info['unlock_door'] = True
        if done and reward > 0.0:
            info['episode_completed'] = True
            
        info['symbolic_obs'] = deepcopy(np.array(grid.grid).reshape(grid.height, grid.width)).transpose()

        return obs, reward, done, info 
 

class OriginalFullObsDoorKeyEnv5x5(OriginalFullObsDoorKeyEnv):
    def __init__(self):
        super().__init__(size=5)

class OriginalFullObsDoorKeyEnv6x6(OriginalFullObsDoorKeyEnv):
    def __init__(self):
        super().__init__(size=6)

class OriginalFullObsDoorKeyEnv7x7(OriginalFullObsDoorKeyEnv):
    def __init__(self):
        super().__init__(size=7)

class OriginalFullObsDoorKeyEnv16x16(OriginalFullObsDoorKeyEnv):
    def __init__(self):
        super().__init__(size=16)

register(
    id='MiniGrid-OriginalFullObsDoorKey-5x5-v0',
    entry_point='gym_minigrid.envs:OriginalFullObsDoorKeyEnv5x5'
)

register(
    id='MiniGrid-OriginalFullObsDoorKey-6x6-v0',
    entry_point='gym_minigrid.envs:OriginalFullObsDoorKeyEnv6x6'
)

register(
    id='MiniGrid-OriginalFullObsDoorKey-7x7-v0',
    entry_point='gym_minigrid.envs:OriginalFullObsDoorKeyEnv7x7'
)

register(
    id='MiniGrid-OriginalFullObsDoorKey-8x8-v0',
    entry_point='gym_minigrid.envs:OriginalFullObsDoorKeyEnv'
)

register(
    id='MiniGrid-OriginalFullObsDoorKey-16x16-v0',
    entry_point='gym_minigrid.envs:OriginalFullObsDoorKeyEnv16x16'
)
