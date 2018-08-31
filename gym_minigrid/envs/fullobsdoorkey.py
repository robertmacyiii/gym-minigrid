from copy import deepcopy 
import numpy as np
from gym_minigrid.minigrid import *
from gym_minigrid.register import register

class FullObsDoorKeyEnv(MiniGridEnv):
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
        #import pdb;pdb.set_trace()
        #np.random.seed(1234)
        splitIdx = 3#self._rand_int(2, width-2)
        self.grid.vert_wall(splitIdx, 0)

        # Place the agent at a random position and orientation
        # on the left side of the splitting wall
        self.start_pos = self.place_agent(size=(splitIdx, height))

        # Place a door in the wall
        doorIdx = 4#self._rand_int(1, width-2)
        self.grid.set(splitIdx, doorIdx, Door('yellow'))

        # Place a yellow key on the left side
        # self.place_obj(
        #     obj=Box('yellow'),
        #     top=(0, 0),
        #     size=(splitIdx, height)
        # )

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
        #import pdb;
        #pdb.set_trace()

        if self.lockeddoor == 3:
            if not self.opened_the_door and (not done and reward > 0.0):
                # if action != self.actions.toggle:
                info['get_key'] = True
            u, v = self.dir_vec
            ox, oy = (self.agent_pos[0] + u, self.agent_pos[1] + v)
            if action == self.actions.toggle and self.grid.get(ox, oy) is not None:
                if ((self.grid.get(ox, oy).type == 'locked_door') or (self.grid.get(ox, oy).type == 'door')):
                    info['unlock_door'] = True
                    self.opened_the_door = True
            if done and reward > 0.0:
                info['episode_completed'] = True

            info['symbolic_obs'] = deepcopy(np.array(grid.grid).reshape(grid.height, grid.width)).transpose()

            return obs, reward, done, info

        else:

            if preCarrying is None and self.carrying is not None:
                if self.carrying.type == 'key' or self.carrying.type == 'box':
                    info['get_key'] = True
            u, v = self.dir_vec
            ox, oy = (self.agent_pos[0] + u, self.agent_pos[1] + v)
            if action == self.actions.toggle and preCarrying and self.grid.get(ox, oy) is not None:
                if (preCarrying.type == 'key' or preCarrying.type == 'box') and ((self.grid.get(ox, oy).type == 'locked_door') or (self.grid.get(ox, oy).type == 'door')):
                    info['unlock_door'] = True
            if done and reward > 0.0:
                info['episode_completed'] = True

            info['symbolic_obs'] = deepcopy(np.array(grid.grid).reshape(grid.height, grid.width)).transpose()

            return obs, reward, done, info
 

class FullObsDoorKeyEnv5x5(FullObsDoorKeyEnv):
    def __init__(self):
        super().__init__(size=5)

class FullObsDoorKeyEnv6x6(FullObsDoorKeyEnv):
    def __init__(self):
        super().__init__(size=6)

class FullObsDoorKeyEnv7x7(FullObsDoorKeyEnv):
    def __init__(self):
        super().__init__(size=7)

class FullObsDoorKeyEnv16x16(FullObsDoorKeyEnv):
    def __init__(self):
        super().__init__(size=16)

register(
    id='MiniGrid-FullObsDoorKey-5x5-v0',
    entry_point='gym_minigrid.envs:FullObsDoorKeyEnv5x5'
)

register(
    id='MiniGrid-FullObsDoorKey-6x6-v0',
    entry_point='gym_minigrid.envs:FullObsDoorKeyEnv6x6'
)

register(
    id='MiniGrid-FullObsDoorKey-7x7-v0',
    entry_point='gym_minigrid.envs:FullObsDoorKeyEnv7x7'
)

register(
    id='MiniGrid-FullObsDoorKey-8x8-v0',
    entry_point='gym_minigrid.envs:FullObsDoorKeyEnv'
)

register(
    id='MiniGrid-FullObsDoorKey-16x16-v0',
    entry_point='gym_minigrid.envs:FullObsDoorKeyEnv16x16'
)
