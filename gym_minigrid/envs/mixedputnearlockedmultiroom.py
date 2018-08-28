from gym_minigrid.minigrid import *
from gym_minigrid.register import register
from copy import deepcopy

class Room:
    def __init__(self,
        top,
        size,
        entryDoorPos,
        exitDoorPos
    ):
        self.top = top
        self.size = size
        self.entryDoorPos = entryDoorPos
        self.exitDoorPos = exitDoorPos

class MixedPutNearLockedMultiRoomEnv(MiniGridEnv):
    """
    Environment with multiple rooms (subgoals)
    """

    def __init__(self,
        minNumRooms,
        maxNumRooms,
        maxRoomSize=10
    ):
        assert minNumRooms > 0
        assert maxNumRooms >= minNumRooms
        assert maxRoomSize >= 4

        self.minNumRooms = minNumRooms
        self.maxNumRooms = maxNumRooms
        self.maxRoomSize = maxRoomSize

        self.rooms = []

        super(MixedPutNearLockedMultiRoomEnv, self).__init__(
            grid_size=25,
            max_steps=self.maxNumRooms * 30
        )

    def _gen_grid(self, width, height):
        roomList = []

        # Choose a random number of rooms to generate
        numRooms = self._rand_int(self.minNumRooms, self.maxNumRooms+1)

        def near_obj(env, p1):
            for p2 in objPos:
                dx = p1[0] - p2[0]
                dy = p1[1] - p2[1]
                if abs(dx) <= 1 and abs(dy) <= 1:
                    return True
            return False

        while len(roomList) < numRooms:
            curRoomList = []

            entryDoorPos = (
                self._rand_int(0, width - 2),
                self._rand_int(0, width - 2)
            )

            # Recursively place the rooms
            self._placeRoom(
                numRooms,
                roomList=curRoomList,
                minSz=6, # Size includes walls and we need at least 4x4 interiod for putNear task.
                maxSz=self.maxRoomSize,
                entryDoorWall=2,
                entryDoorPos=entryDoorPos
            )

            if len(curRoomList) > len(roomList):
                roomList = curRoomList

        # Store the list of rooms in this environment
        assert len(roomList) > 0
        self.rooms = roomList

        # Create the grid
        self.grid = Grid(width, height)
        wall = Wall()

        prevDoorColor = None

        objs = []
        objPos = []
        self.objPairs = {}

        # For each room
        for idx, room in enumerate(roomList):

            topX, topY = room.top
            sizeX, sizeY = room.size

            # Draw the top and bottom walls
            for i in range(0, sizeX):
                self.grid.set(topX + i, topY, wall)
                self.grid.set(topX + i, topY + sizeY - 1, wall)

            # Draw the left and right walls
            for j in range(0, sizeY):
                self.grid.set(topX, topY + j, wall)
                self.grid.set(topX + sizeX - 1, topY + j, wall)

            # If this isn't the first room, place the entry door
            if idx > 0:
                # Pick a door color different from the previous one
                doorColors = set(COLOR_NAMES)
                if prevDoorColor:
                    doorColors.remove(prevDoorColor)
                # Note: the use of sorting here guarantees determinism,
                # This is needed because Python's set is not deterministic
                doorColor = self._rand_elem(sorted(doorColors))

               
                if np.random.random() < 0.5:
                    entryDoor = Door(doorColor)
                    self.grid.set(*room.entryDoorPos, entryDoor)
                    prevDoorColor = doorColor
    
                    prevRoom = roomList[idx-1]
                    prevRoom.exitDoorPos = room.entryDoorPos
      
                    if doorColor not in self.objPairs.keys():
                        prevTopX, prevTopY = prevRoom.top
                        prevSizeX, prevSizeY = prevRoom.size
                        topItemPos = (prevTopX + 1, prevTopY + 1)
                        sizeItemPos = (prevSizeX - 1, prevSizeY - 1)
                        objColor = doorColor
                        #ball = Ball(objColor)
                        #ball_pos = self.place_obj(
                        #    obj=ball,
                        #    reject_fn=near_obj,
                        #    top=topItemPos,
                        #    size=sizeItemPos
                        #)
                        #objs.append(('ball', ball))
                        #objPos.append(ball_pos)
                        box = Box(objColor)
                        box_pos = self.place_obj(
                            obj=box,
                            reject_fn=near_obj,
                            top=topItemPos,
                            size=sizeItemPos
                        )
                        objs.append(('box', box))
                        objPos.append(box_pos)
                        #self.objPairs[doorColor] = {'ball': ball_pos, 'box': box_pos}
                        #ball2 = Ball(objColor)
                        #ball2_pos = self.place_obj(
                        #    obj=ball2,
                        #    reject_fn=near_obj,
                        #    top=topItemPos,
                        #    size=sizeItemPos
                        #)
                        #objs.append(('ball', ball2))
                        #self.objPairs[doorColor] = {'ball1': ball1, 'ball2': ball2}
                    
                else:
                    entryDoor = LockedDoor(doorColor)
                    self.grid.set(*room.entryDoorPos, entryDoor)
                    prevDoorColor = doorColor
    
                    prevRoom = roomList[idx-1]
                    prevRoom.exitDoorPos = room.entryDoorPos
    
                    # Place a key to the current room's door somewhere in the previous room.
                    prevTopX, prevTopY = prevRoom.top
                    prevSizeX, prevSizeY = prevRoom.size
                    topKeyPos = (prevTopX + 1, prevTopY + 1)
                    sizeKeyPos = (prevSizeX - 2, prevSizeY - 2)
                    self.place_obj(
                        obj=Key(doorColor),
                        top=topKeyPos,
                        size=sizeKeyPos
                    )

        # Randomize the starting agent position and direction
        self.place_agent(roomList[0].top, roomList[0].size)

        # Place the final goal in the last room
        self.goal_pos = self.place_obj(Goal(), roomList[-1].top, roomList[-1].size)

        self.mission = 'traverse the rooms to get to the goal'

    def _placeRoom(
        self,
        numLeft,
        roomList,
        minSz,
        maxSz,
        entryDoorWall,
        entryDoorPos
    ):
        # Choose the room size randomly
        sizeX = self._rand_int(minSz, maxSz+1)
        sizeY = self._rand_int(minSz, maxSz+1)

        # The first room will be at the door position
        if len(roomList) == 0:
            topX, topY = entryDoorPos
        # Entry on the right
        elif entryDoorWall == 0:
            topX = entryDoorPos[0] - sizeX + 1
            y = entryDoorPos[1]
            topY = self._rand_int(y - sizeY + 2, y)
        # Entry wall on the south
        elif entryDoorWall == 1:
            x = entryDoorPos[0]
            topX = self._rand_int(x - sizeX + 2, x)
            topY = entryDoorPos[1] - sizeY + 1
        # Entry wall on the left
        elif entryDoorWall == 2:
            topX = entryDoorPos[0]
            y = entryDoorPos[1]
            topY = self._rand_int(y - sizeY + 2, y)
        # Entry wall on the top
        elif entryDoorWall == 3:
            x = entryDoorPos[0]
            topX = self._rand_int(x - sizeX + 2, x)
            topY = entryDoorPos[1]
        else:
            assert False, entryDoorWall

        # If the room is out of the grid, can't place a room here
        if topX < 0 or topY < 0:
            return False
        if topX + sizeX > self.grid_size or topY + sizeY >= self.grid_size:
            return False

        # If the room intersects with previous rooms, can't place it here
        for room in roomList[:-1]:
            nonOverlap = \
                topX + sizeX < room.top[0] or \
                room.top[0] + room.size[0] <= topX or \
                topY + sizeY < room.top[1] or \
                room.top[1] + room.size[1] <= topY

            if not nonOverlap:
                return False

        # Add this room to the list
        roomList.append(Room(
            (topX, topY),
            (sizeX, sizeY),
            entryDoorPos,
            None
        ))

        # If this was the last room, stop
        if numLeft == 1:
            return True

        # Try placing the next room
        for i in range(0, 8):

            # Pick which wall to place the out door on
            wallSet = set((0, 1, 2, 3))
            wallSet.remove(entryDoorWall)
            exitDoorWall = self._rand_elem(sorted(wallSet))
            nextEntryWall = (exitDoorWall + 2) % 4

            # Pick the exit door position
            # Exit on right wall
            if exitDoorWall == 0:
                exitDoorPos = (
                    topX + sizeX - 1,
                    topY + self._rand_int(1, sizeY - 1)
                )
            # Exit on south wall
            elif exitDoorWall == 1:
                exitDoorPos = (
                    topX + self._rand_int(1, sizeX - 1),
                    topY + sizeY - 1
                )
            # Exit on left wall
            elif exitDoorWall == 2:
                exitDoorPos = (
                    topX,
                    topY + self._rand_int(1, sizeY - 1)
                )
            # Exit on north wall
            elif exitDoorWall == 3:
                exitDoorPos = (
                    topX + self._rand_int(1, sizeX - 1),
                    topY
                )
            else:
                assert False

            # Recursively create the other rooms
            success = self._placeRoom(
                numLeft - 1,
                roomList=roomList,
                minSz=minSz,
                maxSz=maxSz,
                entryDoorWall=nextEntryWall,
                entryDoorPos=exitDoorPos
            )

            if success:
                break

        return True

    def step(self, action):
        preCarrying = self.carrying
        u, v = self.dir_vec
        fx, fy = (self.agent_pos[0] + u, self.agent_pos[1] + v)
        preFront = self.grid.get(fx, fy)

        obs, reward, done, info = super().step(action)
        if done and reward > 0.0:
            info['episode_completed'] = True

        if action == self.actions.pickup and self.carrying and preCarrying is None:
            if self.carrying.type == 'key':
                info['get_key'] = True
            

        u, v = self.dir_vec
        ox, oy = (self.agent_pos[0] + u, self.agent_pos[1] + v)

        if action == self.actions.toggle and preCarrying and self.grid.get(ox, oy) is not None:
            if preCarrying.type == 'key' and (self.grid.get(ox, oy).type == 'locked_door'):
                if preCarrying.color == self.grid.get(ox, oy).color:
                    reward = self._reward()
                    info['unlock_door'] = True

        if action == self.actions.toggle and self.grid.get(ox, oy) is not None:
            if self.grid.get(ox, oy).type == 'door':
                info['open_door'] = True

        ## If successfully dropping a ball near a box of the same color or vice versa.
        #if action == self.actions.drop and preCarrying:
        #    if self.grid.get(ox, oy) is preCarrying:
        #        if preCarrying.color in self.objPairs.keys():
        #            if preCarrying.type in ['ball', 'box']:
        #                if preCarrying.type == 'ball':
        #                    tx, ty = self.objPairs[preCarrying.color]['box']
        #                elif preCarrying.type == 'box':
        #                    tx, ty = self.objPairs[preCarrying.color]['ball']
        #                if abs(ox - tx) <= 1 and abs(oy - ty) <= 1:
        #                    reward = self._reward()
        #                    self.objPairs.pop(preCarrying.color)
        #return obs, reward, done, info
        #if action == self.actions.drop and preCarrying:
        #    if self.grid.get(ox, oy) is preCarrying:
        #        if preCarrying.type == 'ball':
        #            color = preCarrying.color
        #            if self.objPairs[color]['ball1'] is preCarrying:
        #                tx, ty = self.objPairs[color]['ball2']
        #            elif self.objPairs[color]['ball2'] is preCarrying:
        #                tx, ty = self.objPairs[color]['ball1']
                 
        if action == self.actions.toggle and preFront:
            if preFront.type == 'box':
                reward += self._reward()
                info['open_box'] = True

        return obs, reward, done, info
                

class MixedPutNearLockedMultiRoomEnvN2S6(MixedPutNearLockedMultiRoomEnv):
    def __init__(self):
        super().__init__(
            minNumRooms=2,
            maxNumRooms=2,
            maxRoomSize=6
        )

class MixedPutNearLockedMultiRoomEnvN4S6(MixedPutNearLockedMultiRoomEnv):
    def __init__(self):
        super().__init__(
            minNumRooms=2,
            maxNumRooms=4,
            maxRoomSize=6
        )

class MixedPutNearLockedMultiRoomEnvN6(MixedPutNearLockedMultiRoomEnv):
    def __init__(self):
        super().__init__(
            minNumRooms=6,
            maxNumRooms=6
        )

register(
    id='MiniGrid-MixedPutNearLockedMultiRoom-N2-S6-v0',
    entry_point='gym_minigrid.envs:MixedPutNearLockedMultiRoomEnvN2S6',
    reward_threshold=1000.0
)

register(
    id='MiniGrid-MixedPutNearLockedMultiRoom-N4-S6-v0',
    entry_point='gym_minigrid.envs:MixedPutNearLockedMultiRoomEnvN4S6',
    reward_threshold=1000.0
)

register(
    id='MiniGrid-MixedPutNearLockedMultiRoom-N6-v0',
    entry_point='gym_minigrid.envs:MixedPutNearLockedMultiRoomEnvN6',
    reward_threshold=1000.0
)
