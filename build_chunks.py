import os
import pickle
import numpy as np
import itertools
import actr


root_path = '/Users/paulsomers/COGLE/saved_gym_minigrid_episodes/mixedputnearlockedmultiroom_n4s6/'
episode_folders = (next(os.walk(root_path))[1])





def build_chunk(isa,symbolic_obs,locked_flag,fc=-1):

    #for ep in symbolic_obs:
    ep = symbolic_obs
    print("EP>>>>>>>>>", type(ep))
    visibles = {'Wall': False, 'Key': False, 'LockedDoor': False, 'Goal': False, 'Door': False}
    for x, y in list(itertools.product(range(0, ep.shape[0]), range(0, ep.shape[1]))):
        if ep[x, y] is not None:
            if (type(ep[x, y])).__name__ in visibles:
                visibles[type(ep[x, y]).__name__] = True

    chunk = ['isa', isa]

    for key in visibles.keys():
        chunk.append(key)
        chunk.append([key,int(visibles[key])])
    if type(fc) != int:
        chunk.append('fc')
        chunk.append(['fc',repr(list(fc))])

    if locked_flag:
        chunk.append('get_key')
        if visibles['Key'] and visibles['LockedDoor']:
            chunk.append(['get_key',1])
        else:
            chunk.append(['get_key',0])
        chunk.append('open_door')
        if visibles['LockedDoor'] and not visibles['Key']:
            chunk.append(['open_door',1])
        else:
            chunk.append(['open_door',0])
        chunk.append('goto_goal')
        if not visibles['LockedDoor'] and not visibles['Key']:
            chunk.append(['goto_goal',1])
        else:
            chunk.append(['goto_goal',0])
    else:
        chunk.append('get_key')
        chunk.append(['get_key',0])
        chunk.append('open_door')
        if visibles['Door']:
            chunk.append(['open_door',1])
        else:
            chunk.append(['open_door',0])
        chunk.append('goto_goal')
        if not visibles['Door']:
            chunk.append(['goto_goal',1])
        else:
            chunk.append(['goto_goal',0])
    return chunk




def build_chunks_from_data():
    chunks = []
    for episode in episode_folders:
        filepath = os.path.join(root_path,episode,"inputs.npy")
        inputs = np.load(filepath)

        #labels from Robert III
        filepath = os.path.join(root_path,episode,"labels.npy")
        labels = np.load(filepath)

        #activations
        filepath = os.path.join(root_path,episode,"final_activations.npy")
        activations = np.load(filepath)

        #targets
        filepath = os.path.join(root_path,episode,"targets.npy")
        targets = np.load(filepath)

        #switches
        filepath = os.path.join(root_path, episode, "switches.npy")
        switches = np.load(filepath)

        #symbolic obs
        filepath = os.path.join(root_path, episode, "symbolic_obs.pkl")
        symbolic_obs = pickle.load(open(filepath,'rb'))
        #rebuilt the symbolic obs
        s_obs = []

        filepath = os.path.join(root_path, episode, "lockeddoor.pkl")
        locked_flag = pickle.load(open(filepath, 'rb'))
        #visibles = {'Wall':False,'Key':False,'LockedDoor':False,'Goal':False}#,'Box':False}

        # for ep in symbolic_obs:
        #     new_ob = np.empty(ep.shape)
        #     for x, y in list(itertools.product(range(0, ep.shape[0]), range(0, ep.shape[1]))):
        #         if ep[x,y] is not None:
        #             if (type(ep[x,y])).__name__ in visibles:
        #                 visibles[type(ep[x,y]).__name__] = True

        #get the step(index) of goal changes
        previous_lab = 2
        steps = [0]
        for i in range(len(labels)):
            if labels[i] != previous_lab:
                steps.append(i)
                previous_lab = labels[i]
        labels2 = [labels[i] for i in range(len(labels)) if i in steps]
        symbolic_obs2 = [symbolic_obs[i] for i in range(len(symbolic_obs)) if i in steps]
        activations = [activations[i] for i in range(len(activations)) if i in steps]
        for s,a in zip(symbolic_obs2,activations):
            chunks.append(build_chunk('decision',s,locked_flag,a))
    return chunks


#Sample code
# chunks = build_chunks_from_data()
# pickle.dump(chunks, open('chunks.p','wb'))
#print("done")