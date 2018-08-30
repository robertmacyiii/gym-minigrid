import json
import actr
import numpy as np
import random
import itertools
import pickle

not_dicts = ['FILLED-SLOTS','EMPTY-SLOTS','BLENDED-SLOTS','IGNORED-SLOTS','RESULT-CHUNK','MATCHED-CHUNK-VALUES',
             'MATCHED-VALUES-MAGNITUDES']

similarities_by_time = {}

def choose(items,chances):
    p = chances[0]
    x = random.random()
    i = 0
    while x > p :
        i = i + 1
        p = p + chances[i]
    return items[i]

def access_by_key(key, list):
    '''Assumes key,vallue pairs and returns the value'''
    if not key in list:
        raise KeyError("Key not in list")

    return list[list.index(key)+1]


def y_value(*args):
    print("y_value", args)
    return a*args[0] + b*args[1]# + c*args[2]


def new_blend_request(args):
    '''Add a new time key for similarities'''
    similarities_by_time[actr.get_time()] = []

def similarity(val1, val2):
    '''Linear tranformation, abslute difference'''
    if val1 == None:
        return None
    val1 = val1[1]
    val2 = val2[1]
    if type(val1) == str:
        if val1[0] == '[':
            val1 = eval(val1)
            val2 = eval(val2)
            va11 = np.array(val1)
            val2 = np.array(val2)
            ed = np.linalg.norm(val1-val2)
            print("ED", ed / 7 * -1)
            return ed / 7 * -1

    max_val = 1#max(map(max, zip(*feature_sets)))
    min_val = 0#min(map(min, zip(*feature_sets)))
    print("max,min,val1,val2",max_val,min_val,val1,val2)
    val1_t = (((val1 - min_val) * (0 + 1)) / (max_val - min_val)) + 0
    val2_t = (((val2 - min_val) * (0 + 1)) / (max_val - min_val)) + 0
    #print("val1_t,val2_t", val1_t, val2_t)
    #print("sim returning", abs(val1_t - val2_t) * -1)
    #print("sim returning", ((val1_t - val2_t)**2) * - 1)
    #return float(((val1_t - val2_t)**2) * - 1)
    #return abs(val1_t - val2_t) * - 1
    #return 0
    #print("sim returning", abs(val1_t - val2_t) * - 1)
    #return abs(val1_t - val2_t) * -1
    print("sim returning", ((abs(val1 - val2) * - 1)/max_val))
    return ((abs(val1 - val2) * - 1)/max_val)

    print("sim returning", abs(val1 - val2) / (max_val - min_val) * - 1)
    return abs(val1 - val2) / (max_val - min_val) * - 1


actr.add_command('similarity_function',similarity)
#actr.add_command('new_blend_request', new_blend_request)
actr.load_act_r_model("/Users/paulsomers/COGLE/robert-minigrid/gym-minigrid/key-door-box.lisp")
actr.record_history("blending-trace")



#actions
#not sure if this makes sense. The actions are cetegorical.
#0 = go for key
#1 = go for box
#2 = go for door
#no, this makes no sense because




def load_chunks():
    #some prototype chunks
    chks = pickle.load(open('/Users/paulsomers/COGLE/robert-minigrid/gym-minigrid/chunks.p','rb'))

    for x in chks:
        actr.add_dm(x)




#a probe
#chk = ['isa', 'observation', 'key', 0, 'wall', 1, 'has_key', 1, 'lockeddoor', 0, 'goal', 1]

def probe(observation):
    chunk = actr.define_chunks(observation)
    actr.schedule_simple_event_now("set-buffer-chunk",
                                ['imaginal', chunk[0]])
    actr.run(10)

    d = actr.get_history_data("blending-trace")
    d = json.loads(d)

    asdf = actr.get_history_data("blending-times")

    MP = actr.get_parameter_value(':mp')
    # #get t
    t = access_by_key('TEMPERATURE', d[0][1])
    # #the values
    # vs = [actr.chunk_slot_value(x,'value') for x in chunk_names]
    #
    factors = ['key', 'lockeddoor', 'door', 'goal', 'wall']
    # factors = ['needsFood', 'needsWater']
    result_factors = ['get_key', 'open_door', 'goto_goal']
    # result_factors = ['food','water']
    results = compute_S(d, factors)  # ,'f3'])
    for sums, result_factor in zip(results, result_factors):
        print("For", result_factor)
        for s, factor in zip(sums, factors):
            print(factor, MP / t * sum(s))

    print("actual value is", actr.chunk_slot_value('OBSERVATION0', 'ACTUAL'))

    print("done")


def compute_S(blend_trace, keys_list):
    '''For blend_trace @ time'''
    #probablities
    probs = [x[3] for x in access_by_key('MAGNITUDES', access_by_key('SLOT-DETAILS', blend_trace[0][1])[0][1])]
    #feature values in probe
    FKs = [access_by_key(key.upper(),access_by_key('RESULT-CHUNK',blend_trace[0][1]))[1] for key in keys_list]
    chunk_names = [x[0] for x in access_by_key('CHUNKS', blend_trace[0][1])]

    #Fs is all the F values (may or may not be needed for tss)
    #They are organized by chunk, same order as probs
    vjks = []
    for name in chunk_names:
        chunk_fs = []
        for key in keys_list:
            chunk_fs.append(actr.chunk_slot_value(name,key)[1])
        vjks.append(chunk_fs)

    #tss is a list of all the to_sum
    #each to_sum is Pj x dSim(Fs,vjk)/dFk
    #therefore, will depend on your similarity equation
    #in this case, we need max/min of the features because we use them to normalize
    max_val = 1#max(map(max, zip(*feature_sets)))
    min_val = 0#min(map(min, zip(*feature_sets)))
    n = max_val - min_val
    n = max_val
    #n = 1
    #this case the derivative is:
    #           Fk > vjk -> -1/n
    #           Fk = vjk -> 0
    #           Fk < vjk -> 1/n
    #compute Tss
    #there should be one for each feature
    #you subtract the sum of each according to (7)
    tss = {}
    ts2 = []
    for i in range(len(FKs)):
        if not i in tss:
            tss[i] = []
        for j in range(len(probs)):
            if FKs[i] > vjks[j][i]:
                dSim = -1/n
            elif FKs[i] == vjks[j][i]:
                dSim = 0
            else:
                dSim = 1/n
            tss[i].append(probs[j] * dSim)
        ts2.append(sum(tss[i]))

    #vios
    viosList = []
    viosList.append([actr.chunk_slot_value(x,'get_key')[1] for x in chunk_names])
    viosList.append([actr.chunk_slot_value(x, 'open_door')[1] for x in chunk_names])
    viosList.append([actr.chunk_slot_value(x, 'goto_goal')[1] for x in chunk_names])

    #compute (7)
    rturn = []
    for vios in viosList:
        results = []
        for i in range(len(FKs)):
            tmp = 0
            sub = []
            for j in range(len(probs)):
                if FKs[i] > vjks[j][i]:
                    dSim = -1/n
                elif FKs[i] == vjks[j][i]:
                    dSim = 0
                else:
                    dSim = 1/n
                #print("TMP", probs[j], "dSim", dSim, "ts2[i]", ts2[i], "vios[j]", vios[j])
                tmp = probs[j] * (dSim - ts2[i]) * vios[j]#sum(tss[i])) * vios[j]
                sub.append(tmp)
            results.append(sub)

        print("compute S complete")
        rturn.append(results)
    return rturn


   #print("compute_S complete")




#sample code
load_chunks()
# probe(['isa', 'observation', 'Key', ['Key',1], 'Wall', ['Wall',1], 'LockedDoor', ['LockedDoor',0], 'Door', ['Door',1],
#        'FC',['FC', '[-0.25531888,  0.41669422,  0.23226152,  0.63078272, -0.76010662, \
#        -0.09029893, -0.54343832, -0.64854711,  0.42531583, -0.04132686, \
#        -0.79317451, -0.67575014,  0.4574528 ,  0.33132109, -0.65403599, \
#         0.08258377,  0.95579618,  0.44455597, -0.70887607,  0.32312101, \
#         0.29047346,  0.79060382, -0.66067988, -0.96150863,  0.83770859, \
#         0.09570065, -0.74923313,  0.83078092, -0.11669892, -0.84619361, \
#        -0.20215416, -0.01367205, -0.96715194, -0.33959997,  0.86536831, \
#         0.96893632, -0.53886598,  0.99018013,  0.47000992, -0.50985444, \
#        -0.89989883,  0.2125448 , -0.10747594,  0.17840764, -0.6217283 , \
#        -0.99449921, -0.7496469 , -0.99991882,  0.3341113 , -0.25206232, \
#        -0.14160313,  0.98164427,  0.0744861 ,  0.2034508 , -0.32933751, \
#         0.21059594, -0.58526903,  0.70122516,  0.9825834 ,  0.26257905, \
#        -0.0146167 , -0.40424445, -0.404843  ,  0.99300772]'],
#        'Goal', ['Goal',1]])


