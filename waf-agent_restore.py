import collections
import sys
import gym
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, ELU, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy,BoltzmannQPolicy
from rl.memory import SequentialMemory, EpisodeParameterMemory
from tqdm import tqdm
from gym_restore.envs.idsEnv  import samples_test,samples_train,origin_for_attack_data
from gym_restore.envs.features import Restore_Features
from gym_restore.envs.ids import Darknet_Check
from gym_restore.envs.flow_manipulator import  flow_Manipulator


ENV_NAME = 'IDS_restore-v0'

nb_max_episode_steps_train = 50
nb_max_episode_steps_test = 50

ACTION_LOOKUP = {i: act for i, act in enumerate(flow_Manipulator.ACTION_TABLE.keys())}

def generate_dense_model(input_shape,layers, nb_actions):

    ######DDQN########
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dropout(0.1))

    model.add(Dense(16))
    model.add(Activation("relu"))
    model.add(Dense(16))
    model.add(Activation("relu"))

    model.add(Dense(nb_actions))
    model.add(Activation('linear'))
    print(model.summary())

    return model


def train_dqn_model(layers,rounds):

    env = gym.make(ENV_NAME)
    env.seed(1)
    nb_actions = env.action_space.n
    window_length = 1

    print  ("nb_actions:")
    print  (nb_actions)
    print  ("env.observation_space.shape:")
    print  (env.observation_space.shape)

    model = generate_dense_model((window_length,) + env.observation_space.shape, layers, nb_actions)

    #############################DDQN#############################
    policy = BoltzmannQPolicy()

    memory = SequentialMemory(limit=100000, ignore_episode_boundaries=False, window_length=window_length)

    agent = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=16,
                     enable_double_dqn = True,target_model_update=1e-2, policy=policy, batch_size=16)
    agent.compile(RMSprop(lr=1e-3), metrics=['mae'])


    output = sys.stdout
    outputfile = open(r"tmp_add_more_experiment/loss_defense_more.txt", 'w')
    sys.stdout = outputfile

    history = agent.fit(env, nb_steps=rounds, nb_max_episode_steps=nb_max_episode_steps_train,visualize=False, verbose=2)

    model.save(r"tmp_add_more_experiment/agent_defense_more.pkl")

    print  ("#####Test######")

    features_extra = Restore_Features()
    ids_checker = Darknet_Check()

    flow_manipulator = flow_Manipulator()

    success=0
    total=0
    index_count = 0
    shp = (1,) + tuple(model.input_shape[1:])

    success_sample, restored_samples, origin_samples = [], [], []
    for sample_origin in tqdm(samples_test):
        success_action = []
        sample = [sample_origin]
        plain_sample = [origin_for_attack_data[index_count]]
        index_count += 1
        if not ids_checker.check_darknet(sample):
            continue
        total += 1
        for _ in range(nb_max_episode_steps_test):
            if not ids_checker.check_darknet(sample):
                success += 1
                success_sample += success_action
                restored_samples.append(str(sample).strip("[]").strip("'"))
                origin_samples.append(str(plain_sample).strip("[]").strip("'"))
                break
            f = features_extra.feature_extraction(sample).reshape(shp)
            act_values = model.predict(f)
            action = np.argmax(act_values[0])
            success_action.append(action)
            sample = flow_manipulator.modify(sample, ACTION_LOOKUP[action])

    print("Sum:{} Success:{}".format(total, success))

    data_count = collections.Counter(success_sample)
    sum_actions = sum(data_count.values())
    print(sum_actions)


    if len(restored_samples) != 0:
        dict_data = {'features': restored_samples, 'type': [4] * len(restored_samples)}
        dict_data = pd.DataFrame(dict_data, columns=["features", "type"])
        dict_data.to_csv(r"data_for_defense_add_more_experiment/restored_samples/restored_more.csv", index=False)

        dict_data = {'features': origin_samples, 'type': [3] * len(origin_samples)}
        dict_data = pd.DataFrame(dict_data, columns=["features", "type"])
        dict_data.to_csv(r"data_for_defense_add_more_experiment/restored_samples/origin_malicious_more.csv", index=False)

    dict_data = {'add_js': [data_count[0] / sum_actions], 'delete_js': [data_count[1] / sum_actions],"add_page": [data_count[2] / sum_actions],"delete_page": [data_count[3] / sum_actions],"add_obj": [data_count[4] / sum_actions],"delete_obj": [data_count[5] / sum_actions],
                 'success_rate': [success / total], 'average_number_of_attempts': [len(success_sample) / success]}
    dict_data = pd.DataFrame(dict_data,
                             columns=["add_js", "delete_js",  "add_page","delete_page","add_obj","delete_obj",
                                      "success_rate", "average_number_of_attempts"])
    dict_data.to_csv(r"tmp_add_more_experiment/action_result/restore_result_more.csv", index=False)

    outputfile.close()
    return agent, model

if __name__ == '__main__':
    train_dqn_model([5, 2],rounds=100000)
