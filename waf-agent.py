import collections
import shutil
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
from gym_ids.envs.idsEnv  import samples_test,samples_train
from gym_ids.envs.features import Features
from gym_ids.envs.ids import Darknet_Check
from gym_ids.envs.flow_manipulator import  flow_Manipulator

ENV_NAME = 'IDS-v0'

nb_max_episode_steps_train = 10
nb_max_episode_steps_test = 10

ACTION_LOOKUP = {i: act for i, act in enumerate(flow_Manipulator.ACTION_TABLE.keys())}

def generate_dense_model(input_shape,layers, nb_actions):

    ######DDQN########
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dropout(0.1))

    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))

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

    memory = SequentialMemory(limit=20000, ignore_episode_boundaries=False, window_length=window_length)

    agent = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=16,
                     enable_double_dqn = True,target_model_update=1e-2, policy=policy, batch_size=16)
    agent.compile(RMSprop(lr=1e-3), metrics=['mae'])

    output = sys.stdout
    outputfile = open(r"tmp_add_more_experiment/loss_attack.txt", 'w')
    sys.stdout = outputfile

    history = agent.fit(env, nb_steps=rounds, nb_max_episode_steps=nb_max_episode_steps_train,visualize=False, verbose=2)

    model.save(r"tmp_add_more_experiment/agent.pkl")

    # model = load_model(r"tmp/agent.pkl")

    print  ("##Test##")

    features_extra = Features()
    ids_checker = Darknet_Check()

    flow_manipulator = flow_Manipulator()

    success=0
    total=0
    shp = (1,) + tuple(model.input_shape[1:])

    success_sample,evaded_samples,origin_samples = [],[],[]
    for sample in tqdm(samples_test):
        success_action = []
        if not ids_checker.check_darknet(sample):       #encounter misclassification
            continue
        total+=1

        ########## Copy file to target directory #########
        name_pre = sample[sample.rfind('\\') + 1:]
        shutil.copy(sample,r"D:\Download\DRL_attack_defense\malicious_add_more_experiment\malicious_test_copy")
        sample = r"D:\Download\DRL_attack_defense\malicious_add_more_experiment\malicious_test_copy\{}".format(name_pre)

        ########## Extract the original feature vector first time#########
        plain_sample =  str(features_extra.feature_extraction(sample)).strip("[]")

        for _ in range(nb_max_episode_steps_test):
            try:
                if not ids_checker.check_darknet(sample) :
                    success+=1
                    success_sample += success_action       #success, save success_action
                    evaded_samples.append(str(features_extra.feature_extraction(sample)).strip("[]"))        #evaded_samples, save feature vectors
                    origin_samples.append(plain_sample)         # save original feature vectors of evaded files
                    break
                f = features_extra.feature_extraction(sample).reshape(shp)
                act_values = model.predict(f)

                # exp_values = np.exp(np.clip(act_values[0] / 1., -500., 500.))   # BoltzmannQPolicy
                # probs = exp_values / np.sum(exp_values)
                # action = np.random.choice(range(nb_actions), p=probs)

                action=np.argmax(act_values[0])           #GreedyQPolicy

                success_action.append(action)                           #save actions
                sample=flow_manipulator.modify(sample,ACTION_LOOKUP[action])

            except IndexError:
                break

            except ValueError:
                break

    print  ("Sum:{} Success:{}".format(total,success))

    data_count = collections.Counter(success_sample)
    sum_actions = sum(data_count.values())
    print(sum_actions)

    if len(evaded_samples) != 0:
        dict_data = {'features': evaded_samples,'type': [2] * len(evaded_samples)}  # Evaded, label is 2
        dict_data = pd.DataFrame(dict_data, columns=["features", "type"])
        dict_data.to_csv(r"data_for_defense_add_more_experiment/evaded_samples_with_origin/evaded.csv", index=False)

        dict_data = {'features': origin_samples,'type': [3] * len(origin_samples)}  # Original, label is 3
        dict_data = pd.DataFrame(dict_data, columns=["features", "type"])
        dict_data.to_csv(r"data_for_defense_add_more_experiment/evaded_samples_with_origin/origin.csv", index=False)

    dict_data = {'rotate': [data_count[0] / sum_actions],'add_annotation':[data_count[1] / sum_actions],'addblankpage':[data_count[2] / sum_actions],
        "addwatermark":[data_count[3] / sum_actions],"concatebenign":[data_count[4] / sum_actions],'success_rate':[success / total],'average_number_of_attempts':[len(success_sample) / success]}
    dict_data = pd.DataFrame(dict_data,columns=["rotate","add_annotation","addblankpage","addwatermark","concatebenign","success_rate","average_number_of_attempts"])
    dict_data.to_csv(r"tmp_add_more_experiment/action_result/result.csv", index=False)

    outputfile.close()
    return agent, model

if __name__ == '__main__':
    train_dqn_model([5, 2],rounds=20000)
