import re
import matplotlib.pyplot as plt
import numpy as np

# with open(r"tmp/loss_attack.txt", 'r', encoding='utf-8') as f:
#     content = f.read()
#
# result = re.findall(".*loss: (.*).*, mae", content)
# result_ep = re.findall(".*episode: (.*).*, duration", content)
# result_q = re.findall(".*mean_q: (.*).*", content)
# print(result_q)
#
# num_list_loss = list(map(lambda x: float(x) if x != "--" else float('inf'), result))
# num_list_mean_q = list(map(lambda x: float(x) if x != "--" else float('inf'), result_q))
# num_list_episode = list(map(lambda x: int(x), result_ep))
#
# plt.plot(num_list_episode, num_list_mean_q,label="Mean_q")
# plt.xlabel("Episodes")
# plt.ylabel("Mean_q")
# plt.grid(True, linestyle='--', alpha=0.2)
# plt.legend(loc="upper right")
# plt.savefig("tmp/attack_episode_mean_q.jpeg", dpi=600,bbox_inches='tight')
# plt.show()
#
# plt.plot(num_list_episode, num_list_loss,label="Loss")
# plt.xlabel("Episodes")
# plt.ylabel("Loss")
# plt.grid(True, linestyle='--', alpha=0.2)
# plt.legend(loc="upper right")
# plt.savefig("tmp/attack_episode_loss.jpeg", dpi=600,bbox_inches='tight')
# plt.show()




# import re
# import matplotlib.pyplot as plt
# import numpy as np
#
# with open(r"tmp/loss_defense.txt", 'r', encoding='utf-8') as f:
#     content = f.read()
#
# with open(r"tmp/loss_defense_more.txt", 'r', encoding='utf-8') as f:
#     content_more = f.read()
#
# result = re.findall(".*loss: (.*).*, mae", content)
# result_more = re.findall(".*loss: (.*).*, mae", content_more)
# result_ep = re.findall(".*episode: (.*).*, duration", content)
# result_ep_more = re.findall(".*episode: (.*).*, duration", content_more)
# result_q = re.findall(".*mean_q: (.*).*", content)
# result_q_more = re.findall(".*mean_q: (.*).*", content_more)
#
# num_list_loss = list(map(lambda x: float(x) if x != "--" else float('inf'), result))
# num_list_loss_more = list(map(lambda x: float(x) if x != "--" else float('inf'), result_more))
# num_list_mean_q = list(map(lambda x: float(x) if x != "--" else float('inf'), result_q))
# num_list_mean_q_more = list(map(lambda x: float(x) if x != "--" else float('inf'), result_q_more))
# num_list_episode = list(map(lambda x: int(x), result_ep))
# num_list_episode_more = list(map(lambda x: int(x), result_ep_more))
#
#
# plt.plot(num_list_episode_more, num_list_mean_q_more,label="feature_analysis")
# plt.plot(num_list_episode, num_list_mean_q,label="random")
# plt.xlabel("Episodes")
# plt.ylabel("Mean_q")
# plt.grid(True, linestyle='--', alpha=0.2)
# plt.legend(loc="upper right")
# plt.savefig("tmp/defense_mean_q.jpeg", dpi=600,bbox_inches='tight')
# plt.show()
#
# plt.plot(num_list_episode_more, num_list_loss_more,label="feature_analysis")
# plt.plot(num_list_episode, num_list_loss,label="random")
# plt.xlabel("Episodes")
# plt.ylabel("Loss")
# plt.grid(True, linestyle='--', alpha=0.2)
# plt.legend(loc="upper right")
# plt.savefig("tmp/defense_loss.jpeg", dpi=600,bbox_inches='tight')
# plt.show()


with open(r"tmp_add_more_experiment/loss_defense_more.txt", 'r', encoding='utf-8') as f:
    content = f.read()

result_steps = re.findall(".*episode steps: (.*).*, steps per second", content)
result_rewards = re.findall(".*episode reward: (.*).*, mean reward", content)
result_steps = list(map(lambda x: int(x), result_steps))
result_rewards = list(map(lambda x: float(x), result_rewards))


print(sum(result_steps) / len(result_steps))
print(sum(result_rewards) / len(result_rewards))