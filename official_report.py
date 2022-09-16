import pandas as pd
from matplotlib import pyplot as plt

data = dict({0: 49, 1: 16, 2: 19, 3: 27, 4: 13, 5: 12, 6: 28, 7: 42, 8: 74, 9: 104, 10: 454, 11: 337, 12: 543, 13: 115, -6: 1, -5: 3, -1: 10})
print(data)

change_by,count = [],[]
for i,j in data.items():
    change_by.append(i)
    count.append(j)

plt.bar(change_by, count)
for a, b in zip(change_by, count):
    plt.text(a, b, b, ha='center', va='bottom')
plt.xticks(range(-6,14))
plt.grid(True, linestyle='--', alpha=0.2)
plt.xlabel("Variation of active engines (before - after)")
plt.ylabel("Frequency")
plt.savefig("tmp/official_report.jpeg", dpi=600,bbox_inches='tight')
plt.show()