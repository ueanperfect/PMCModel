import re

file = open("work_dir/resnet50/2023-01-08 122807.312030/logger/resnet50.log", "r", encoding="UTF-8")
file = file.readlines()

trainging_line = []
evaling_line = []

for line in file:
    if line[0]=='2' and 'loss' in line:
        trainging_line.append(line)
    if 'Acc' in line:
        evaling_line.append(line)

def find_str(string):
    b = re.findall(r"\d+\.?\d*", string)
    return b


trainging_loss = []
evaluating_loss = []
acc = []

for i in trainging_line:
    trainging_loss.append(float(find_str(i)[-3]))

for j in evaling_line:
    acc.append(float(find_str(j)[0]))
    evaluating_loss.append(float(find_str(j)[-1]))