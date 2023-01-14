import numpy as np
import matplotlib.pyplot as plt
import random
import re
import os


def read_file(file_path):
    file = open(file_path, "r", encoding="UTF-8")
    lines = file.readlines()
    return lines


def find_str(string):
    b = re.findall(r"\d+\.?\d*", string)
    return b


def split_lines(lines):
    trainging_line = []
    evaling_line = []
    for line in lines:
        if line[0] == '2' and 'loss' in line:
            trainging_line.append(line)
        if 'Acc' in line:
            evaling_line.append(line)
    return trainging_line, evaling_line


def create_training_loss_fuc(training_lines):
    trainging_loss = []
    for i in training_lines:
        trainging_loss.append(float(find_str(i)[-3]))
    return trainging_loss


def create_eval_loss_fuc(eval_lines):
    eval_loss = []
    for i in eval_lines:
        eval_loss.append(float(find_str(i)[-1]))
    return eval_loss


def create_acc_function(eval_lines):
    acc = []
    for i in eval_lines:
        acc.append(float(find_str(i)[0]))
    return acc


def get_all(file_path):
    lines = read_file(file_path)
    tr, eval = split_lines(lines)
    tr_loss = create_training_loss_fuc(tr)
    ev_loss = create_eval_loss_fuc(eval)
    acc = create_acc_function(eval)
    return tr_loss, ev_loss, acc


def draw_train_loss(train_loss_dict):
    for model_name in train_loss_dict:
        x = np.linspace(1, len(train_loss_dict[model_name]), 1)
        y = train_loss_dict[model_name]
        plt.plot(x, y, label=model_name)
    plt.grid()
    plt.show()


def draw_eval_loss(eval_loss_dict):
    for model_name in eval_loss_dict:
        x = np.linspace(1, len(eval_loss_dict[model_name]), 1)
        y = eval_loss_dict[model_name]
        plt.plot(x, y, label=model_name)
    plt.grid()
    plt.show()


def color_list():
    def hex2rgb(h):
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

    return [hex2rgb(h) for h in plt.rcParams['axes.prop_cycle'].by_key()['color']]


def randomcolor():
    colorArr = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0, 14)]
    return "#" + color


def draw_best_acc(acc_dict,path=None):
    plt.figure(figsize=(15, 4), dpi=150)
    plt.rcParams['figure.figsize'] = (15.0, 4.0)
    model_name_list = [i for i in acc_dict]
    model_name_list.sort()
    num_list = [max(acc_dict[i]) for i in model_name_list]
    plt.bar(range(len(num_list)), num_list, color=[randomcolor() for i in num_list], tick_label=model_name_list)
    plt.grid()
    plt.ylabel('best acc of each model(%)')
    plt.xlabel('model name')
    if path:
        plt.savefig(path)
    plt.show()

def draw_stable_acc(acc_dict,path=None):
    plt.figure(figsize=(15, 4), dpi=150)
    plt.rcParams['figure.figsize'] = (15.0, 4.0)
    model_name_list = [i for i in acc_dict]
    model_name_list.sort()
    num_list = [acc_dict[i][-5:].mean() for i in model_name_list]
    plt.bar(range(len(num_list)), num_list, color=[randomcolor() for i in num_list], tick_label=model_name_list)
    plt.grid()
    plt.ylabel('last 5 epoch accs mean value of each model(%)')
    plt.xlabel('model name')
    if path:
        plt.savefig(path)
    plt.show()


def draw_eval_loss(loss_dict,path=None):
    plt.figure(figsize=(15, 4), dpi=150)
    model_name_list = [i for i in loss_dict]
    model_name_list.sort()
    for index, model_name in enumerate(model_name_list):
        x = np.arange(1, len(loss_dict[model_name]) + 1)
        plt.plot(x, loss_dict[model_name], c=randomcolor(), label=model_name)
    plt.ylabel('eval loss')
    plt.xlabel('epochs')
    plt.legend()
    if path:
        plt.savefig(path)
    plt.show()


def draw_train_loss(loss_dict,path=None):
    plt.figure(figsize=(15, 4), dpi=150)
    model_name_list = [i for i in loss_dict]
    model_name_list.sort()
    for index, model_name in enumerate(model_name_list):
        x = np.arange(1, len(loss_dict[model_name]) + 1)
        plt.plot(x, loss_dict[model_name], c=randomcolor(), label=model_name)
    plt.ylabel('train loss')
    plt.xlabel('iters')
    plt.legend()
    if path:
        plt.savefig(path)
    plt.show()


if __name__ == "__main__":
    file_list = os.listdir('log_files')

    train_loss_dict = {}
    eval_loss_dict = {}
    acc_dict = {}

    for file in file_list:
        real_path = os.path.join('log_files', file)
        train_loss, eval_loss, acc = get_all(real_path)
        model_name = file[:-4]
        train_loss_dict[model_name] = np.array(train_loss)
        eval_loss_dict[model_name] = np.array(eval_loss)
        acc_dict[model_name] = np.array(acc)

    draw_eval_loss(eval_loss_dict,'/Users/yueyanli/Desktop/eval_loss.png')
    draw_train_loss(train_loss_dict,'/Users/yueyanli/Desktop/training_loss.png')
    draw_best_acc(acc_dict,'/Users/yueyanli/Desktop/best_acc.png')
    draw_stable_acc(acc_dict, '/Users/yueyanli/Desktop/stable_acc.png')
