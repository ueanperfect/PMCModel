import json
import matplotlib.pyplot as plt
import random

def randomcolor():
    colorArr = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0, 14)]
    return "#" + color

def draw_flops(flops_dict, path=None):
    plt.figure(figsize=(15, 4), dpi=150)
    plt.rcParams['figure.figsize'] = (15.0, 4.0)
    model_name_list = [i for i in flops_dict]
    model_name_list.sort()
    num_list = [flops_dict[i] for i in model_name_list]
    plt.bar(range(len(num_list)), num_list, color=[randomcolor() for i in num_list], tick_label=model_name_list)
    plt.grid()
    plt.ylabel('model flops')
    plt.xlabel('model name')
    if path:
        plt.savefig(path)
    plt.show()

def draw_params(params_dict, path=None):
    plt.figure(figsize=(15, 4), dpi=150)
    plt.rcParams['figure.figsize'] = (15.0, 4.0)
    model_name_list = [i for i in params_dict]
    model_name_list.sort()
    num_list = [params_dict[i] for i in model_name_list]
    plt.bar(range(len(num_list)), num_list, color=[randomcolor() for i in num_list], tick_label=model_name_list)
    plt.grid()
    plt.ylabel('model params')
    plt.xlabel('model name')
    if path:
        plt.savefig(path)
    plt.show()

if __name__ == "__main__":

    f = open('data.json')

    # returns JSON object as
    # a dictionary
    data = json.load(f)

    draw_flops(data['flops'],'/Users/yueyanli/Desktop/flops.png')
    draw_flops(data['params'],'/Users/yueyanli/Desktop/params.png')
