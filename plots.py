import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns


class Plots:
    @staticmethod
    def plot_density(matrix=None,x_tick=None, y_tick= None, title=None, labels=None, annot=False):
        figure(figsize=(12, 10), dpi=100)
        plt.title(title)
        plt.ylabel(labels[0])
        plt.xlabel(labels[1])
        sns.heatmap(data = matrix ,xticklabels=x_tick, yticklabels=y_tick,cbar=False, annot=annot)
        plt.savefig('./plots/'+title+'.png')

    @staticmethod
    def plot_mantel(matrix=None,x_tick=None, y_tick= None, title=None, labels=None, annot=False):
        fig, ax = plt.subplots(figsize=(8,8))
        plt.ylabel(labels[0])
        plt.xlabel(labels[1])
        sns.heatmap(data = matrix, xticklabels=x_tick, yticklabels=y_tick, cbar=False, annot=annot, ax=ax)
        plt.yticks(rotation=0, ha='right')
        plt.xticks(rotation=0, ha='center', position=(0,1.04))
        plt.savefig('./plots/'+title+'.png')
    
    @staticmethod
    def bar_plot(x_val = None, y_val = None, title = None, xlabel=None, ylabel=None, xtick=None,ytick=None):
        plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        sns.barplot(x=x_val,y=y_val)
        plt.show()
    
    @staticmethod
    def line_plot(data = None, x_label=None, y_label=None, title=None):
        max_val = max(data.values())
        data =  {key:value/max_val for key,value in data.items()}
        plt.title(title)
        sns.lineplot(data=data)
        plt.ylabel(y_label)
        plt.xlabel(x_label)
        plt.show()
        return
    
    

        
