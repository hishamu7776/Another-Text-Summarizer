import matplotlib.pyplot as plt
import seaborn as sns


class Plots:
    @staticmethod
    def plot_density(matrix=None,x_tick=None,title=None):
        plt.title(title)
        plt.ylabel("sentences")
        plt.xlabel("words")
        sns.heatmap(data = matrix ,xticklabels=x_tick, cbar=False) #annot=True
        plt.savefig('./plots/'+title+'.png')
    
    @staticmethod
    def bar_plot(x_val = None, y_val = None, title = None, xlabel=None, ylabel=None, xtick=None,ytick=None):
        plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        sns.barplot(x=x_val,y=y_val)
        plt.show()
    @staticmethod
    def plot_frequency():
        return
        
    

        
