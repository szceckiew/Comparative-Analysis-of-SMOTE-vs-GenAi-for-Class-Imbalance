import seaborn as sns
import matplotlib.pyplot as plt

def plot_class_distribution(y, title):
    sns.countplot(x=y)
    plt.title(title)
    plt.show()
