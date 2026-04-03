import matplotlib.pyplot as plt

def plot_quantities(plot_save_path, plot_title, series_a, series_b, label_a, label_b, x_label, y_label):
    plt.figure(figsize = (10, 12))
    plt.plot(list(range(1, len(series_a) + 1)), series_a, label = label_a)
    plt.plot(list(range(1, len(series_b) + 1)), series_b, label = label_b)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(plot_title)
    plt.legend()
    plt.savefig(plot_save_path)
    plt.close()