import numpy as np
import matplotlib.pyplot as plt


def plot_learning_rate(save_dir='./results/parameter'):
    """Plot accuracy vs learning rate for different datasets."""
    plt.rcParams['font.family'] = 'Times New Roman'

    learning_rates = [0.01, 0.001, 0.0001, 0.00001, 0.000001]
    results = {
        'Salinas':      [77.01, 82.86, 78.84, 86.56, 63.78],
        'Botswana':     [93.60, 93.29, 93.41, 93.01, 74.26],
        'Houston 2013': [67.60, 68.31, 72.73, 61.17, 53.89],
        'Indian-Pines': [61.40, 65.30, 64.82, 55.05, 46.79],
    }
    styles = {
        'Salinas':      {'marker': 's', 'linestyle': '-.', 'color': 'g'},
        'Botswana':     {'marker': 'x', 'linestyle': '--', 'color': 'r'},
        'Houston 2013': {'marker': 'o', 'linestyle': '-',  'color': 'b'},
        'Indian-Pines': {'marker': 'd', 'linestyle': ':',  'color': 'm'},
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    for name, acc in results.items():
        plt.plot(learning_rates, acc, label=name, **styles[name])

    plt.xscale('log')
    plt.xlabel('Learning Rate', fontsize=20, fontweight='bold')
    plt.ylabel('ACC', fontsize=20, fontweight='bold')
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.legend(loc='lower right')
    plt.tick_params(labelsize=15)
    plt.tick_params(axis='both', which='both', length=0)
    plt.subplots_adjust(bottom=0.2)
    fig.savefig(f'{save_dir}/learning_rate.pdf', bbox_inches='tight', pad_inches=0)
    plt.show()


def plot_alpha(save_dir='./results/parameter'):
    """Plot accuracy vs alpha weight for different datasets."""
    plt.rcParams['font.family'] = 'Times New Roman'

    alpha_results = {
        '1e-4': [82.83, 91.78, 71.38, 61.62],
        '1e-3': [86.56, 93.29, 72.73, 65.30],
        '1e-2': [76.90, 90.27, 76.43, 65.20],
        '1e-1': [77.86, 89.87, 73.39, 58.02],
    }
    colors = ['#757e97', '#ceddec', '#6891c6', '#8abbdd']
    labels = ['Salinas', 'Botswana', 'Houston 2013', 'Indian-Pines']

    x = np.arange(len(labels))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (alpha, acc) in enumerate(alpha_results.items()):
        offset = (i - 1.5) * width
        rects = ax.bar(x + offset, acc, width, color=colors[i], label=alpha)
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}', fontsize=13,
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom')

    ax.set_xlabel(r"$\alpha$", fontsize=30, fontweight='bold')
    ax.set_ylabel("ACC", fontsize=25, fontweight='bold')
    ax.set_ylim(30, 100)
    plt.tick_params(labelsize=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(fontsize=20, loc='upper right')
    fig.tight_layout()
    fig.savefig(f'{save_dir}/alpha.pdf', bbox_inches='tight', pad_inches=0)
    plt.show()


if __name__ == '__main__':
    import os
    os.makedirs('./results/parameter', exist_ok=True)
    plot_learning_rate()
    plot_alpha()
