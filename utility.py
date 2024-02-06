import matplotlib.pyplot as plt
import os
from sklearn.metrics import ConfusionMatrixDisplay


def plot_confusion(target, output, out_name):
    ConfusionMatrixDisplay.from_predictions(target, output, cmap='Blues')
    plt.savefig(out_name)
    plt.close()


def create_dir_for_path(path):
    if path == '':
        raise ValueError('Given path is empty.')
    start_index = 0
    if path[0] in ['/', '.'] or path[1] == ':':
        start_index = 1
    folders = path.split('/')
    for i in range(start_index+1, len(folders)+1):
        path_create = '/'.join(folders[0:i])
        if not os.path.exists(path_create):
            os.mkdir(path_create)
