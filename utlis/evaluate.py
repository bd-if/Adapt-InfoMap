
import numpy as np
from utlis import metrics
from configs import config


def evaluate(metric):
    file = config.result_path
    label_path = config.label_path
    pred_labels = np.load(file)

    with open(label_path, 'r') as f:
        gt_labels = f.readlines()
        f.close()
    gt_labels = np.array([k.strip() for k in gt_labels], dtype=np.uint32)

    print('#cls:{}, pred:{}'.format(len(np.unique(gt_labels)), len(np.unique(pred_labels))))

    for met in metric:
        metric_func = metrics.__dict__[met]
        metric_func(gt_labels, pred_labels)