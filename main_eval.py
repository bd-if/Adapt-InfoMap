
from utlis.evaluate import evaluate


if __name__ == '__main__':

    metrics = ['pairwise', 'bcubed', 'nmi', 'new_metrics']

    evaluate(metrics)