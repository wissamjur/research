'''
Custom script to compute precision and recall for a Surprise algorithm
'''

from helpers.metrics import precision_recall_at_k
from surprise.model_selection import KFold

def compute_prec_rec(trainset, testset, data, algo, predictions):
    avg_precision = []
    avg_recall = []
    kf = KFold(n_splits=5)
    for trainset, testset in kf.split(data):
        algo.fit(trainset)
        predictions = algo.test(testset)
        precisions, recalls = precision_recall_at_k(predictions, k=5, threshold=4)

        # Precision and recall can then be averaged over all users
        avg_precision.append(round(sum(prec for prec in precisions.values()) / len(precisions), 4))
        avg_recall.append(round(sum(rec for rec in recalls.values()) / len(recalls), 4))

    print("Precision: ", avg_precision)
    print("Recall: ", avg_recall)