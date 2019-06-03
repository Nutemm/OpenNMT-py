import numpy as np
import sys


def compute_distance_review(trues, preds):
    #Compute the distance of ratings considering that there are 5 different ratings
    return sum(abs(trues-preds))/(4*len(trues))

if __name__ == "__main__":

    if len(sys.argv) <= 2:
        raise Exception("Need to give the file of the true results in first, and of the prediction in second") 

    else:
        file_true = sys.argv[1]
        file_pred = sys.argv[2]

        with open(file_true) as f:
            trues = np.array([int(line) for line in f])

        with open(file_pred) as f:
            preds = np.array([int(line) for line in f])

        assert len(trues) == len(preds)

        accuracy = sum(trues == preds)/len(trues)

        print("The accuracy is:", accuracy)
        print("Error rate is", 1-accuracy)

        print("")
        review_dist = compute_distance_review(trues, preds)
        print("The review score is", 1-review_dist)
        print("The review distance is", review_dist)



