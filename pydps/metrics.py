"""
Modules for special metrics for the line process
"""

import numpy as np
from scipy.spatial.distance import cdist 
import torch



def ball_precision(y_true, y_pred,radius):
    """Compute the precision withing a ball of distance r. 
    i.e two positions are concidered correct if they lie within a ball or 
    radius r

    :y_true: line process of the ground truth solution  ( elements are 0 or 1)
    :y_pred: line process of the computed solution 
    : radius: radius of the ball
    :returns: Precision
    """

    #numpy array of the true line process
    gt_positions = torch.nonzero(y_true).numpy()
    pred_positions = torch.nonzero(y_pred).numpy()


    # minimal distance distances
    dist = cdist(pred_positions, gt_positions).min(axis=1)

    return (np.mean(dist<radius))


def ball_recall(y_true, y_pred,radius):
    """Compute the recall withing a ball of distance r. 
    i.e two positions are concidered correct if they lie within a ball or 
    radius r

    :y_true: line process of the ground truth solution  ( elements are 0 or 1)
    :y_pred: line process of the computed solution 
    : radius: radius of the ball
    :returns: recall
    """

    #numpy array of the true line process
    gt_positions = torch.nonzero(y_true).numpy()
    pred_positions = torch.nonzero(y_pred).numpy()


    # minimal distance distances
    dist = cdist(pred_positions, gt_positions).min(axis=1)

    return (np.sum(dist<radius)/len(gt_positions))


def ball_f1score(y_true, y_pred,radius):
    """Compute the f1 score withing a ball of distance r. 
    i.e two positions are concidered correct if they lie within a ball or 
    radius r

    :y_true: line process of the ground truth solution  ( elements are 0 or 1)
    :y_pred: line process of the computed solution 
    : radius: radius of the ball
    :returns: f1-score
    """

    prec = ball_precision(y_true,y_pred,radius)
    recall = ball_recall(y_true,y_pred,radius)


    return 2*(prec*recall)/(prec+recall)




if __name__ == "__main__":
    
    yt = torch.tensor([0,0,0,0,1,0,0,0,0,1,0,0,0])
    y_com = torch.tensor([0,0,0,1,0,0,0,0,1,0,0,0])


    prec = ball_precision(yt,y_com,2)
    recall = ball_recall(yt,y_com,2)
    f1 = ball_f1score(yt,y_com,2)
    print("precision is %.2f, recall = %.2f, f1_score =%.2f"%(prec,recall,f1))
