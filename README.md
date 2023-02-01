# kth-dd2421
1. Decision tree
ASSIGNMENT 1: 

1.0 0.957117428264771 0.9998061328047111

ASSIGNMENT 3: 

The information gain matrix: 

 [[0.07527256 0.00583843 0.00470757 0.0263117  0.28703075 0.00075786]
 [0.00375618 0.0024585  0.00105615 0.01566425 0.01727718 0.00624762]
 [0.00712087 0.29373617 0.00083111 0.00289182 0.25591172 0.00707703]] 

For dataset monk1, node 0 in Layer 0 can be splitted by attribute A5.
For dataset monk2, node 0 in Layer 0 can be splitted by attribute A5.
For dataset monk3, node 0 in Layer 0 can be splitted by attribute A2.

Node 0 in Layer 1 has no leaf node. Its majority class is  True
Node 1 in Layer 1 can be splitted by attribute A4. Its majority class is False
Node 2 in Layer 1 can be splitted by attribute A6. Its majority class is False
Node 3 in Layer 1 can be splitted by attribute A1. Its majority class is False

ASSIGNMENT 5:

results of the errors on the 3 datasets: 

Monk-1, E_train:  0.0
Monk-1, E_test:  0.17129629629629628
Monk-2, E_train:  0.0
Monk-2, E_test:  0.30787037037037035
Monk-3, E_train:  0.0
Monk-3, E_test:  0.05555555555555558

ASSIGNMENT 6:

Explain pruning from a bias variance trade-off perspective.

The bias–variance tradeoff is the property of a model that the variance of the parameter estimated across samples can be reduced by increasing the bias in the estimated parameters. Tree pruning basically means avoiding a high variance and overfitting of the model, at the cost of increasing the bias a bit. The overall error of prediction should be  in this case be reduced.

ASSIGNMENT 7: 

see the plots of dtree pruning: 
