# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 20:56:57 2023

@author: joshu
"""

import monkdata as m
import dtree as fun
import numpy as np
import drawtree_qt5 as draw

##### ASSIGNMENT 1 ###########################################################
# calculate entropy of the training datasets
entropy1 = fun.entropy(m.monk1)
entropy2 = fun.entropy(m.monk2)
entropy3 = fun.entropy(m.monk3)

print(entropy1, entropy2, entropy3)

##### ASSIGNMENT 3 ###########################################################
# calculate the gain associated with a certain attribute for all datasets
G = np.zeros((3,6))

for i in range (0,6):
     G[0,i] = fun.averageGain(m.monk1, m.attributes[i])
     G[1,i] = fun.averageGain(m.monk2, m.attributes[i])
     G[2,i] = fun.averageGain(m.monk3, m.attributes[i])

print(G)

##### CHAPTER 5 ##############################################################
# not sure if this is needed for the presentation
# not sure if it is correct either

# split dataset1 into two parts
monk11 = fun.select(m.monk1, m.attributes[4], 1)
monk12 = fun.select(m.monk1, m.attributes[4], 2)
monk12 = monk12 + fun.select(m.monk1, m.attributes[4], 3)
monk12 = monk12 + fun.select(m.monk1, m.attributes[4], 4)

#compute information gain on the nodes
G_monk1 = np.zeros((2,6))

for i in range (0,6):
    G_monk1[0,i] = fun.averageGain(monk11, m.attributes[i])
    G_monk1[1,i] = fun.averageGain(monk12, m.attributes[i])

print(G_monk1)

# build and draw tree with predefined function
# draw.drawTree(fun.buildTree(m.monk1, m.attributes,2))


##### ASSIGNMENT 5  ##########################################################
# build trees
tree1 = fun.buildTree(m.monk1, m.attributes)
tree2 = fun.buildTree(m.monk2, m.attributes)
tree3 = fun.buildTree(m.monk3, m.attributes)

#compute the errors on the datasets
print(fun.check(tree1, m.monk1))
print(fun.check(tree1, m.monk1test))
print(fun.check(tree2, m.monk2))
print(fun.check(tree2, m.monk2test))
print(fun.check(tree3, m.monk3))
print(fun.check(tree3, m.monk3test))












