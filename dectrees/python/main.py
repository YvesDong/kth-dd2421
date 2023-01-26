# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 20:56:57 2023

@author: joshu
"""

import monkdata as m
import dtree as fun
import numpy as np
import drawtree_qt5 as draw

NumAttributes = len(m.attributes)

print(NumAttributes)
##### ASSIGNMENT 1 ###########################################################
# calculate entropy of the training datasets
entropy1 = fun.entropy(m.monk1)
entropy2 = fun.entropy(m.monk2)
entropy3 = fun.entropy(m.monk3)

print("ASSIGNMENT 1: ")
print(entropy1, entropy2, entropy3)

##### ASSIGNMENT 3 ###########################################################
# calculate the gain associated with a certain attribute for all datasets
G = np.zeros((3,NumAttributes))

for i in range (0,NumAttributes):
     G[0,i] = fun.averageGain(m.monk1, m.attributes[i])
     G[1,i] = fun.averageGain(m.monk2, m.attributes[i])
     G[2,i] = fun.averageGain(m.monk3, m.attributes[i])
splitAttri_l0 = np.argmax(G, axis=1)

print("\nASSIGNMENT 3: ")
print("The information gain matrix: \n", G)
for i in range(3):
    print("For dataset monk{}, node 0 in Layer 0 can be splitted by attribute {}.".format(i+1, m.attributes[splitAttri_l0[i]].name))

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

# # split dataset1 into 4 parts based on attribute 5
# NumValAttr5 = len(m.attributes[4].values)
# monk1_l1 = []
# for i in range(NumValAttr5):
#     monk1_l1.append(fun.select(m.monk1, m.attributes[4], m.attributes[4].values[i]))

# # compute information gain on the nodes
# G_monk_l1 = np.zeros((NumValAttr5, NumAttributes))
# for i in range(NumAttributes):
#     for j in range(NumValAttr5):
#         G_monk_l1[j,i] = fun.averageGain(monk1_l1[j], m.attributes[i])

# # attribute for splitting
# # G_monk1[0,i] == 0 means no need to split again
# print("Node 0 in Layer 1 has no leaf node. Its majority class is ", fun.mostCommon(monk1_l1[0]))

# # print(G_monk_l1)
# splitAttri_l1 = np.argmax(G_monk_l1[1:], axis=1)
# for i in range(3):
#     print("Node {} in Layer 1 can be splitted by attribute {}. Its majority class is {}".format(i+1, m.attributes[splitAttri_l1[i]].name, fun.mostCommon(monk1_l1[i+1])))

# # build and draw tree with predefined function
# draw.drawTree(fun.buildTree(m.monk1, m.attributes, 2))


##### ASSIGNMENT 5  ##########################################################
# build trees
tree1 = fun.buildTree(m.monk1, m.attributes)
tree2 = fun.buildTree(m.monk2, m.attributes)
tree3 = fun.buildTree(m.monk3, m.attributes)

# compute the errors on the datasets
print("\nASSIGNMENT 5: \nresults of the errors on the 3 datasets: ")
print("Monk-1, E_train: ", 1-fun.check(tree1, m.monk1))
print("Monk-1, E_test: ", 1-fun.check(tree1, m.monk1test))
print("Monk-2, E_train: ", 1-fun.check(tree2, m.monk2))
print("Monk-2, E_test: ", 1-fun.check(tree2, m.monk2test))
print("Monk-3, E_train: ", 1-fun.check(tree3, m.monk3))
print("Monk-3, E_test: ", 1-fun.check(tree3, m.monk3test))












