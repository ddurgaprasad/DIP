# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 11:33:46 2019

@author: E442282
"""

import sys
import os
import math


import sys

# it's easy to print this list of course:
#print( sys.argv)
#
#
#print(len(sys.argv))
#if len(sys.argv) != 3:
#    print('Two files are required')
#
#exit()    
#    

#Open first file and read text
srcFile=open(sys.argv[1],'r')
text_x=srcFile.read()

print(text_x)
#open second file and write
tgtFile=open(sys.argv[2],'w')
tgtFile.write(text_x)

#close both files after finishing
srcFile.close()
tgtFile.close()