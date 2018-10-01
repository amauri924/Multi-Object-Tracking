# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 10:35:46 2018

@author: Antoine
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 15:27:02 2018

@author: Antoine
"""


import os
import glob

import PIL.Image as pil
import imageio

import scipy.misc


            
def create_video(images_dir,output_dir):
    images=[]
    images_path = glob.glob(images_dir + '*.png')
    gif=os.path.join(output_dir, 'sequence' + '.gif')
    for i in images_path:
            images.append(imageio.imread(i))

    imageio.mimsave(gif, images)
    
#concatenate()    
#create_video()    
    
    
    
    
    
    
    
    
    
    
    
    