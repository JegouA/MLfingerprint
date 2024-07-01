# Copyright 2024 by Aude Jegou.
# All rights reserved.
# This file is part of the MLfingerprint project,
# and is released under the "GPL-v3". Please see the LICENSE
# file that should have been included as part of this package.

import utils as ut
import os

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Initiate path and output path
    path = r'C:\Users\auj10\OneDrive - University of Pittsburgh\MLfingerprint\database'
    outpath = os.path.join(path, 'MLanalysis-June')
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    # get a list of images for classification
    pathimagename = 'petitEq'
    fingerprints = ut.get_images(path, pathimagename)

    # run different model of classification
    modeltype = ['VGG16', 'VGG19', 'ResNet50']
    # color = ['rgb', 'grayscale']
    for mod in modeltype[1::]:
        ut.run_model(outpath, 'images', 'rgb',  modelType=mod, images=fingerprints, display=True) #(path, type, color, modelType=None, images=None, display=None, edges=False)
        ut.run_model(outpath, 'images', 'rgb', modelType=mod, images=fingerprints, display=True, edges=True)
    #data = ut.run_model(path, 'table')
    print("Analysis is over!!!")
# See PyCharm help at https://www.jetbrains.com/help/pycharm/

