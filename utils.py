import os
# To manage the different model with feature_extraction of the images
import tensorflow as tf
from skimage.filters import prewitt

# to try different cluster classification
from sklearn.cluster import KMeans, AffinityPropagation
from sklearn.decomposition import PCA

# To manage the data
import numpy as np
import matplotlib.pyplot as plt
import pandas


def get_images(path, pathimagename, subject=None, allimages=None):
    if allimages is None:
        allimages = []

    with os.scandir(path) as files:
        for f in files:
            if f.name.startswith('EMU') and f.is_dir():
                get_images(f.path, pathimagename, subject=None, allimages=allimages)
            elif f.name.startswith('SZ') and f.is_dir():
                newpath = os.path.join(f.path, pathimagename)
                get_images(newpath, pathimagename, subject=True, allimages=allimages)
            elif subject is True:
                allimages.append(f.path)

    return allimages


def extract_features(file, model, func, color, edges=False):
    # load the image as a 224x224 array
    img = tf.keras.preprocessing.image.load_img(file, target_size=(224,224), color_mode=color)
    # convert from 'PIL.Image.Image' to numpy array
    img = np.array(img)
    if edges is True:
        edges = prewitt(img)
        img = edges
    # reshape the data for the model reshape(num_of_samples, dim 1, dim 2, channels) channels = color for rgb =3, forgray =1, for rgba = 4
    # if color == 'rgb':
    #     channels = 3
    # elif color == 'grayscale':
    #     channels = 1
    # elif color == 'rgba':
    #     channels = 4
    reshaped_img = img.reshape(1, 224, 224, 3)
    # prepare image for model
    # this part depend on the model choose
    # imgx = tf.keras.applications.vgg16.preprocess_input(reshaped_img)
    imgx = func.preprocess_input(reshaped_img)
    # get the feature vector
    features = model.predict(imgx)
    return features


def run_model(path, type, color, modelType=None, images=None, display=None, edges=False):
    if type == 'table':
        # Do it with the extracted value
        file = os.path.join(path, 'allresults-nostring.xlsx')
        tab = pandas.read_excel(file)
        name = tab.subname + tab.channels
        name = tab.subname + '_' + tab.channels
        testtab = tab
        testtab = testtab.drop(columns=['subname', 'channels', 'brain_area', 'region_norm', 'epilepsy_type', 'pathology', 'clinical_ez', 'resected', 'sz_free'])
        featall = testtab.to_numpy()
        pca = PCA(n_components=10, random_state=22)
        pca.fit(featall)
        x = pca.transform(featall)
        affin = AffinityPropagation(random_state=22).fit(x)
        clusterId = affin.labels_
        tab['clusterId'] = list(clusterId)
        outfile = os.path.join(path, 'allresults-cluster.xlsx')
        tab.to_excel(outfile)
    elif type == 'images':
        # try another model VGG16, VGG19 and ResNet50
        figurepath = os.path.join(path, modelType, color)
        if edges is True:
            figurepath = os.path.join(path, modelType, 'edges')
        match modelType:
            case 'VGG16':
                func = tf.keras.applications.vgg16
                model = tf.keras.applications.VGG16()
            case 'VGG19':
                func = tf.keras.applications.vgg19
                model = tf.keras.applications.VGG19()
            case 'ResNet50':
                func = tf.keras.applications.resnet50
                model = tf.keras.applications.ResNet50()
            case 'None':
                return

        model = tf.keras.Model(inputs=model.inputs, outputs=model.layers[-2].output)


        # Do it with the images
        data = {}
        for fingerprint in images:
            try:
                feat = extract_features(fingerprint, model, func, color, edges=edges)
                data[fingerprint] = feat
            except:
                continue

        filenames = np.array(list(data.keys()))
        featall = np.array(list(data.values()))

        size = featall.shape
        # Why do I reshape at 4096?
        featall = featall.reshape(-1, size[2])

        for var in ['PCA', 'Var']:
            match var:
                case 'PCA':
                    ## First case with PCA
                    # PCA is usefull to reduce the number of clusters
                    pca = PCA()
                    pca.fit(featall)
                    x = pca.transform(featall)
                case 'Var':
                    ## Second case with variance for featues selection
                    # Can try tu use the variance to select features with higher variance
                    try:
                        variance = np.var(featall, axis=0)
                        idx = np.argwhere(variance > 0.2)
                        x = featall[:, idx]
                        x = x.reshape(-1, len(idx))
                    except ValueError:
                        continue

            affin = AffinityPropagation().fit(x)
            nbclust = np.bincount(affin.labels_)
            idxC = np.argwhere(nbclust >= 10)
            idxC += 1
            idtowrite = np.isin(affin.labels_, idxC)

            # save in excel file
            df = pandas.DataFrame()
            df['filename'] = filenames
            df['clusterID'] = affin.labels_
            finalpath = os.path.join(figurepath, var)
            if not os.path.exists(finalpath):
                os.makedirs(finalpath)
            outfile = os.path.join(finalpath, 'clusters.xlsx')
            df.to_excel(outfile)

            if display is not None:
                # holds the cluster id and the images { id: [images] }
                groups = {}
                for file, cluster in zip(filenames[idtowrite], affin.labels_[idtowrite]):
                    if cluster not in groups.keys():
                        groups[cluster] = []
                        groups[cluster].append(file)
                    else:
                        groups[cluster].append(file)

                for clust in groups:
                    #plt.figure(figsize=(50, 25))
                    # gets the list of filenames for a cluster
                    try:
                        nbr = len(groups[clust])
                        col =round(nbr/10)+1
                        for index, file in enumerate(groups[clust]):
                            name = os.path.basename(file)
                            plt.subplot(10, col, index + 1)
                            img = tf.keras.preprocessing.image.load_img(file)
                            img = np.array(img)
                            plt.imshow(img)
                            plt.axis('off')
                            #plt.title(name)
                        plt.savefig(os.path.join(finalpath, 'cluster-'+str(clust)+'.jpg'))
                        plt.close()
                    except ValueError:
                        continue

    return

