import csv
import boto3
import os
import io
import cv2
import botocore
import sklearn
import pyspark
import pyarrow
import pandas as pd
import numpy as np
import findspark
import tensorflow.experimental.numpy as tnp
import tensorflow as tf
import matplotlib.pyplot as plt
from functools import reduce
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import lit, col, pandas_udf, PandasUDFType, udf
from pyspark.ml import Pipeline
from pyspark.ml.feature import PCA
from pyspark.ml.linalg import Vectors, DenseVector, VectorUDT
from pyspark.ml.image import ImageSchema
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import LogisticRegression
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
from tensorflow.keras.applications.vgg16 import decode_predictions, VGG16, preprocess_input
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

from tensorflow.keras.layers import Dense, Flatten
from PIL import Image, ImageOps, ImageFilter


def info(dataframe):
    """Prints dataframe parameters
    
    Args:
        dataframe (pd.Dataframe): data source
    Returns:
        Prints parameters: number of columns, number of rows, rate of missing values
    """
    print(str(len(dataframe.columns.values)) + " columns" )
    print(str(len(dataframe)) + " rows")
    print("Rate of missing values in df : " + str(dataframe.isnull().mean().mean()*100) + " %")
    print("")

    
def vgg16_base_layers_function():
    """xxx
        Args:
            -
    
        Returns:
            -
    """
    # load pre trained VGG-16 on ImageNet without fully-connected layers
    conv_base = VGG16(weights="imagenet", 
                  include_top=False, 
                  input_shape=(100, 100, 3))
    
    # 2 : features extraction
    for layer in conv_base.layers:
        layer.trainable = False

    # retrieve output layer of the network 
    top_model = conv_base.output

    # transform matrix to vector
    #top_model = Flatten(name="flatten")(top_model)

    # define model
    my_model = Model(inputs=conv_base.input, outputs=top_model)
    
    
    return my_model 



def display_scree_plot(pca, n_comp_pca):
    """ Plots PCA variance histogram
    Args:
        pca (sklearn.decomposition._pca.PCA):
        n_comp_pca (int): number of components
    Returns:
        -
    """
    print("\n(PCA) explained variance for " + str(n_comp_pca) + " components: {}\n".format(pca.explainedVariance.cumsum()[-1]))
    
    scree = pca.explainedVariance*100
    plt.figure(figsize=(7,5))
    plt.bar(np.arange(len(scree))+1, scree)
    plt.plot(np.arange(len(scree))+1, scree.cumsum(),c="red",marker='o')
    plt.xlabel("Rang de l'axe d'inertie")
    plt.ylabel("Pourcentage d'inertie")
    plt.title("Eboulis des valeurs propres")
    plt.show(block=False)
    
    return


@udf(returnType=VectorUDT())
def features_vectorizer_1(content):
    """
    """
    # option: resnet50
    model = ResNet50(include_top=False,
                 weights="imagenet",
                 pooling="avg")
    
    #
    img = Image.open(io.BytesIO(content))
    
    # convert the image pixels to a numpy array
    arr = img_to_array(img)
    
    # preprocessing
    arr = preprocess_input(arr)

    # reshape data for the model
    arr = arr.reshape((1, 
                       arr.shape[0], 
                       arr.shape[1], 
                       arr.shape[2]))
    
    # apply the model to get the image features
    X_features = model.predict(arr)
    
    #
    vec = X_features.flatten().tolist()
    
    #
    vec = DenseVector(vec)
    
    return vec


@udf(returnType=VectorUDT())
def features_vectorizer_2(content):
    """
    """
    # option: vgg16
    model = vgg16_base_layers_function()
    
    #
    img = Image.open(io.BytesIO(content))
    
    # convert the image pixels to a numpy array
    arr = img_to_array(img)
    
    # preprocessing
    arr = preprocess_input(arr)

    # reshape data for the model
    arr = arr.reshape((1, 
                       arr.shape[0], 
                       arr.shape[1], 
                       arr.shape[2]))
    
    # apply the model to get the image features
    X_features = model.predict(arr)
    
    #
    vec = X_features.flatten().tolist()
    
    #
    vec = DenseVector(vec)
    
    return vec



def dataframe_to_s3(s3_client, input_datafame, bucket_name, filepath, format_):
    """
    """
    if format_ == 'parquet':
        out_buffer = BytesIO()
        input_datafame.toPandas().to_parquet(out_buffer, index=False)

    elif format_ == 'csv':
        out_buffer = StringIO()
        input_datafame.toPandas().to_parquet(out_buffer, index=False)

    s3_client.put_object(Bucket=bucket_name, Key=filepath, Body=out_buffer.getvalue())