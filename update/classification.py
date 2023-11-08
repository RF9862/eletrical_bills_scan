import numpy as np 
from PIL import Image
from tensorflow.keras.preprocessing import image

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from keras.applications.vgg16 import VGG16
from sklearn.metrics.pairwise import cosine_similarity
import os

vgg16 = VGG16(weights='weights/.keras/vgg16_weights.h5', include_top=False, 
              pooling='max', input_shape=(224, 224, 3))

# print the summary of the model's architecture.
vgg16.summary()

for model_layer in vgg16.layers:
    model_layer.trainable = False

def load_image(image_path):
    """
        -----------------------------------------------------
        Process the image provided
        - Resize the image 
        -----------------------------------------------------
        return resized image
    """
    
    input_image = Image.open(image_path)
    w, h = input_image.size
    input_image = input_image.crop((0, 0, w, int(h/2)))
    resized_image = input_image.resize((224, 224))

    return resized_image

def get_image_embeddings(object_image : image):
    
    """
      -----------------------------------------------------
      convert image into 3d array and add additional dimension for model input
      -----------------------------------------------------
      return embeddings of the given image
    """

    image_array = np.expand_dims(image.img_to_array(object_image), axis = 0)
    image_embedding = vgg16.predict(image_array)

    return image_embedding

def get_similarity_score(first_image : str, second_image : str):
    """
        -----------------------------------------------------
        Takes image array and computes its embedding using VGG16 model.
        -----------------------------------------------------
        return embedding of the image
        
    """

    first_image = load_image(first_image)
    second_image = load_image(second_image)

    first_image_vector = get_image_embeddings(first_image)
    second_image_vector = get_image_embeddings(second_image)
    
    similarity_score = cosine_similarity(first_image_vector, second_image_vector).reshape(1,)

    return similarity_score

def show_image(image_path):
    image = mpimg.imread(image_path)
    imgplot = plt.imshow(image)
    plt.show()   


def classificationFromImg(img_path):
    adani_templateImg = 'help/adaniTemplate.jpg'
    tata_templateImg = 'help/tataTemplate.jpg'
    tata_templateImg = 'help/tataTemplate.jpg'
    adaniCheck = False
    similarity_score_0 = get_similarity_score(adani_templateImg, img_path)
    score_1, score_2 = 0.85, 0.77
    if similarity_score_0 > score_1: 
        return 0
    similarity_score_1 = get_similarity_score(tata_templateImg, img_path)
    if similarity_score_1 > score_1: 
        return 1
    if max(similarity_score_1, similarity_score_0) >score_2:
        if similarity_score_0 > similarity_score_1: return 0
        else: return 1
    return 2

# classification in lots of images
def classificationFromFolder(folderName):
    files = os.listdir(folderName)
    output, score = [], []
    for f in files:
            
        # use the show_image function to plot the images
        # show_image(templateImg)

        similarity_score = get_similarity_score(templateImg, f"inputs/{f}")
        
        print(similarity_score)
        if similarity_score > 0.79: 
            output.append(f)
            score.append(similarity_score)

    # with open('out.txt','w') as f:
    #     for out in output:
    #         f.write(out)
    #         f.write('\n')
    # import shutil       
    # for im in output: 
    #     shutil.move(f"inputs/{im}", os.path.join("adani", im))