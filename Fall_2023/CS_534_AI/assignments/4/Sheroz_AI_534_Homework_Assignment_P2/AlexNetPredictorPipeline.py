
"""
Overall Program Documentation:

This script demonstrates the creation of an AlexNet-based binary classification model for
skin images (naevus and melanoma) using TensorFlow and Keras.

Data Preparation:
- The script loads images of naevus and melanoma skin lesions from the 'complete_mednode_dataset' directory.
- Randomly selects 70 images from each class,splits them into 50 for training and 20 for testing.
- Copies the selected images into 'trainset' and 'testset' directories for further processing.

Image Preprocessing (Based on AlexNet Paper - https://papers.nips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf):
- Calculates the mean activity over the training set to subtract from each pixel during preprocessing.
- Preprocesses images by resizing,cropping,and subtracting mean activity,creating NumPy arrays for training and testing data.

Model Creation:
- Defines a function 'tensorflow_alex_net_binary_class' to create an AlexNet-based binary classification model.
- Initializes weights using a zero-mean Gaussian distribution with a standard deviation of 0.01.
- Configures Stochastic Gradient Descent (SGD) optimizer with momentum and weight decay.
- Utilizes the Keras Sequential API to build the model with convolutional and fully connected layers.

Cross-Validation:
- Implements 5-fold cross-validation to evaluate the model's performance with different dropout rates.
- Trains the model on each fold and calculates the average accuracy for varying dropout rates between 0.0 and 0.9.

Model Training:
- Selects the best dropout rate based on cross-validation results and trains the final model.
- Evaluates the model on the test set and prints the test accuracy.
"""



"""
Libraries and Modules:

- sys: System-specific parameters and functions.
- pandas: Data manipulation and analysis.
- numpy: Numerical operations and array manipulation.
- os: Interaction with the operating system for file and directory handling.
- random: Generation of random samples and shuffling.
- shutil: High-level file operations.
- PIL (Pillow): Image processing library.
- tensorflow.keras.models: High-level neural networks API for building and training models.
- tensorflow.keras.layers: Pre-built layers for constructing neural network architectures.
- tensorflow.keras.optimizers: Optimizers for configuring the model training process.
- tensorflow.keras.initializers: Weight initialization methods for layers.
- tensorflow.keras.utils: Utilities for working with Keras models.
- sklearn.model_selection: Provides tools for model selection.
- matplotlib.pyplot: Plotting and visualization.
"""

import sys
import pandas as pd
import numpy as np
import os
import random
import shutil
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPool2D,Flatten,Dense,Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
%matplotlib inline


def main()->None:


    """
    Data Preparation:

    In this section,the script loads and prepares the dataset for training and testing. It performs the following steps:

    1. Counts the number of images in the 'naevus' and 'melanoma' directories.
    2. Randomly selects 70 images from each class,splitting them into 50 for training and 20 for testing.
    3. Copies the selected images into 'trainset' and 'testset' directories for further processing.
    4. Verifies the correctness of the selected images and prints the counts.
    """


    def initial_data_creation():

        print(len(os.listdir('complete_mednode_dataset/naevus/')),len(os.listdir('complete_mednode_dataset/melanoma/')))

        naevus_img_names=os.listdir('complete_mednode_dataset/naevus/')
        melanoma_img_names=os.listdir('complete_mednode_dataset/melanoma/')
        print(f'{len(naevus_img_names)=},{len(melanoma_img_names)=}')
        naevus_img_names=random.sample(population=naevus_img_names,k=70)
        assert len(set(naevus_img_names))==70
        naevus_img_names_train=random.sample(population=naevus_img_names,k=50)
        melanoma_img_names_train=random.sample(population=melanoma_img_names,k=50)
        assert len(set(naevus_img_names_train))==50
        assert len(set(melanoma_img_names_train))==50
        naevus_img_names_test=list(set(naevus_img_names).difference(set(naevus_img_names_train)))
        melanoma_img_names_test=list(set(melanoma_img_names).difference(set(melanoma_img_names_train)))
        assert len(set(naevus_img_names_test))==20
        assert len(set(melanoma_img_names_test))==20
        print(f'{len(naevus_img_names_train)=},{len(melanoma_img_names_train)=},{len(naevus_img_names_test)=},{len(melanoma_img_names_test)=}')

        for file_name in naevus_img_names_train:
            temp1=shutil.copy(src='complete_mednode_dataset/naevus/'+file_name,dst='complete_mednode_dataset/trainset/'+file_name)

        for file_name in melanoma_img_names_train:
            temp1=shutil.copy(src='complete_mednode_dataset/melanoma/'+file_name,dst='complete_mednode_dataset/trainset/'+file_name)

        for file_name in naevus_img_names_test:
            temp1=shutil.copy(src='complete_mednode_dataset/naevus/'+file_name,dst='complete_mednode_dataset/testset/'+file_name)

        for file_name in melanoma_img_names_test:
            temp1=shutil.copy(src='complete_mednode_dataset/melanoma/'+file_name,dst='complete_mednode_dataset/testset/'+file_name)

        return None


    # Don't run as this will add files in the dataset
    # initial_data_creation()



    """
    Hyperparameters:

    This section defines the hyperparameters used in the script, influencing various aspects of data processing, model architecture, and training.

    - `image_height` (int): The height of the images after preprocessing. Images are resized to this height during the preprocessing step.
    - `image_width` (int): The width of the images after preprocessing. Images are resized to this width during the preprocessing step.
    - `n_classes` (int): The number of classes in the classification task. In this script, it is set to 2 for binary classification (naevus and melanoma).
    - `EPOCHS` (int): The number of training epochs, representing the number of times the entire training dataset is passed through the neural network.
    - `BATCH_SIZE` (int): The number of samples processed in one iteration during training. It influences the speed and memory usage during training.

    These hyperparameters are crucial for configuring the model architecture and training process.
    """

    image_height=224
    image_width=224
    n_classes=2
    EPOCHS=12
    BATCH_SIZE=16




    """
    Image Preprocessing (Based on CNN Paper):

    This section calculates the mean activity over the training set and preprocesses each image for training and testing. Steps include:

    1. Calculation of mean activity over the training set.
    2. Preprocessing images by resizing,cropping,and subtracting mean activity.
    3. Conversion of images to NumPy arrays for training and testing.
   """

    naevus_img_names=os.listdir('complete_mednode_dataset/naevus/')
    melanoma_img_names=os.listdir('complete_mednode_dataset/melanoma/')
    all_train_img_names=os.listdir('complete_mednode_dataset/trainset/')
    all_test_img_names=os.listdir('complete_mednode_dataset/testset/')
    print(f'{len(naevus_img_names)=},{len(melanoma_img_names)=},{len(all_train_img_names)=},{len(all_test_img_names)=}')

    def calculate_mean_activity(train_path,train_images,img_h,img_w):
        sum_activity=np.zeros(3)
        for file_name in train_images:
            sum_activity +=np.sum(np.array(Image.open(train_path+file_name)),axis=(0,1))
        mean_activity=sum_activity / (len(train_images) * img_h * img_w)
        return mean_activity

    mean_activity=calculate_mean_activity(train_path='complete_mednode_dataset/trainset/',train_images=all_train_img_names,img_h=image_height,img_w=image_width)
    print(f'{mean_activity=}')

    def preprocess_image(image_path,img_name,m_act,img_h,img_w):
        img=Image.open(image_path+img_name)

        # Resize the image so that the shorter side is 300 pixels
        img.thumbnail((300,300))

        # Calculate the center crop box
        left=(img.width - img_w) // 2
        top=(img.height - img_h) // 2
        right=left + img_h
        bottom=top + img_h

        # Crop the center patch
        img=img.crop((left,top,right,bottom))

        # Convert the image to a NumPy array
        img_array=np.array(img)

        # Subtract the mean activity over the training set from each pixel
        img_array=img_array.astype(np.float64)
        img_array -=m_act

        # convert img_array back to uint8
        img_array=np.clip(img_array,0,255).astype(np.uint8)

        return img_array

    x_train=[]
    y_train=[]
    x_test=[]
    y_test=[]

    for file_name in all_train_img_names:
        if file_name in naevus_img_names:
            y_train.append(1)
        else:
            y_train.append(0)
        x_train.append(preprocess_image(image_path='complete_mednode_dataset/trainset/',img_name=file_name,m_act=mean_activity,img_h=image_height,img_w=image_width))

    for file_name in all_test_img_names:
        if file_name in naevus_img_names:
            y_test.append(1)
        else:
            y_test.append(0)
        x_test.append(preprocess_image(image_path='complete_mednode_dataset/testset/',img_name=file_name,m_act=mean_activity,img_h=image_height,img_w=image_width))

    x_train=np.array(x_train)
    x_test=np.array(x_test)
    y_train=np.array(y_train)
    y_test=np.array(y_test)

    print(f'{x_train.shape=},{y_train.shape=},{x_test.shape=},{y_test.shape=}')

    # shuffle training data
    shuffling_indices=np.arange(x_train.shape[0])
    np.random.shuffle(shuffling_indices)

    x_train=x_train[shuffling_indices]
    y_train=y_train[shuffling_indices]

    y_train=to_categorical(y=y_train,num_classes=n_classes)
    y_test=to_categorical(y=y_test,num_classes=n_classes)

    print(f'{x_train.shape=},{y_train.shape=},{x_test.shape=},{y_test.shape=}')









    """
    Model Creation:

    Defines a function to create an AlexNet-based binary classification model using TensorFlow and Keras. Key components include:

    1. Convolutional layers with specified parameters.
    2. Fully connected layers with dropout.
    3. Output layer with softmax activation.
    4. Stochastic Gradient Descent (SGD) optimizer with momentum and weight decay.
    """

    def tensorflow_alex_net_binary_class(image_width,image_height,n_classes,d_value,lr):
        model=Sequential()

        # Initialize weights with a zero-mean Gaussian distribution with std deviation 0.01
        weight_initializer=RandomNormal(mean=0.0,stddev=0.01,seed=1234)

        # Convolutional Layer 1
        model.add(Conv2D(input_shape=(image_width,image_height,3),filters=96,kernel_size=(11,11),strides=(4,4),padding='same',dilation_rate=(1,1),activation='relu',kernel_initializer=weight_initializer,name='conv_1'))

        # Pooling Layer 1
        model.add(MaxPool2D(pool_size=(3,3),strides=(2,2),name='max_pool_1'))

        # Convolutional Layer 2
        model.add(Conv2D(filters=256,kernel_size=(5,5),strides=(1,1),padding='same',dilation_rate=(1,1),activation='relu',kernel_initializer=weight_initializer,name='conv_2'))

        # Pooling Layer 2
        model.add(MaxPool2D(pool_size=(3,3),strides=(2,2),name='max_pool_2'))

        # Convolutional Layers 3-5
        model.add(Conv2D(filters=384,kernel_size=(3,3),strides=(1,1),padding='same',dilation_rate=(1,1),activation='relu',kernel_initializer=weight_initializer,name='conv_3'))
        model.add(Conv2D(filters=384,kernel_size=(3,3),strides=(1,1),padding='same',dilation_rate=(1,1),activation='relu',kernel_initializer=weight_initializer,name='conv_4'))
        model.add(Conv2D(filters=256,kernel_size=(3,3),strides=(1,1),padding='same',dilation_rate=(1,1),activation='relu',kernel_initializer=weight_initializer,name='conv_5'))

        # Pooling Layer 3
        model.add(MaxPool2D(pool_size=(3,3),strides=(2,2),name='max_pool_3'))

        # Flatten Layer
        model.add(Flatten(name='flatten_1'))

        # Fully Connected Layer 1
        model.add(Dense(units=4096,activation='relu',kernel_initializer=weight_initializer,name='fc1'))
        model.add(Dropout(rate=d_value))

        # Fully Connected Layer 2
        model.add(Dense(units=4096,activation='relu',kernel_initializer=weight_initializer,name='fc2'))
        model.add(Dropout(rate=d_value))

        # Output Layer
        model.add(Dense(units=n_classes,activation='softmax',kernel_initializer=weight_initializer,name='output'))

        # Compile the model with Stochastic Gradient Descent with specified parameters
        model.compile(optimizer=SGD(learning_rate=lr,momentum=0.9,weight_decay=5e-4,name='SGD'),loss='categorical_crossentropy',metrics=['accuracy'])

        return model






    """
    Cross-Validation:

    Implements 5-fold cross-validation to evaluate the model's performance with different dropout rates. Key steps include:

    1. Looping over each dropout rate in a specified range.
    2. Performing 5-fold cross-validation,training the model on each fold,and calculating average accuracy.
    3. Printing the average accuracy for each dropout rate.
    4. Printing the best accuracy for dropout rate.
    """

    def custom_dropout_simulation(dropout_rates,x_train,y_train,BATCH_SIZE):
        avg_accuracy=[]

        # Perform 5-fold cross-validation
        kf=KFold(n_splits=5,shuffle=True,random_state=1234)

        # range of dropout rates to try
        for dropout_rate in dropout_rates:
            print(f"Testing dropout rate: {dropout_rate}")
            all_accuracies=[]

            for train_index,val_index in kf.split(x_train):
                x_train_fold,x_val_fold=x_train[train_index],x_train[val_index]
                y_train_fold,y_val_fold=y_train[train_index],y_train[val_index]

                # Create the model with specified dropout rate
                alexnet_model=tensorflow_alex_net_binary_class(224,224,2,dropout_rate,0.01)

                # Train the model
                alexnet_model.fit(
                    x_train_fold,
                    y_train_fold,
                    batch_size=BATCH_SIZE,
                    epochs=5,
                    verbose=0,
                    validation_data=(x_val_fold,y_val_fold),
                )

                # Evaluate the model on the validation set
                _,current_accuracy=alexnet_model.evaluate(x_val_fold,y_val_fold)
                all_accuracies.append(current_accuracy)

            # Calculate the average accuracy for the current dropout rate
            average_accuracy=sum(all_accuracies) / len(all_accuracies)
            avg_accuracy.append(average_accuracy)
            print(f"Average accuracy for dropout rate {dropout_rate}: {average_accuracy}")
        best_acc=max(avg_accuracy)
        best_dropout=dropout_rates[avg_accuracy.index(best_acc)]
        print(f'Best Accuracy {best_acc} with dropout {best_dropout}')
        return best_dropout


    # Don't run this as it takes time based on the resources
    # # 1st Simulation
    # custom_dropout_simulation([0.1,0.3,0.5,0.7,0.9],x_train,y_train,BATCH_SIZE)
    # # 2nd Simulation
    # custom_dropout_simulation([0.5,0.55,0.6,0.65,0.7],x_train,y_train,BATCH_SIZE)
    # # 3rd Simulation
    # best_dp=custom_dropout_simulation([0.51,0.53,0,56,0.59],x_train,y_train,BATCH_SIZE)
    best_dp=0.53





    """
    Final Model Training:

    Based on the best dropout rate based on cross-validation results and trains the final model.
    The model is evaluated on the test set,and the test accuracy is printed.
    """

    # Create the model with specified dropout rate
    alexnet_model=tensorflow_alex_net_binary_class(image_width,image_height,n_classes,best_dp,0.001)

    # Train the model
    alexnet_model.fit(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=3,
        verbose=1,
        validation_split=0.05,
    )

    print('Test Accuracy with dropout 0.53-> ',alexnet_model.evaluate(x_test,y_test,batch_size=BATCH_SIZE))

    return None


if __name__ == '__main__':
    main()
    sys.exit()
