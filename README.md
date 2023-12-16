The CPResNet Prototype 

The research aimed to develop a prototype with high accuracy on the predictions, two different strategies were followed one by changing the architecture of the ResNet-50 and another by changing the architecture of the AlexNet. 
The study took in consideration the performance of different architectures on the detection of Covid-19 on CT scans and several related studies were analyzed.
The results in Covid-19 detection on CT scans may vary across studies and datasets. Performance metrics such as accuracy, sensitivity, specificity, and area under the curve (AUC) are commonly used to evaluate these models. The choice of architecture that showed higher results often depends on the specific dataset characteristics, size, and diversity.
Additionally, some studies have utilized transfer learning techniques by fine-tuning pre-trained models like Inception, MobileNet, or NASNet on Covid-19 CT datasets, leveraging the representations learned on large-scale image datasets like ImageNet.
Several deep learning architectures and models have been employed for Covid-19 detection on CT scans. The performance of these models can vary based on the dataset used, the preprocessing techniques, and the specific architecture's design.
ResNet architectures with different depths (axample given, ResNet-18, ResNet-50) have been utilized due to their ability to handle deeper networks without facing the vanishing gradient problem.
VGG architectures with varying layers have also been employed for their simplicity and effectiveness in feature extraction from images.
The CPResNet prototype took in consideration the ResNet-50 architecture, changing the last layers, where a new learnable layer with a weight learn factor of 10 and a bias learn rate factor also of 10 is added to the model replacing the current “fc1000”, a softmax layer is created to replace the current “fc1000_softmax” layer and a classification layer is also created to replace the “ClassificationLayer_fc1000”. Also, a new batch normalization layer is created to replace the “activation_49_relu” and a new RELU layer to replace the “avg_pool” as shown on figure 49.
The code used to develop the prototype was adapted from (Narayanan et al., 2020).

![image](https://github.com/kucaantunes/Automatic_Detection_of_Covid-19_on_CT_Scans/assets/26171557/ee5bce62-ecbf-4a68-a8a1-ae633834c503)

 
Figure 49. MatLab code used to replace layers from the standard ResNet-50 architecture.

This approach showed high accuracy results when compared with other architectures. For this reason, the developed prototype CPTresNet was made taking in consideration the modification of some layers of the ResNet-50 model and fine-tunning the model to the datasets in cause.
Concerning CPResNet a strategy of modifying the standard ResNet-50 architecture was followed by replacing specific layers: the "fc1000" was substituted with a new learnable layer featuring increased weight learn factor (10) and bias learn rate factor (10). The original "fc1000_softmax" layer was replaced by a newly created softmax layer for classification purposes. Additionally, a new classification layer took the place of the "ClassificationLayer_fc1000".
Furthermore, adjustments were made by introducing a fresh batch normalization layer to replace "activation_49_relu", while a new RELU layer was incorporated to replace "avg_pool".
The CPResNet was tested on the Covid-19 lung CT Scans dataset (Aria, 2021) (Dataset 4), being 7496 with Covid-19 and 944 normal (see Figure 35). Throughout development, this dataset was partitioned into training, validation, and testing subsets. The testing subset remains separate from the training data and is exclusively reserved for evaluating the model's ability to generalize that is, its capability to perform effectively on previously unseen data. Meanwhile, the validation subset serves the purpose of fine-tuning the model's hyperparameters. These hyperparameters encompass settings governing the training process, such as learning rate, the number of training epochs, and batch size. The validation set aids in selecting the hyperparameters that optimize the model's performance. Both the test and validation sets play pivotal roles in crafting robust deep learning models capable of generalizing to novel data. Leveraging these sets, researchers ensure their models aren't overly tailored to the training data, confirming their competence on real-world datasets.
 
 
![image](https://github.com/kucaantunes/Automatic_Detection_of_Covid-19_on_CT_Scans/assets/26171557/a16f9b4f-2464-485f-b8e0-e8c9d34a36e7)

Figure 35. Number of images with Covid-19 and without on the Covid-19 lung CT Scans dataset (Aria, 2021) (Dataset 4) and some sample images of the dataset.

![image](https://github.com/kucaantunes/Automatic_Detection_of_Covid-19_on_CT_Scans/assets/26171557/96204d48-8eaf-4044-93af-a94321050cbb)

![image](https://github.com/kucaantunes/Automatic_Detection_of_Covid-19_on_CT_Scans/assets/26171557/bf1ca881-9687-4cc9-8190-4bfd55816d44)



Figure 81 shows details about the “bnlayer” a batch normalization layer that replaced the standard “activation_49_relu” layer of the ResNet-50 and details of the “relulayer” a RELU layer that replaced the “avg_pool” of the standard ResNet-50 model.


 ![image](https://github.com/kucaantunes/Automatic_Detection_of_Covid-19_on_CT_Scans/assets/26171557/c78f5cc4-0412-4643-9946-5652297999f5)

Figure 79.  Details of two created layers that replaced two standard layers of the ResNet-50.

ResNet-50 is a popular convolutional neural network architecture, comprises several layers that facilitate deep learning for image recognition tasks. 
The initial layer accepts the input images, typically of size 224x224 pixels in three color channels (RGB).
ResNet-50 consists of multiple convolutional layers organized in blocks. These blocks contain residual connections (shortcuts) that allow the network to avoid vanishing gradient problems in deeper architectures.
The core concept of ResNet-50 lies in its residual blocks, which include stacked convolutional layers along with identity shortcuts. These blocks enable the learning of residual functions instead of attempting to directly learn the underlying mapping, which eases the training of very deep networks.
Throughout the architecture, max-pooling layers reduce the spatial dimensions of the features extracted by the convolutional layers, aiding in translation invariance and dimensionality reduction.
Towards the end of the network, fully connected layers process the high-level features extracted by previous layers to perform the final classification. In the case of ResNet-50, the architecture initially includes a fully connected layer called "fc1000" for 1000-class ImageNet classification.
The final layer applies the softmax function to the output of the last fully connected layer, transforming the network's final predictions into probabilities for each class.
Throughout the network, Rectified Linear Unit (ReLU) activation functions are commonly used after each convolutional and fully connected layer to introduce non-linearity and enable the network to learn complex patterns.
The batch normalization technique normalizes the output of each layer, mitigating issues related to internal covariate shift and accelerating convergence during training.
ResNet-50 specifically features 48 Convolutional layers along with these key components. Its design innovation lies in the introduction of residual connections, which facilitated the training of deeper networks while maintaining accuracy.
Understanding the architecture and function of each layer in ResNet-50 is crucial for adapting, fine-tuning, or exploring modifications to suit specific tasks like medical imaging or other specialized domains.
Adapting ResNet-50, a pre-existing convolutional neural network architecture, for Covid-19 detection in CT scans involved making strategic modifications to its layers. This process aimed to leverage the network's ability to extract features from images while tailoring it to recognize Covid-19-related patterns in CT scans.
The final layers (fully connected or classification layers) were altered to suit the specific classification needs for Covid-19 detection. This involved adjusting the number of output nodes and reconfiguring the architecture to accommodate binary (Covid-19 vs. non-Covid-19) classification.
The weights of certain layers were fine-tuned to learn Covid-19-specific features. Transfer learning allows leveraging pre-trained models like ResNet-50 and updating their weights using a smaller dataset of Covid-19 CT scan images, which helps the network adapt to this specific task.
Replacing and adding layers within the ResNet-50 architecture allowed to enhance its ability to recognize Covid-19-related patterns. This involved inserting additional convolutional layers, adjusting pooling layers and integrating attention mechanisms to focus on relevant regions in CT scans.
Preprocessing was used to normalize, augment and enhance the CT images before inputting them into the modified ResNet-50 model. Proper data preprocessing significantly impacts model performance.
Rigorous validation and optimization processes were used to fine-tune hyperparameters, such as learning rates, batch sizes and regularization techniques, to maximize the model's accuracy and generalization on Covid-19 detection tasks.
Interpretability techniques were incorporated like Grad-Cam to visualize and understand where the modified ResNet-50 focuses its attention when identifying Covid-19 indicators in CT scans. This aided in validating the network's decisions and aligning them with medical expertise.
Adapting ResNet-50 for Covid-19 detection in CT scans involved a combination of architectural adjustments, fine-tuning, and validation processes to ensure the model effectively learns and recognizes relevant patterns indicative of the disease. It required a nuanced understanding of both deep learning architectures and medical imaging to achieve accurate and reliable results.
Figure 82 shows the details of the model used for CPResNet by modifying the current ResNet-50 standard, mentioning the name, type, activation, learnable parameters and states.  
 
 
 
 
 ![image](https://github.com/kucaantunes/Automatic_Detection_of_Covid-19_on_CT_Scans/assets/26171557/3edbcbc6-5ce7-4fa0-9b50-6e8cf2c43b52)

 
 
 
 
 
Figure 80.  All the layers of the developed model used on the first approach.

In order to develop the model CPResNet, it was used MatLab, that also plots the confusion matrix and the ROC curve.

![image](https://github.com/kucaantunes/Automatic_Detection_of_Covid-19_on_CT_Scans/assets/26171557/8f8e4f17-558a-4aa7-93fc-1a15f3572c09)

Figure 83 shows the code used to define the path to the dataset, to display the number of images to each label of the dataset, and details of the fold.

![image](https://github.com/kucaantunes/Automatic_Detection_of_Covid-19_on_CT_Scans/assets/26171557/6883a781-6d8e-4812-9abe-1f2a124e7ca9)

 
Figure 81.  MatLab code to define details of the Covid-19 lung CT Scans dataset (Aria, 2021) (Dataset 4).

MatLab is a high-level programming language and numeric computing environment developed by MathWorks which provides a rich set of mathematical functions and libraries for solving a wide range of mathematical problems, from linear algebra to differential equations.
MatLab is a tool for data analysis, including data visualization, statistical analysis, and signal processing. It is also popular for developing algorithms for a variety of applications, such as deep learning, image processing, and control systems. MatLab is often used for prototyping and simulating new ideas and designs in a variety of fields, such as engineering, physics, and finance.
The IDE provides an interactive environment that allows users to experiment with code and see the results immediately. This makes it a great tool for learning and prototyping new ideas. It has a large collection of toolboxes that provide specialized functionality for a variety of domains, such as control systems, signal processing, and deep learning.
MatLab can be integrated with other programming languages and tools, such as C, C++, and Python. This allows users to leverage the strengths of different tools for their specific needs. MatLab is increasingly being used in the field of data science for tasks such as deep learning, big data analysis, and data visualization. Can be seen as a tool for scientific computing that is used in a wide variety of applications. It is a valuable tool for researchers, engineers, scientists, and data scientists alike.
With its extensive toolboxes, interactive environment, and integration with other tools, MatLab is a powerful tool for solving complex problems in a variety of domains.
Figure 84 shows the implementation of the ResNet-50 architecture. The code shows how to modify the ResNet-50 architecture's final layers. This entailed adding a new learnable layer with weight and bias learn factors of 10 to replace the existing "fc1000". Additionally, a newly created softmax layer replaced the current "fc1000_softmax", and a fresh classification layer was introduced to take the place of "ClassificationLayer_fc1000". Moreover, adjustments included the creation of a new batch normalization layer to substitute "activation_49_relu", and a new RELU layer was implemented in place of "avg_pool". The code also shows the preprocessing technique by applying a function to each of the images of the Covid-19 lung CT Scans dataset (Aria, 2021) (Dataset 4).
 
Figure 82.  MatLab code that changes the standard layers of the ResNet-50 architecture.

Some MatLab functions were used that are instrumental in training neural networks and handling image data augmentation:
The trainingOptions is a function used for configuring the training options or settings for training a neural network model. It allows to specify various parameters related to training, such as optimization algorithm, mini-batch size, maximum epochs, learning rate, and more. 
The imageDataAugmenter is a function in MatLab used for creating an image data augmenter object. This object facilitates the augmentation of image data during the training process. Augmentation involves applying transformations to the input images, like rotation, scaling, flipping, or adjusting brightness and contrast. 
The trainNetwork is a function used for training a neural network model using the specified data, layers, and training options. It takes in the layers of the neural network, training data, and the previously defined training options to initiate the training process. 
These functions are essential components of the MatLab Deep Learning Toolbox, providing a streamlined way to configure training parameters, augment image data for improved model generalization, and perform the actual training of neural network models. They enable users to efficiently train deep learning models with flexibility in defining training configurations and data augmentation strategies.
Figure 85 displays the training options where the maximum number of epochs and the initial learning rate can be defined among other parameters, also shows the process used to augment images and set to 224x224 and also the function to perform the network training.

 
Figure 83.  MatLab code to define training options, augment images and perform the network training.

Figure 86 mentions the MatLab code used for displaying the confusion matrix and the ROC curve and also to perform calculation of some of the performance metrics.  
To plot a confusion matrix in MatLab, was utilized the confusionmat function to compute the confusion matrix from your predicted and actual labels. Then, use the confusionchart function (introduced in newer MatLab versions) to visualize it.
For plotting an ROC curve in MatLab, was used the perfcurve function, which evaluates classifier performance by varying the discrimination threshold.
The perfcurve function allows to compute the ROC curve from predicted scores and actual labels. Adjust the actual labels and predicted scores with your own data. The resulting plot shows the ROC curve, where labels, titles, and styling can be customized as needed.
These MatLab functions provide straightforward ways to visualize the performance metrics of classification models, aiding in assessing their accuracy and effectiveness. The code was adjusted according to the datasets used and the requirements for optimal representation and analysis.
 
Figure 84.  MatLab code to generate the performance metrics, the confusion matrix and the ROC curve.

Some functions that are part of the Image Processing and Deep Learning Toolboxes were used for image manipulation and deep learning visualization:
The imresize is a function used for resizing images in MatLab. It allows users to adjust the dimensions of images, either enlarging or reducing them. This function takes an input image and resizes it to a specified size or scale. 
The IMout is a placeholder or variable name used to store or represent an output image in MatLaB. It was used to designate an output image resulting from image processing or deep learning operations.
The activations are a function used in deep learning workflows to extract and compute activations (outputs) from specific layers of a neural network. It allows users to retrieve the activations of neurons or feature maps at different layers of a pre-trained network using input data.
The deepDreamImage is a function used to generate visually intriguing images by modifying input images to enhance the features that activate specific neurons in a neural network. It's a technique that amplifies patterns recognized by the network to create artistic or dreamlike visuals. 
These functions cater to different aspects of image manipulation, feature extraction from neural networks, and generating visually appealing images using deep learning techniques within the MatLab environment.
Figure 87 shows the code used to preprocess the images.

 
Figure 85.  MatLab code to manipulate the images.
The imread is a function used to read images into MatLab. It reads various image file formats (like JPEG, PNG, BMP, etc…) and creates a matrix representing the image. 
The ismatrix is a logical function that checks if a variable is a matrix. It returns true if the input is a 2-dimensional matrix and false otherwise. This function is useful for verifying if a variable is a matrix before performing matrix-specific operations.
The rgb2gray is a function used for converting RGB images to grayscale images in MatLab. It takes an RGB image as input and converts it to a grayscale image by averaging the red, green, and blue channels' values. 
The cat is a function used for concatenating arrays along specified dimensions in MatLab. It can concatenate arrays horizontally or vertically based on the dimension specified. 
These functions in MatLab serve various purposes, from handling image data to checking matrix properties and manipulating arrays for effective data handling and processing.
Figure 88 shows the code used for treatment of the images, converting to grayscale.

 
Figure 86.  MatLab code of the function to treat images.

The prototype CPResNet has a mechanism to apply XAI on the predictions. Figure 44 shows the classification predictions of some CT scans of the Covid-19 lung CT Scans dataset (Aria, 2021) (Dataset 4).

![image](https://github.com/kucaantunes/Automatic_Detection_of_Covid-19_on_CT_Scans/assets/26171557/081ef26d-390c-43bc-8575-fe1bea417a55)

 
Figure 44.  Prediction obtained for the CT scans, mentioning if they have Covid-19 or not.

In order to make the results more understandable, the prototype uses XAI, to provide explanations and justifications for deep learning model predictions, recommendations, or decisions. This is crucial in scenarios where the model's output impacts critical decisions in healthcare. XAI aims to make AI models more interpretable, allowing humans to understand the reasoning behind the model's predictions or decisions. It involves techniques that reveal the inner workings of AI models, making them more transparent and understandable. By providing explanations for AI model outputs, XAI helps build trust in AI systems. XAI is important in healthcare by interpreting medical diagnosis made by AI models. XAI aims to bridge the gap between the "black-box" nature of complex AI models and the need for human understanding and trust by providing explanations that help humans comprehend and trust AI-driven decisions.


Figure 45 shows the use of Grad-CAM on the predictions of certain CT scans. Grad-CAM is a technique used for visualizing and understanding Convolutional Neural Networks by highlighting regions of an image that influence the network's decision-making process.
In MATLAB, Grad-CAM was implemented to visualize the important regions of an image that contribute to the network's prediction for a specific class. It helps understand which parts of the input image are critical in the network's decision.
The images were loaded and preprocessed according to the requirements of the trained CPResNet model.
The CPResNet model was used to extract feature maps from intermediate convolutional layers. The gradients were computed of the target class's output with respect to these feature maps.The importance of each feature map was computed by taking a weighted sum of the feature maps using the gradients obtained in the previous step.
A heatmap was generated by overlaying the weighted feature maps on the input image. This heatmap highlights the regions that the model focuses on for predicting a specific class.

 ![image](https://github.com/kucaantunes/Automatic_Detection_of_Covid-19_on_CT_Scans/assets/26171557/9ee3b8fb-5160-478a-bd32-540209399c4a)

Figure 45.  Use of Grad-CAM on the predicted classifications.

Figure 46 shows the mechanism used to apply LIME on the results. LIME is a technique used for explaining the predictions of machine learning models, regardless of the underlying algorithm or model type. It's designed to provide local interpretability, explaining individual predictions made by a model in a human-understandable way. In MATLAB, LIME was implemented to explain the predictions of the CPResNet. The coefficients or importance of features provided by the interpretable model were analyzed to understand the influence of different features on the model's prediction for the selected instance. LIME is a technique used in this research to provide explanations or interpretations for individual predictions made by the developed prototypes. It aims to create locally faithful explanations that help humans understand why a particular prediction was made by a model, even if the model itself is a "black-box" and not easily interpretable. LIME focuses on explaining individual predictions rather than the overall behavior of a model. It creates simple and interpretable models that approximate the complex model's behavior around a specific instance. LIME is model-agnostic, meaning it can be applied to any deep learning model, whether it's a decision tree, random forest, support vector machine, neural network, or other complex models. LIME generates perturbed samples around the instance to be explained, creates a simpler interpretable model  on these perturbed samples, and interprets the coefficients or feature importance of this interpretable model to explain the prediction. By providing local explanations for specific instances, LIME helps users understand which features were influential in a model's prediction for that instance. LIME serves as a tool for explaining individual predictions made by machine learning models in a way that is more understandable and interpretable for humans, aiding in trust, transparency, and debugging of deep learning models, especially in critical or high-stakes applications.

![image](https://github.com/kucaantunes/Automatic_Detection_of_Covid-19_on_CT_Scans/assets/26171557/ad2a4eb7-2278-437d-a352-fe20c65cc769)

 
Figure 46.  Use of LIME on the predicted classifications.


Figure 47 shows the structure of CPResNet that uses datasets of CT scans with and without Covid-19, the model uses the initial layers of ResNet-50 and a novel learnable layer with weight and bias learn factors increased by a factor of 10 was introduced to replace the existing "fc1000" layer. Furthermore, a newly constructed softmax layer took the position of the previous "fc1000_softmax", and a freshly created classification layer was added, replacing "ClassificationLayer_fc1000". Additional modifications involved the introduction of a new batch normalization layer, which replaced "activation_49_relu", and a new RELU layer was implemented in lieu of "avg_pool". XAI techniques are applied to the predicted images like grad-CAM and LIME and the performance results are obtained such as precision, recall, AUC, accuracy among others.

![image](https://github.com/kucaantunes/Automatic_Detection_of_Covid-19_on_CT_Scans/assets/26171557/c10adffe-7f56-4fbd-97c5-c3cbd7fa4b5b)


 Figure 47.  Schematic representation of CPResNet.

The developed prototype CPResNet was developed to classify CT scans using a modified architecture if the ResNet-50 and adding LIME and grad-CAM to explain the predictions and obtaining the results like accuracy, precisiona, recall, AUC, F1-score among others.
Figure 63 shows details of the Covid-19 lung CT Scans dataset (Aria, 2021) (Dataset 4), where 7495 CT scans with Covid-19 were used and 944 CT scans without Covid-19 were referenced.  

 
Figure 63.  Number of images with and without Covid-19 and some CR scans of the Covid-19 lung CT Scans dataset (Aria, 2021) (Dataset 4).

Figure 64 shows details of the initial layers of the CPResNet, being the first an image input layer of dimensions 224x224x3 with zerocenter normalization, the second layer a 2D convolution with stride [2 2], the third a batch normalization with 64 channels and the fourth a RELU.
 
Figure 64.  Details of the initial layers of the CPResNet architecture.

 
 
  

