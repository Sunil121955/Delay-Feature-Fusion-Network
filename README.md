# Delay-Feature-Fusion-Network

Steps:
1. Train the Delay Feature Fusion Network using the Training_model.m file.
2. After training, we utilize this model for extracting image features for constructing the feature vector.
3. For any Query image, we extract the feature vector using a trained model.
4. Merge the image features vector with the query image feature vector.
5. Apply the t-SNE on these merged features.
6. Retrieve the top N relevant images to the query image.

Delay Feature Fusion Network design within the framework of Squeeze Network. Extracting more relevant image features as compared to another network as well as fewer parameters. The architecture of DFFnet is shown above figure. 

This experiment was carried out on two publically available datasets Corel and ImageNet dataset. Corel dataset mostly use for image retrieval tasks while imageNet dataset mostly uses for object recognition tasks.
