% Load the dataset of image
% Open a dialog to select a folder
selectedFolder = uigetdir('.', 'Select a Folder');

% Check if a folder was selected
if isequal(selectedFolder, 0)
    disp('No folder selected.');
else
    disp(['Selected folder: ', selectedFolder]);
end
image = dir(fullfile(selectedFolder, '*.jpg'));

% Initialize an empty cell array to store the features of all the videos in the dataset
img_features = cell(length(image), 1);
img_name=cell(length(image),1);

%Load the Network
net = DFFnet;
layerName='pool10';
inputSize = net.Layers(1).InputSize;

%Dataset creation
for i = 1:length(image)
    % Load the current image
    current_image = imread(fullfile(selectedFolder, image(i).name));
    % Resize the current image to the input size of the CNN
    resize_image = imresize(current_image, inputSize(1:2));
    % Extract features from the current image using the CNN
    currentFeatures = activations(net, resize_image, layerName, 'OutputAs', 'rows');
     % Extract features from gray_frame and store in features matrix
     img_features{i, :} = currentFeatures;   
    %Extract the file name
    [~,image_name,ext] = fileparts(image(i).name);  
     img_name{i} = str2num(image_name);
end
%Mearge the image feature with image name for ceate feature vector of each images
feature = [img_features img_name];
save('dataset.mat','feature');