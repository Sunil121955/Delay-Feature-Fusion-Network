%Load the query image 
[filename, pathname] = uigetfile({'*.jpg', 'Image Files (*.jpg)'}, 'Select an Image');

% Check if a file was selected
if isequal(filename, 0) || isequal(pathname, 0)
    disp('No image selected.');
else
    % Read the selected image
    fullFilePath = fullfile(pathname, filename);
    query_img = imread(fullFilePath);

    % Display the image
    figure;
    imshow(query_img);
    title('Query Imgage');
end

%Load the Network
net = DFFnet;
layerName='pool10';
inputSize = net.Layers(1).InputSize;

%Extract the query image feature 
resize_image = imresize(query_img, inputSize(1:2));
query_feature = activations(net, resize_image, layerName, 'OutputAs', 'rows');
query_feature = [query_feature 0];

%Load the extracted feature of all images
load('dataset.mat','feature');

%Combined the feature with query image feature
dataset = [query_feature; feature];

features = dataset(:,1:end-1);
labels=dataset(:,end);
% Perform t-SNE
Y = tsne(features);

% Plot the results
scatter(Y(:,1), Y(:,2));
scatter(Y(:,1), Y(:,2),[], labels);
colormap(jet(256));
dataset=[Y labels];

numOfReturnedImages=20;
metric=2;
queryImageFeatureVector=dataset(1,:);
dataset = dataset(2:end,:);
[precision, recall]= L2_metric(numOfReturnedImages, queryImageFeatureVector, dataset, metric, filename);

