%% LAB 3 - COMPUTER VISION, December 2019
%% by Federico Favia, Martin De Pellegrini

%% Initialization
clear ; close all; clc

%% Part 1: K-means segmentation
K = 5;               % number of clusters used
L = 12;              % number of iterations
seed = 14;           % seed used for random initialization
scale_factor = 1.0;  % image downscale factor
image_sigma = 1.0;   % image preblurring scale
verbose = 1;

I = imread('tigers.jpg');
I = imresize(I, scale_factor);
Iback = I;
d = 2*ceil(image_sigma*2) + 1;
h = fspecial('gaussian', [d d], image_sigma);
I = imfilter(I, h);
Ivec = reshape(I, size(I,1)*size(I,2), 3); %added

tic
[segm, centers, empty, cen_idx, count] = kmeans_segm(I, K, L, seed);
toc
segm = reshape(segm,size(I,1),size(I,2),1); % added
Inew = mean_segments(Iback, segm);
Iover = overlay_bounds(Iback, segm);
%imwrite(Inew,'result/kmeans1.png')
%imwrite(I,'result/kmeans2.png')

% plot
if verbose > 0
    figure('name','K-means segm')
    subplot(1,3,1); imshow(I); title('Original');
    subplot(1,3,2); imshow(Inew); title(sprintf('K = %d, L = %d',K,L));
    subplot(1,3,3); imshow(Iover); title('Overlay bound');
    sgtitle('K-means segm');
end

if verbose > 1
    figure('name','Histogram')
    histogram(I)
    sgtitle('Histogram');
    
    figure('name','Plot of clusters')
    [m, n, d] = size(I);
    Ximg = reshape(I, m * n, d);
    Xsegm = reshape(segm, m * n, 1);
    for i = 1 : K
        X = Ximg(Xsegm == i, :);
        plot3(X(:, 1), X(:, 2), X(:, 3), '.');
        hold on
    end
    plot3(centers(:,1), centers(:,2), centers(:,3), 'k*', 'MarkerSize', 30); % 3d plot of clusters
    sgtitle('Plot of clusters');
    hold off    
end

%% Part 2: Mean-shift segmentation
scale_factor = 0.5;       % image downscale factor
spatial_bandwidth = 5.0;  % spatial bandwidth
colour_bandwidth = 5.0;   % colour bandwidth
num_iterations = 40;      % number of mean-shift iterations
image_sigma = 1.0;        % image preblurring scale

I = imread('orange.jpg');
I = imresize(I, scale_factor);
Iback = I;
d = 2*ceil(image_sigma*2) + 1;
h = fspecial('gaussian', [d d], image_sigma);
I = imfilter(I, h);

segm = mean_shift_segm(I, spatial_bandwidth, colour_bandwidth, num_iterations);
Inew = mean_segments(Iback, segm);
I = overlay_bounds(Iback, segm);
%imwrite(Inew,'result/meanshift1.png')
%imwrite(I,'result/meanshift2.png')
figure('name','Mean-shift segm')
subplot(1,2,1); imshow(Inew); title(sprintf('Spatial bandwidth = %.1f, colour bandwidth = %.1f', spatial_bandwidth, colour_bandwidth));
subplot(1,2,2); imshow(I); title('Overlay bound');
sgtitle('Mean-shift segm');

%% Part 3: Normalized Cut segmentation
colour_bandwidth = 20.0; % color bandwidth
radius = 6;              % maximum neighbourhood distance
ncuts_thresh = 0.4;      % cutting threshold
min_area = 20;          % minimum area of segment
max_depth = 8;           % maximum splitting depth
scale_factor = 0.4;      % image downscale factor
image_sigma = 2.0;       % image preblurring scale

I = imread('orange.jpg');
I = imresize(I, scale_factor);
Iback = I;
d = 2*ceil(image_sigma*2) + 1;
h = fspecial('gaussian', [d d], image_sigma);
I = imfilter(I, h);

segm = norm_cuts_segm(I, colour_bandwidth, radius, ncuts_thresh, min_area, max_depth);
Inew = mean_segments(Iback, segm);
I = overlay_bounds(Iback, segm);
%imwrite(Inew,'result/normcuts1.png')
%imwrite(I,'result/normcuts2.png')

figure('name','Norm Cut segm')
subplot(1,2,1); imshow(Inew); title('Norm Cut Segmented');
subplot(1,2,2); imshow(I);  title('Overlay bound');
sgtitle('Norm Cut segm');

%% Part 4: Graph Cut segmentation
scale_factor = 0.5;          % image downscale factor
area = [ 80, 110, 570, 300 ] % image region to train foreground with
K = 8;                      % number of mixture components
alpha = 8.0;                 % maximum edge cost
sigma = 10.0;                % edge cost decay factor

I = imread('tiger1.jpg');
I = imresize(I, scale_factor);
Iback = I;
area = int16(area*scale_factor);
[ segm, prior ] = graphcut_segm(I, area, K, alpha, sigma);

Inew = mean_segments(Iback, segm);
I = overlay_bounds(Iback, segm);
%imwrite(Inew,'result/graphcut1.png')
%imwrite(I,'result/graphcut2.png')
%imwrite(prior,'result/graphcut3.png')
figure('name','Graph Cut segm')
subplot(2,2,1); imshow(Inew); title('Graph Cut segmented');
subplot(2,2,2); imshow(I); title('Overlay bound');
subplot(2,2,3); imshow(prior); title('Prior foreground probabilities');
sgtitle('Graph Cut segm');


