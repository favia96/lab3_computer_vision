%% LAB 3 - COMPUTER VISION, December 2019
%% by Federico Favia, Martin De Pellegrini

%% Initialization
clear ; close all; clc

%% Part 1: K-means segmentation
K = 4;               % number of clusters used
L = 15;              % number of iterations
seed = 14;           % seed used for random initialization
scale_factor = 1.0;  % image downscale factor
image_sigma = 1.0;   % image preblurring scale
RANDSAMP = true;
DEBUG = true;
verbose = 1;

I = imread('orange.jpg');
I = imresize(I, scale_factor);
Iback = I;
d = 2*ceil(image_sigma*2) + 1;
h = fspecial('gaussian', [d d], image_sigma);
I = imfilter(I, h);
Ivec = reshape(I, size(I,1)*size(I,2), 3); %added

tic
[segm, centers, empty, cen_idx, count] = kmeans_segm(I, K, L, seed, RANDSAMP, DEBUG);
toc
segm = reshape(segm,size(I,1),size(I,2),1); % added
Inew = mean_segments(Iback, segm);
Iover = overlay_bounds(Iback, segm);
%imwrite(Inew,'result/kmeans1.png')
%imwrite(I,'result/kmeans2.png')

% plot
if verbose > 0
    subplot(1,3,1); imshow(I); title('original');
    subplot(1,3,2); imshow(Inew); title(sprintf('K = %d, L = %d',K,L));
    subplot(1,3,3); imshow(Iover); title('overlay bound');
end

if verbose > 1
    figure()
    histogram(I)
    
    figure()
    [m, n, d] = size(I);
    Ximg = reshape(I, m * n, d);
    Xsegm = reshape(segm, m * n, 1);
    for i = 1 : K
        X = Ximg(Xsegm == i, :);
        plot3(X(:, 1), X(:, 2), X(:, 3), '.');
        hold on
    end
    plot3(centers(:,1), centers(:,2), centers(:,3), 'k*', 'MarkerSize', 30); % 3d plot of clusters
    hold off 
    
end
