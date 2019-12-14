function [segmentation, centers, empty, cen_idx, count] = kmeans_segm(image, K, L, seed)

    nrows = size(image, 1);
    ncols = size(image, 2);
    Ivec = reshape(double(image), nrows * ncols, 3);
    delta = 100 * ones(K, 3);
    clus_centers = zeros(K, 3);
    clus_centers_new = zeros(K, 3);
    idx_old = zeros(1, nrows * ncols);
    empty = false;
    threshold = 0.01;
    count = 0;

    % Let X be a set of pixels and V be a set of K cluster centers in 3D (R,G,B).
    % Randomly initialize the K cluster centers
    rng(seed, 'twister');
    idx = randsample(nrows * ncols, K);
    for i = 1 : K
            clus_centers(i, :) = Ivec(idx(i), :);
    end

    % Compute all distances between pixels and cluster centers
    D = pdist2(clus_centers, Ivec, 'euclidean');

    % Iterate L times
    for i = 1 : L
            % Assign each pixel to the cluster center for which the distance is minimum
            [~, cen_idx] = min(D);
        
            % Recompute each cluster center by taking the mean of all pixels assigned to it
            for j = 1 : K
                n_idx = find(cen_idx == j);
                if isempty(n_idx) == true
                    empty = 1;
                    n = randsample(nrows * ncols, 1);
                    clus_centers(j, :) = Ivec(n, :);
                else
                    clus_centers(j, :) = mean(Ivec(n_idx, :));
                end
            end
            % Recompute all distances between pixels and cluster centers
            D = pdist2(clus_centers, Ivec, 'euclidean');
    end
   
    [~, cen_idx] = min(D);
    segmentation = reshape(cen_idx, nrows, ncols);
    centers = clus_centers;
end