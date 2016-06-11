function [ xPCAWhite, U, xmean, k] = pca( x )
% Author: Jan-David Hof
% algorithm is based on "http://deeplearning.stanford.edu/wiki/index.php/Exercise:PCA_and_Whitening"


%Input: 
%   (Matrix) x :            matrix of log-scaled mel spectrograms (maybe shingled)

%Output:
%   (Matrix) xPCAWhite :    PCA (whitned) mel spectrograms
%   (Matrix) U :            Eigenvectors
%   (Scalar) k :            number of components to retain
%   (Vector) xmean:         Mean of each datapoint

%



%---------------------------Zero-mean the data------------------------

xmean = mean(x, 2);             %variables = rows, observations = columns
x = bsxfun(@minus, x, xmean);   %normalize data

%-------------------------Compute PCA------------------------------

PCA = zeros(size(x)); 
[nfeatures, nsamples ] = size(x);
sigma = x * x' ./ nfeatures;
[U S V] = svd(sigma);   %Compute Eigenvectors
PCA = U' * x;   %Transform data in Basis

%------------------------Find K for 99% variance (Reduzierung der Dimensionalität)-------------------

k = 0; % Set k accordingly
eigenvalues = diag(S);
current_var = 0;
total_var = sum(eigenvalues);
while current_var < 0.99 * total_var
    k = k+1;
    current_var = current_var + eigenvalues(k);
end

 xHat = zeros(size(x));
 xHat = U(:, 1:k) * U(:, 1:k)' * x;
 
%--------------------------Whitening--------------------------
epsilon = 0.1;
xPCAWhite = zeros(size(x));
xPCAWhite = diag(1./sqrt(diag(S) + epsilon)) * U' * x; 
%xPCAWhite = diag(1./sqrt(diag(S) + epsilon)) * xHat; %uncomment,
%if dimensionality should reduced by calculated value k (components to retain)


%--------------------------ZCA whitening-----------------------
xZCAWhite = zeros(size(x));
xZCAWhite = U * xPCAWhite;

end
