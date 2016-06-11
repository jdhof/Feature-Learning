function [Codebook, iter] = MySphericalKMeans(X, K, Codebook)
% Author: Jan-David Hof

%Spherical K-Means Codebook

%Usage: Clustering the data in to K cluster using the sperical K-means

%s(i) is a codevektor, which is assigned to input vector (X) x(i).
%D(j) j- codeweord.

%Input: 
%   (matrix) X : D x N      matrix representing feature vectors by columns
%                       where D is the number of dimensions and N is the
%                       number of vectors.
%   (scalar) K          The number of clusters.
%   (matrix) Codebook   if available


%Output:
%   (Matrix) Codebook 
%   (scalar) number of iterations
%

%Variables:
%   X : input data vector
%   D : dictionary of k vectors
%   s : codevector of D
%
%
%
[D N] = size(X);
if K > N,  
    %error('K must be less than or equal to the number of vectors N');
end

if numel(Codebook)==0 %if codebook is not handed over, initialize it new
    
% ----------------Initial centroids/codebook-------------------------
    
%initialise centroids: random on the unit sphere

     [num_datapoints, dim] = size(X);
     d = size(X(:,1));
     Codebook = normrnd(0,1,D,K);
     
%-----------------normalise Codebook to unit length (L2 norm)-----------
      

    for j = 1:K
      Codebook(:,j) = Codebook(:,j)./norm(Codebook(:,j),2);
      norm(Codebook(:,j))
    end  
end

%------------------Calculate Codebook-----------------------------------

[row col] = size(Codebook);

improvedRatio = Inf;
distortion = Inf;
iter = 0;

dist = zeros(col,N); % distances between Codebookvectors and inputvectors
while true
   S=zeros(col,N);
   ClusterData = zeros(1,N);
   
   %Centroid = Codebookvektor
        %------------------- samples to centroids-------------------

    for j = 1:K %over all clusters
        for i = 1:N %over all inputcolumns               
          dist(j,i) = Codebook(:,j)' * X(:, i);
        end
    end
    
    [dataForCluster, Cluster] = max(dist); %argmax
    %analysize all distances. Distance D x(i) which is the nearest
    %is saved in dataForCluster. Cluster consists
    %the D, for which biggst value occured

    for i=1:col
        % Get the id of samples which were clusterd into cluster i.
        idx = find(Cluster == i);
        S(i,idx) = Codebook(:,i)' * X(:, idx); %sj(i) i = row
    end
    %every s(i) vector is constrained to have at most one
    %non-zero entry (each column of input-matrix is assigned to one centroide)
    

    %------------------ updating of centroids----------------------
   
    Codebook = X*S' + Codebook;

%-----------------normalise Codebook to unit length (L2 norm)-----------
      

    for j = 1:K
      Codebook(:,j) = Codebook(:,j)./norm(Codebook(:,j),2);
      norm(Codebook(:,j))
    end
    
     %------------------ abort criterion ----------------------

     %distortion = distance
     
    old_distortion = distortion;
    distortion = 0;
  for i = 1:N
       A = Codebook * S(:,i);
       B = X(:,i);
       sqr = sqrt(sum((A-B).^2));
       distortion = distortion + sqr^2;
  end  
   
  improvedRatio = 1 - (distortion / old_distortion);
  stopIter = .005;
  iter = iter + 1;
  if (abs(improvedRatio) < stopIter) || (iter >= 10) , break, end;
  
  
end
fprintf('Finished in %d iterations\n',iter)

end