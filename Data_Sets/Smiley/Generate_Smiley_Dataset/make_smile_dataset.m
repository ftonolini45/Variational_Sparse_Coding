function make_smile_dataset

% Function to synthesise artificial dataset of smiley with group-sparse
% features

n_tot = 60000; % number of total samples to synthesise
alpha = 0.5; % sparsity coefficient - probability of each feature group being present
n_feat = 18; % number of total features
r = 10; % radius of the smiley circular head in pixels

X = ones(32,1)*(1:32)-16.5;
Y = (1:32)'*ones(1,32)-16.5;
R = sqrt(X.^2+Y.^2);

S = zeros(n_tot,n_feat);
V = zeros(n_tot,n_feat);
X = zeros(n_tot,32*32);

bt = imread('bowtie.png'); % bow-tie image to be used for one of the feature groups
bt = sum(bt,3);
bt = imresize(bt,[11,17]);
bt = round(mat2gray(bt));

% making the samples
for i = 1:n_tot

    I = zeros(32,32);
    I(R<=r)=1;
    
    s = round(rand(1,4)-0.5+alpha);
    sv = zeros(1,n_feat);
    
    v1a = 0.5*randn;
    v1b = 0.5*randn;
    v1c = 0.5*randn;
    v1d = 0.5*randn;
    if s(1)==1
        sv(1:4) = 1;
    end
    
    v2a = 0.5*randn;
    v2b = 0.5*randn;
    v2c = 0.5*randn;
    v2d = 0.5*randn;
    v2e = 0.5*randn;
    v2f = 0.5*randn;
    if s(2)==1
        sv(5:10) = 1;
    end
    
    v3a = 0.5*randn;
    v3b = 0.5*randn;
    v3c = 0.5*randn;
    if s(3)==1
        sv(11:13) = 1;
    end
    
    v4a = 0.5*randn;
    v4b = 0.5*randn;
    v4c = 0.5*randn;
    v4d = 0.5*randn;
    v4e = 0.5*randn;
    if s(4)==1
        sv(14:18) = 1;
    end
    
    % bowtie
    I = add_bowtie(I,s(1),bt,v1a,v1b,v1c,v1d);
    % eyes
%     I = internal_circles(I,s(2),v3a,v3b,v3c,v3d,v3e,v3f);
      I = add_eyes(I,s(3),v3a,v3b,v3c);
    % mouth
    I = mouth(I,s(4),v4a,v4b,v4c,v4d,v4e);
    % hat
    I = make_hat(I,s(2),v2a,v2b,v2c,v2d,v2e,v2f);
    
    v = [v1a,v1b,v1c,v1d,v2a,v2b,v2c,v2d,v2e,v2f,v3a,v3b,v3c,v4a,v4b,v4c,v4d,v4e];
    x = reshape(I,1,32*32);
    X(i,:) = x;
    S(i,:) = sv;
    V(i,:) = v;
    
end

% display a sample from the generated set
xi = X(1,:);
Ii = reshape(xi,32,32);
figure(1)
imshow(Ii,'InitialMagnification',1000)

% save the generated dataset with associated continuous source variables V and binary source variables S 
save('smily_sparse_dataset_train_Alpha05.mat','X','S','V')