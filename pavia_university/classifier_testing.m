%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Routine to obtain bands of the fused features for different lambdas  
%   Reference
%
%   [1] Ramirez, J., Martinez-Torre, J.I., & Arguello H. 2020. Feature
%   Fusion From Multispectral And Hyperspectral Compressive Data For
%   Spectral Image Classification.
%
%   Author:
%   Juan Marcos Ramirez, PhD. GOT-FUND Fellow (2019-2021)
%   Universidad Rey Juan Carlos, Mostoles, Spain
%   email: juanmarcos.ramirez@urjc.es
%
%   Date: Jan, 2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
close all;

addpath(genpath('../sources/'));

lambda1 = 0.01;
lambda2 = 5e-4;

% Loading data
load('../data/Pavia_university_cropped.mat');
Io = pavia_university_cropped;
clear pavia_university_cropped;

load('../Data/Pavia_university_gt.mat');
ground_truth = pavia_university_gt;
clear pavia_university_gt;

[N1, N2, L] = size(Io);

%% Building HS and MS images
% spatial downsampling
p       = 4;
window  = 5;
sigma   = 1.00;
I_HS    = spatial_blurring(Io, p, window, sigma,'sum');

% spectral downsampling
q = 4;
I_MS = spectral_blurring(Io, q,'sum');

%% CSI shots (capturing compressive measurements)
compression_ratio   = 0.25;
[N1_HS, N2_HS,L_HS] = size(I_HS);
[N1_MS, N2_MS,L_MS] = size(I_MS);

% Patterned measurements
[shot_patt_HS, shots_HS, num_hs_filters, ~, filter_pattHS, filterSetPattHS] = ...
    patterned_shots(I_HS, compression_ratio,'binary');
[shot_patt_MS, shots_MS, num_ms_filters, ~, filter_pattMS] = ...
    patterned_shots(I_MS, compression_ratio,'binary');

%% Feature Fusion
% Proposed feature fusion using the patterned architecture
dictionary = 'wav2_dct1'; %2D Wavelet + 1D DCT (for dyadic spatial dimensions only)
% dictionary = 'dct2_dct1'; %3-D DCT
alg_parameters.tol  = 1e-6;
alg_parameters.prnt = 1;
alg_parameters.mitr = 200;
alg_parameters.rho  = 0.1;

fusedFeatPatt    = feature_fusion_direct(shot_patt_HS,shot_patt_MS,...
    filter_pattHS,filter_pattMS,q,p,lambda1,lambda2,dictionary,alg_parameters);

signatures = reshape(Io,[N1*N2 L])';

%% Classification Stage
training_rate = 0.1;
[training_indexes, test_indexes] = classification_indexes(ground_truth, training_rate);
T_classes = ground_truth(training_indexes);

fusedFeatPatt_train = fusedFeatPatt(:,training_indexes);
fusedFeatPatt_test  = fusedFeatPatt(:,test_indexes);

signatures_train    = signatures(:,training_indexes);
signatures_test     = signatures(:,test_indexes);

disp(['Predicting classes...']);
nTrees = 300;
clf = TreeBagger(nTrees,fusedFeatPatt_train',T_classes, 'Method', 'classification');
class_hat1 = str2double(clf.predict(fusedFeatPatt_test'));

t = templateSVM('Standardize',1,'KernelScale','auto');
Md1        = fitcecoc(fusedFeatPatt_train',T_classes,'Learners',t);
class_hat2 = predict(Md1, fusedFeatPatt_test');

num_classes = max(max(ground_truth));
T = zeros(num_classes, length(training_indexes));
for i = 1: length(training_indexes)
    T(ground_truth(training_indexes(i)),i)= 1;
end
net = feedforwardnet(20);
net.trainParam.showWindow = false;
net1 = train(net,fusedFeatPatt_train,T);
class_hat3 = vec2ind(net1(fusedFeatPatt_test))';

nTrees = 300;
clf = TreeBagger(nTrees,signatures_train',T_classes, 'Method', 'classification');
class_hat4 = str2double(clf.predict(signatures_test'));

Md2        = fitcecoc(signatures_train',T_classes,'Learners',t);
class_hat5 = predict(Md2, signatures_test');

net.trainParam.showWindow = false;
net2 = train(net,signatures_train,T);
class_hat6 = vec2ind(net2(signatures_test))';

%% Building the classification maps
training_set_image = zeros(size(ground_truth,1), size(ground_truth,2));
training_set_image(training_indexes) = ground_truth(training_indexes);

classmap1    = class_map_image(ground_truth, class_hat1, training_indexes, test_indexes);
classmap2    = class_map_image(ground_truth, class_hat2, training_indexes, test_indexes);
classmap3    = class_map_image(ground_truth, class_hat3, training_indexes, test_indexes);
classmap4    = class_map_image(ground_truth, class_hat4, training_indexes, test_indexes);
classmap5    = class_map_image(ground_truth, class_hat5, training_indexes, test_indexes);
classmap6    = class_map_image(ground_truth, class_hat6, training_indexes, test_indexes);

[OA1, AA1, kappa1] = compute_accuracy(ground_truth(test_indexes), uint8(classmap1(test_indexes)));
[OA2, AA2, kappa2] = compute_accuracy(ground_truth(test_indexes), uint8(classmap2(test_indexes)));
[OA3, AA3, kappa3] = compute_accuracy(ground_truth(test_indexes), uint8(classmap3(test_indexes)));
[OA4, AA4, kappa4] = compute_accuracy(ground_truth(test_indexes), uint8(classmap4(test_indexes)));
[OA5, AA5, kappa5] = compute_accuracy(ground_truth(test_indexes), uint8(classmap5(test_indexes)));
[OA6, AA6, kappa6] = compute_accuracy(ground_truth(test_indexes), uint8(classmap6(test_indexes)));
%% Display results
band_set_ms1=[floor(2*L_MS/3) floor(L_MS/4) 1];
band_set_hs1=[floor(2*L_HS/3) floor(L_HS/4) 1];
normColor = @(R)max(min((R-mean(R(:)))/std(R(:)),2),-2)/3+0.5;

subplot(241);
imshow(label2color(ground_truth,lower('pavia')),'Border','tight');
title('Ground truth');

subplot(242);
imshow(label2color(classmap1,lower('pavia')),'Border','tight');
title(['SVM. OA: ' num2str(OA1*100,4) ' %'])
xlabel('Proposed');

subplot(243);
imshow(label2color(classmap2,lower('pavia')),'Border','tight');
title(['RF. OA: ' num2str(OA2*100,4) ' %'])
xlabel('Proposed');

subplot(244);
imshow(label2color(classmap3,lower('pavia')),'Border','tight');
title(['FFNN. OA: ' num2str(OA3*100,4) ' %'])
xlabel('Proposed');

subplot(246);
imshow(label2color(classmap4,lower('pavia')),'Border','tight');
title(['SVM. OA: ' num2str(OA4*100,4) ' %'])
xlabel('Spectral signatures');

subplot(247);
imshow(label2color(classmap5,lower('pavia')),'Border','tight');
title(['RF. OA: ' num2str(OA5*100,4) ' %'])
xlabel('Spectral signatures');

subplot(248);
imshow(label2color(classmap6,lower('pavia')),'Border','tight');
title(['FFNN. OA: ' num2str(OA6*100,4) ' %'])
xlabel('Spectral signatures');