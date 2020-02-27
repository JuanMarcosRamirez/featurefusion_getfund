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
lambda2 = [1e-4 2e-4 5e-4];

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

fusedFeatPatt1    = feature_fusion_direct(shot_patt_HS,shot_patt_MS,...
    filter_pattHS,filter_pattMS,q,p,lambda1,lambda2(1),dictionary,alg_parameters);

fusedFeatPatt2    = feature_fusion_direct(shot_patt_HS,shot_patt_MS,...
    filter_pattHS,filter_pattMS,q,p,lambda1,lambda2(2),dictionary,alg_parameters);

fusedFeatPatt3    = feature_fusion_direct(shot_patt_HS,shot_patt_MS,...
    filter_pattHS,filter_pattMS,q,p,lambda1,lambda2(3),dictionary,alg_parameters);

%% Classification Stage
training_rate = 0.1;
[training_indexes, test_indexes] = classification_indexes(ground_truth, training_rate);
T_classes = ground_truth(training_indexes);

test_indexes2 = 1:length(ground_truth(:));
test_indexes2(training_indexes) = [];

fusedFeatPatt1_train = fusedFeatPatt1(:,training_indexes);
fusedFeatPatt1_test  = fusedFeatPatt1(:,test_indexes);
fusedFeatPatt1_test2 = fusedFeatPatt1(:,test_indexes2);

fusedFeatPatt2_train = fusedFeatPatt2(:,training_indexes);
fusedFeatPatt2_test  = fusedFeatPatt2(:,test_indexes);
fusedFeatPatt2_test2 = fusedFeatPatt2(:,test_indexes2);

fusedFeatPatt3_train = fusedFeatPatt3(:,training_indexes);
fusedFeatPatt3_test  = fusedFeatPatt3(:,test_indexes);
fusedFeatPatt3_test2 = fusedFeatPatt3(:,test_indexes2);

disp(['Predicting classes...']);
num_classes = max(max(ground_truth));
T = zeros(num_classes, length(training_indexes));
for i = 1: length(training_indexes)
    T(ground_truth(training_indexes(i)),i)= 1;
end
net = feedforwardnet(20);
net.trainParam.showWindow = false;

net1 = train(net,fusedFeatPatt1_train,T);
class_hat1 = vec2ind(net1(fusedFeatPatt1_test))';
class_hat4 = vec2ind(net1(fusedFeatPatt1_test2))';

net2 = train(net,fusedFeatPatt2_train,T);
class_hat2 = vec2ind(net2(fusedFeatPatt2_test))';
class_hat5 = vec2ind(net1(fusedFeatPatt2_test2))';

net3 = train(net,fusedFeatPatt3_train,T);
class_hat3 = vec2ind(net3(fusedFeatPatt3_test))';
class_hat6 = vec2ind(net1(fusedFeatPatt3_test2))';

%% Building the classification maps
training_set_image = zeros(size(ground_truth,1), size(ground_truth,2));
training_set_image(training_indexes) = ground_truth(training_indexes);

classmap1    = class_map_image(ground_truth, class_hat1, training_indexes, test_indexes);
classmap2    = class_map_image(ground_truth, class_hat2, training_indexes, test_indexes);
classmap3    = class_map_image(ground_truth, class_hat3, training_indexes, test_indexes);
classmap4    = class_map_image(ground_truth, class_hat4, training_indexes, test_indexes2);
classmap5    = class_map_image(ground_truth, class_hat5, training_indexes, test_indexes2);
classmap6    = class_map_image(ground_truth, class_hat6, training_indexes, test_indexes2);

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
xlabel(['\lambda_2= ' num2str(lambda2(1))]);
title(['OA: ' num2str(OA1*100,4) ' %'])

subplot(243);
imshow(label2color(classmap2,lower('pavia')),'Border','tight');
xlabel(['\lambda_2= ' num2str(lambda2(2))]);
title(['OA: ' num2str(OA2*100,4) ' %'])

subplot(244);
imshow(label2color(classmap3,lower('pavia')),'Border','tight');
xlabel(['\lambda_2= ' num2str(lambda2(3))]);
title(['OA: ' num2str(OA3*100,4) ' %'])

subplot(245);
temp_show=Io(:,:,band_set_hs1);temp_show=normColor(temp_show);
imshow(temp_show,[]);
title('Input image')

subplot(246);
imshow(label2color(classmap4,lower('pavia')),'Border','tight');
xlabel(['\lambda_2= ' num2str(lambda2(1))]);

subplot(247);
imshow(label2color(classmap5,lower('pavia')),'Border','tight');
xlabel(['\lambda_2= ' num2str(lambda2(2))]);

subplot(248);
imshow(label2color(classmap6,lower('pavia')),'Border','tight');
xlabel(['\lambda_2= ' num2str(lambda2(3))]);