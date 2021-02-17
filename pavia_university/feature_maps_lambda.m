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
lambda2 = [0 1e-4 2e-4 5e-4];

% Loading data
load('../data/Pavia_university_cropped.mat');
Io = pavia_university_cropped;
clear pavia_university_cropped;

load('../data/Pavia_university_gt.mat');
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
    filter_pattHS,filter_pattMS,q,p,lambda1,lambda2(1),dictionary,alg_parameters)';
fusedFeatPattImg1 = reshape(fusedFeatPatt1, [N1 N2 num_hs_filters]);

fusedFeatPatt2    = feature_fusion_direct(shot_patt_HS,shot_patt_MS,...
    filter_pattHS,filter_pattMS,q,p,lambda1,lambda2(2),dictionary,alg_parameters)';
fusedFeatPattImg2 = reshape(fusedFeatPatt2, [N1 N2 num_hs_filters]);

fusedFeatPatt3    = feature_fusion_direct(shot_patt_HS,shot_patt_MS,...
    filter_pattHS,filter_pattMS,q,p,lambda1,lambda2(3),dictionary,alg_parameters)';
fusedFeatPattImg3 = reshape(fusedFeatPatt3, [N1 N2 num_hs_filters]);

fusedFeatPatt4    = feature_fusion_direct(shot_patt_HS,shot_patt_MS,...
    filter_pattHS,filter_pattMS,q,p,lambda1,lambda2(4),dictionary,alg_parameters)';
fusedFeatPattImg4 = reshape(fusedFeatPatt4, [N1 N2 num_hs_filters]);
%% Display results
band_set_ms1=[floor(2*L_MS/3) floor(L_MS/4) 1];
band_set_hs1=[floor(2*L_HS/3) floor(L_HS/4) 1];

normColor = @(R)max(min((R-mean(R(:)))/std(R(:)),2),-2)/3+0.5;

I1(:,:,1) = mat2gray(I_MS(:,:,band_set_ms1(1)));
I1(:,:,2) = mat2gray(I_MS(:,:,band_set_ms1(2)));
I1(:,:,3) = mat2gray(I_MS(:,:,band_set_ms1(3)));

% Subplot figures
subplot(241);
imshow(label2color(ground_truth,lower('pavia')),'Border','tight');
title('Ground truth');

subplot(242);
temp_show=Io(:,:,band_set_hs1);temp_show=normColor(temp_show);
imshow(temp_show,[]);
title('HR image');

subplot(243);
imshow(I1,[]);
title('MS image');

subplot(244);
temp_show=I_HS(:,:,band_set_hs1);temp_show=normColor(temp_show);
imshow(temp_show,[]);
title('HS image');

subplot(245);
imshow(imadjust(mat2gray(fusedFeatPattImg1(:,:,1))),[]);
xlabel(['\lambda_2= ' num2str(lambda2(1))]);

subplot(246);
imshow(imadjust(mat2gray(fusedFeatPattImg2(:,:,1))),[]);
xlabel(['\lambda_2= ' num2str(lambda2(2))]);

subplot(247);
imshow(imadjust(mat2gray(fusedFeatPattImg3(:,:,1))),[]);
xlabel(['\lambda_2= ' num2str(lambda2(3))]);

subplot(248);
imshow(imadjust(mat2gray(fusedFeatPattImg4(:,:,1))),[]);
xlabel(['\lambda_2= ' num2str(lambda2(4))]);