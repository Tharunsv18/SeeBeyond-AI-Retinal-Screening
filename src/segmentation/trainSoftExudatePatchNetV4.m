%% SeeBeyond - Soft Exudate Patch U-Net V2
% Improved Soft Exudate segmentation

clearvars;
clc;

cd('/home/tharunkumar-s-v/Documents/SeeBeyond');
addpath(genpath(fullfile(pwd,'src')));

rng(42);

fprintf('\n========================================\n');
fprintf('SOFT EXUDATE PATCH U-NET V3\n');
fprintf('========================================\n');

%% Load split

load('models/softExudateV2Split.mat', ...
    'trainSEIdx','valSEIdx', ...
    'positiveImageCount', ...
    'positiveCount', ...
    'hardExNegCount', ...
    'opticDiscNegCount', ...
    'normalNegCount');

%% Paths

patchRoot = fullfile(pwd, ...
    'datasets','IDRiD','patches','soft_exudates_v4');

imageDir = fullfile(patchRoot,'images');
maskDir  = fullfile(patchRoot,'masks');

%% Datastores

imdsSEV4 = imageDatastore(imageDir);

classNamesSEV4 = [
    "background"
    "soft_exudate"
];

pxdsSEV4 = pixelLabelDatastore( ...
    maskDir, ...
    classNamesSEV4, ...
    [0 1], ...
    FileExtensions=".png");

fprintf('Images : %d\n',numel(imdsSEV4.Files));
fprintf('Masks  : %d\n',numel(pxdsSEV4.Files));

%% Pixel statistics

countsSEV4 = countEachLabel(pxdsSEV4);

fprintf('\n===== PIXEL DISTRIBUTION =====\n');
disp(countsSEV4)

freqSEV2 = ...
    countsSEV4.PixelCount ./ ...
    sum(countsSEV4.PixelCount);

fprintf('Background frequency   : %.6f\n',freqSEV2(1));
fprintf('Soft Exudate frequency : %.6f\n',freqSEV2(2));

%% Combine + augmentation

dsSEV2 = combine( ...
    imdsSEV4, ...
    pxdsSEV4);

dsSEV2 = transform( ...
    dsSEV2, ...
    @augmentSoftExudateDataV3);

%% Verify sample

sample = preview(dsSEV2);

fprintf('\nImage size:\n');
disp(size(sample{1}));

fprintf('Mask size:\n');
disp(size(sample{2}));

fprintf('Undefined pixels : %d\n', ...
    sum(isundefined(sample{2}(:))));

%% U-Net

netSEV2 = unet( ...
    [256 256 3], ...
    2, ...
    EncoderDepth=3, ...
    NumFirstEncoderFilters=16);

%% Training options

optionsSEV2 = trainingOptions("adam", ...
    InitialLearnRate=5e-5, ...
    MaxEpochs=30, ...
    MiniBatchSize=4, ...
    Shuffle="every-epoch", ...
    ExecutionEnvironment="gpu", ...
    Plots="training-progress", ...
    Verbose=true);

%% Train

fprintf('\nStarting Soft Exudate V3 training...\n');

softExudatePatchNetV4 = trainnet( ...
    dsSEV2, ...
    netSEV2, ...
    @softExudateV2Loss, ...
    optionsSEV2);

%% Save separately

save('models/softExudatePatchNetV4.mat', ...
    'softExudatePatchNetV4', ...
    'freqSEV2', ...
    'trainSEIdx', ...
    'valSEIdx');

fprintf('\n========================================\n');
fprintf('SOFT EXUDATE V4 TRAINING COMPLETE\n');
fprintf('========================================\n');

fprintf('Saved: models/softExudatePatchNetV4.mat\n');
