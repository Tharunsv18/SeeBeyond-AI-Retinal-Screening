function createSoftExudateHardNegativesV4()
% SeeBeyond - V4 hard-negative mining for soft exudates
%
% Uses V3 predictions on TRAINING images only.
% High-confidence false-positive regions are converted into
% negative training patches.
%
% IMPORTANT:
% Validation images are never used here.

fprintf('\n========================================\n');
fprintf('SOFT EXUDATE V4 HARD-NEGATIVE MINING\n');
fprintf('========================================\n');

%% Paths
load('models/softExudateV2Split.mat','trainSEIdx','valSEIdx');

imageRoot = fullfile(pwd,...
    'datasets','IDRiD','Segmentation','A. Segmentation',...
    '1. Original Images','a. Training Set');

softMaskRoot = fullfile(pwd,...
    'datasets','IDRiD','processed_masks','soft_exudates');

outRoot = fullfile(pwd,...
    'datasets','IDRiD','patches','soft_exudates_v4_hardneg');

imageOut = fullfile(outRoot,'images');
maskOut  = fullfile(outRoot,'masks');

if exist(outRoot,'dir')
    rmdir(outRoot,'s');
end

mkdir(imageOut);
mkdir(maskOut);

%% Load V3 network
S = load('models/softExudatePatchNetV3.mat',...
    'softExudatePatchNetV3');

net = S.softExudatePatchNetV3;

%% Settings
patchSize = 256;
stride = 128;

% Mine predictions that V3 considers strongly positive.
mineThreshold = 0.70;

% Number of hard negatives per training image.
hardNegPerImage = 40;

fprintf('Training images       : %d\n',numel(trainSEIdx));
fprintf('Validation images     : %d\n',numel(valSEIdx));
fprintf('Mining threshold      : %.2f\n',mineThreshold);
fprintf('Hard negatives/image  : %d\n',hardNegPerImage);

%% Verify split
if ~isempty(intersect(trainSEIdx,valSEIdx))
    error('TRAIN/VALIDATION SPLIT OVERLAP DETECTED');
end

%% Output counter
counter = 0;

%% Process TRAINING images only
for n = 1:numel(trainSEIdx)

    idx = trainSEIdx(n);
    imageID = sprintf('%02d',idx);

    imagePath = fullfile(imageRoot,...
        sprintf('IDRiD_%s.jpg',imageID));

    maskPath = fullfile(softMaskRoot,...
        sprintf('IDRiD_%s.png',imageID));

    if ~exist(imagePath,'file') || ~exist(maskPath,'file')
        warning('Skipping IDRiD_%s',imageID);
        continue;
    end

    I = imread(imagePath);
    GT = imread(maskPath) > 0;

    [h,w,~] = size(I);

    fprintf('\n[%d/%d] IDRiD_%s\n',...
        n,numel(trainSEIdx),imageID);

    %% V3 probability map
    scoreMap = zeros(h,w,'single');
    countMap = zeros(h,w,'single');

    rowStarts = 1:stride:(h-patchSize+1);
    colStarts = 1:stride:(w-patchSize+1);

    if rowStarts(end) ~= h-patchSize+1
        rowStarts(end+1) = h-patchSize+1;
    end

    if colStarts(end) ~= w-patchSize+1
        colStarts(end+1) = w-patchSize+1;
    end

    for r = rowStarts
        for c = colStarts

            P = I(r:r+patchSize-1,...
                  c:c+patchSize-1,:);

            X = reshape(P,patchSize,patchSize,3,1);

            scores = minibatchpredict(...
                net,X,...
                MiniBatchSize=1,...
                ExecutionEnvironment="gpu");

            if isa(scores,'dlarray')
                scores = extractdata(scores);
            end

            if ndims(scores) == 4
                scores = scores(:,:,:,1);
            end

            scoreMap(r:r+patchSize-1,...
                     c:c+patchSize-1) = ...
                scoreMap(r:r+patchSize-1,...
                         c:c+patchSize-1) + ...
                single(scores(:,:,2));

            countMap(r:r+patchSize-1,...
                     c:c+patchSize-1) = ...
                countMap(r:r+patchSize-1,...
                         c:c+patchSize-1) + 1;
        end
    end

    probMap = scoreMap ./ max(countMap,1);

    %% Candidate false positives
    %
    % V3 says positive, but ground truth says negative.
    candidate = probMap >= mineThreshold;
    candidate = candidate & ~GT;

    %% Remove very small candidate regions
    candidate = bwareaopen(candidate,20);

    [rr,cc] = find(candidate);

    if isempty(rr)
        fprintf('  No hard negatives found.\n');
        continue;
    end

    %% Randomize candidate locations
    order = randperm(numel(rr));

    created = 0;

    for q = 1:numel(order)

        if created >= hardNegPerImage
            break;
        end

        centerR = rr(order(q));
        centerC = cc(order(q));

        r1 = round(centerR - patchSize/2);
        c1 = round(centerC - patchSize/2);

        r1 = max(1,min(r1,h-patchSize+1));
        c1 = max(1,min(c1,w-patchSize+1));

        rows = r1:r1+patchSize-1;
        cols = c1:c1+patchSize-1;

        gtPatch = GT(rows,cols);

        % Reject patches containing real soft exudate.
        if nnz(gtPatch) > 0
            continue;
        end

        % Make sure the patch actually contains a strong V3 FP.
        probPatch = probMap(rows,cols);

        if nnz(probPatch >= mineThreshold) < 20
            continue;
        end

        patch = I(rows,cols,:);

        counter = counter + 1;
        created = created + 1;

        imageName = sprintf('v4_hardneg_%06d.png',counter);
        maskName  = sprintf('v4_hardneg_%06d.png',counter);

        imwrite(patch,fullfile(imageOut,imageName));

        zeroMask = false(patchSize,patchSize);
        imwrite(zeroMask,fullfile(maskOut,maskName));
    end

    fprintf('  Candidate pixels : %d\n',nnz(candidate));
    fprintf('  Hard negatives   : %d\n',created);
end

fprintf('\n========================================\n');
fprintf('V4 HARD-NEGATIVE MINING COMPLETE\n');
fprintf('========================================\n');
fprintf('Total hard negatives: %d\n',counter);
fprintf('Output: %s\n',outRoot);
fprintf('========================================\n');

end
