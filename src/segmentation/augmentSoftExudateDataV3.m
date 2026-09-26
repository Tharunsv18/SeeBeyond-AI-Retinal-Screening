function dataOut = augmentSoftExudateDataV3(data)
% SeeBeyond - Soft Exudate V3 augmentation
%
% Spatial + controlled photometric augmentation.
% The segmentation mask is NEVER photometrically modified.

I = data{1};
C = data{2};

%% Horizontal flip
if rand > 0.5
    I = fliplr(I);
    C = fliplr(C);
end

%% Convert image to double for controlled augmentation
Id = double(I);

%% Random brightness
brightnessFactor = 0.80 + 0.40 * rand;
Id = Id * brightnessFactor;

%% Random contrast around image mean
contrastFactor = 0.85 + 0.30 * rand;
mu = mean(Id,[1 2]);
Id = (Id - mu) * contrastFactor + mu;

%% Mild gamma augmentation
gammaValue = 0.90 + 0.20 * rand;

% Normalize each channel to [0,1]
Id = max(0,min(255,Id));
Idn = Id / 255;

Idn = Idn .^ gammaValue;

Id = Idn * 255;

%% Mild independent RGB scaling
rgbScale = [
    0.95 + 0.10*rand, ...
    0.95 + 0.10*rand, ...
    0.95 + 0.10*rand
];

for k = 1:3
    Id(:,:,k) = Id(:,:,k) * rgbScale(k);
end

%% Clip and restore original image type
Id = max(0,min(255,Id));
I = uint8(round(Id));

dataOut = {I,C};

end
