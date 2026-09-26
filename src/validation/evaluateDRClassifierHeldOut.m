function evaluateDRClassifierHeldOut()

fprintf('\n=============================================\n');
fprintf('   SEEBEYOND DR CLASSIFIER VALIDATION\n');
fprintf('=============================================\n\n');

% Load held-out test set
load('models/aptosStage3Split.mat', 'testTable');

N = height(testTable);

trueGrades = double(testTable.Grade);
predGrades = nan(N,1);
confidences = nan(N,1);

fprintf('Test images: %d\n\n', N);

% Load classifier
fprintf('Loading DR classifier...\n');

S = load('models/drClassifier.mat');

if isfield(S,'drClassifier')
    net = S.drClassifier;
elseif isfield(S,'net')
    net = S.net;
else
    error('Could not find DR classifier variable in drClassifier.mat');
end

fprintf('Classifier loaded.\n\n');

% Predict each image
for i = 1:N

    imagePath = testTable.File{i};

    try

        I = imread(imagePath);

        % Correct function interface:
        % result = predictDRSeverity(I, net)
        result = predictDRSeverity(I, net);

        predGrades(i) = double(result.classIndex);
        confidences(i) = double(result.confidence);

    catch ME

        fprintf('ERROR image %d/%d: %s\n', ...
            i, N, ME.message);

    end

    if mod(i,25) == 0 || i == N

        fprintf('Processed %d / %d images (%.1f%%)\n', ...
            i, N, 100*i/N);

    end

end

% Remove failed predictions
valid = ~isnan(predGrades);

fprintf('\n=============================================\n');
fprintf('VALIDATION SUMMARY\n');
fprintf('=============================================\n');

fprintf('Total images      : %d\n', N);
fprintf('Successful        : %d\n', sum(valid));
fprintf('Failed            : %d\n', sum(~valid));

if any(valid)

    yTrue = trueGrades(valid);
    yPred = predGrades(valid);

    % Overall accuracy
    accuracy = mean(yTrue == yPred);

    fprintf('Accuracy          : %.2f%%\n', ...
        accuracy * 100);

    % Per-class results
    fprintf('\nPer-class results:\n');

    for g = 0:4

        idx = yTrue == g;

        if any(idx)

            classAccuracy = mean(yPred(idx) == g);

            fprintf( ...
                'Grade %d : %3d images | Correct %3d | Accuracy %.2f%%\n', ...
                g, ...
                sum(idx), ...
                sum(yPred(idx) == g), ...
                classAccuracy * 100);

        end

    end

    % Confusion matrix
    fprintf('\nConfusion Matrix:\n');
    fprintf('Rows = Actual, Columns = Predicted\n\n');

    C = confusionmat(yTrue, yPred, ...
        'Order', 0:4);

    fprintf('          Pred0 Pred1 Pred2 Pred3 Pred4\n');

    for r = 1:5

        fprintf( ...
            'Actual %d  %5d %5d %5d %5d %5d\n', ...
            r-1, ...
            C(r,1), ...
            C(r,2), ...
            C(r,3), ...
            C(r,4), ...
            C(r,5));

    end

    % Mean confidence
    meanConfidence = mean(confidences(valid));

    fprintf('\nMean confidence : %.2f%%\n', ...
        meanConfidence * 100);

else

    accuracy = NaN;
    C = [];
    meanConfidence = NaN;

end

% Save results
results.trueGrades = trueGrades;
results.predGrades = predGrades;
results.confidences = confidences;
results.accuracy = accuracy;
results.validCount = sum(valid);
results.failedCount = sum(~valid);
results.meanConfidence = meanConfidence;

if any(valid)
    results.confusionMatrix = C;
end

save( ...
    'models/DRClassifierHeldOutResults.mat', ...
    'results');

fprintf('\nResults saved to:\n');
fprintf('models/DRClassifierHeldOutResults.mat\n');

fprintf('\n=============================================\n');
fprintf('VALIDATION COMPLETE\n');
fprintf('=============================================\n');

end
