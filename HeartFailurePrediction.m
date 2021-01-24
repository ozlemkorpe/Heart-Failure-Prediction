%Prepared by �zlem K�rpe
%Github: https://github.com/ozlemkorpe/Heart-Failure-Prediction
%Note1: Please check out the data path before run, fix if necessary.
%Note1: Change the number of iterations for calculating average/general accuracy of prediction.

data = readtable('Dataset\heart_failure_clinical_records_dataset.csv');

% -----------------------------Column headers
% {'age'} : Age of the patient
% {'anaemia'} : Decrease of red blood cells or hemoglobin (boolean)
% {'creatinine_phosphokinase'} : Level of the CPK enzyme in the blood (mcg/L)
% {'diabetes'} : If the patient has diabetes (boolean)
% {'ejection_fraction'} : Percentage of blood leaving the heart at each contraction (percentage)
% {'high_blood_pressure'} : If the patient has hypertension (boolean)
% {'platelets'}: Platelets in the blood (kiloplatelets/mL)
% {'serum_creatinine'}: Level of serum creatinine in the blood (mg/dL)
% {'serum_sodium'} : Level of serum sodium in the blood (mEq/L)
% {'sex'} : Woman or man (binary)
% {'smoking'} : If the patient smokes or not (boolean)
% {'time'} : Follow-up period (days)
% {'DEATH_EVENT'} : If the patient deceased during the follow-up period (boolean)

%-----------------------------Handling Missing Values
%Check for the missing data
missing_rows = sum(ismissing(data)); %No missing data

%-----------------------------Feature Scaling with Standardization
% Age Scaling
stand_age = (data.age - mean(data.age)) / std(data.age);
data.age = stand_age;

% Stand Creatinine Phosphokinase Scaling
stand_creatinine_phosphokinase = (data.creatinine_phosphokinase - mean(data.creatinine_phosphokinase)) / std(data.creatinine_phosphokinase);
data.creatinine_phosphokinase = stand_creatinine_phosphokinase;

% Stand Ejection Fraction Scaling
stand_ejection_fraction = (data.ejection_fraction - mean(data.ejection_fraction)) / std(data.ejection_fraction);
data.ejection_fraction = stand_ejection_fraction;

% Platelets Scaling
stand_platelets = (data.platelets - mean(data.platelets)) / std(data.platelets);
data.platelets = stand_platelets;

% Stand Serum Creatinine Scaling
stand_serum_creatinine = (data.serum_creatinine - mean(data.serum_creatinine)) / std(data.serum_creatinine);
data.serum_creatinine = stand_serum_creatinine;

% Stand Serum Sodium Scaling
stand_serum_sodium = (data.serum_sodium - mean(data.serum_sodium)) / std(data.serum_sodium);
data.serum_sodium = stand_serum_sodium;

% Follow up time
stand_time = (data.time - mean(data.time)) / std(data.time);
data.time = stand_time;

%----------------------------- Classification and Prediction
classification_model = fitctree(data, 'DEATH_EVENT~platelets+serum_creatinine+serum_sodium+sex+smoking+time+age+anaemia+creatinine_phosphokinase+diabetes+ejection_fraction+high_blood_pressure'); 

general_accuracy = 0;
for a = 1:1000
    %----------------------------- Partition data into training and test set
    cv = cvpartition(classification_model.NumObservations,'HoldOut', 0.1); %Built-in function for partitioning
    cross_validated_model = crossval(classification_model, 'cvpartition', cv); %Use training set only to built model 

    %----------------------------- Perform Predictions on test set
    Predictions = predict(cross_validated_model.Trained{1}, data(test(cv),1:end-1));

    %----------------------------- Confusion matris of results
    Results = confusionmat(cross_validated_model.Y(test(cv)),Predictions);
    
    %----------------------------- Analyzing the Result
    Results = confusionmat(cross_validated_model.Y(test(cv)),Predictions); 
    right_results = Results(1,1) + Results(2,2);
    wrong_results = Results(1,2) + Results(2,1);

    truth_score = right_results /(right_results + wrong_results);
    general_accuracy = general_accuracy + truth_score;


end

