% Heart Failure Prediction with Machine Learning

data = readtable('Datasets\heart_failure_clinical_records_dataset.csv');

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

% Fixme: Handle categorical data
% fIXME

% data = categorical_data_to_dummy_variables(data, data.anaemia);
% data.anemia = [];

%-----------------------------Feature Scaling with Standardization

stand_platelets = (data.platelets - mean(data.platelets)) / std(data.platelets);
data.platelets = stand_platelets;

stand_serum_creatinine = (data.serum_creatinine - mean(data.serum_creatinine)) / std(data.serum_creatinine);
data.serum_creatinine = stand_serum_creatinine;

stand_serum_sodium = (data.serum_sodium - mean(data.serum_sodium)) / std(data.serum_sodium);
data.serum_sodium = stand_serum_sodium;

stand_sex = (data.sex - mean(data.sex)) / std(data.sex);
data.sex = stand_sex;

stand_smoking = (data.smoking - mean(data.smoking)) / std(data.smoking);
data.smoking = stand_smoking;

stand_time = (data.time - mean(data.time)) / std(data.time);
data.time = stand_time;

stand_age = (data.age - mean(data.age)) / std(data.age);
data.age = stand_age;

stand_anaemia = (data.anaemia - mean(data.anaemia)) / std(data.anaemia);
data.anaemia = stand_anaemia;

stand_creatinine_phosphokinase = (data.creatinine_phosphokinase - mean(data.creatinine_phosphokinase)) / std(data.creatinine_phosphokinase);
data.creatinine_phosphokinase = stand_creatinine_phosphokinase;

stand_ejection_fraction = (data.ejection_fraction - mean(data.ejection_fraction)) / std(data.ejection_fraction);
data.ejection_fraction = stand_ejection_fraction;

stand_anaemia = (data.anaemia - mean(data.anaemia)) / std(data.anaemia);
data.anaemia = stand_anaemia;


%----------------------------- Create classification model
classification_model = fitcknn(data, 'DEATH_EVENT~platelets+serum_creatinine+serum_sodium+sex+smoking+time+age+anaemia+creatinine_phosphokinase+diabetes+ejection_fraction+high_blood_pressure'); 

general_ratio = 0;
for a = 1:1
    %----------------------------- Partition data into training and test set
    cv = cvpartition(classification_model.NumObservations,'HoldOut', 0.2); %Built-in function for partitioning
    cross_validated_model = crossval(classification_model, 'cvpartition', cv); %Use training set only to built model 

    %----------------------------- Perform Predictions on test set
    Predictions = predict(cross_validated_model.Trained{1}, data(test(cv),1:end-1));

    %----------------------------- Confusion matris of results
    Results = confusionmat(cross_validated_model.Y(test(cv)),Predictions);
end


% Function to handle categorical data which does not have order relation:
    function data = categorical_data_to_dummy_variables(data,variable)
unique_values = unique(variable);
for i=1:length(unique_values)
    dummy_variable(:,i) = double(ismember(variable,unique_values{i})) ;
end 
T = table;
[rows, col] = size(dummy_variable);
for i=1:col
    T1 = table(dummy_variable(:,i));
    T1.Properties.VariableNames = unique_values(i);
    T = [T T1];
end 
    data = [T data]; 
    end
    