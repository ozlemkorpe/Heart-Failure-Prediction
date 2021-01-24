# Heart-Failure-Prediction

Cardiovascular diseases (CVDs) are the number 1 cause of death globally, taking an estimated17.9 million lives each year, which accounts for 31. Heart failure is a common event caused by CVDs and this dataset contains 12 features that can be used to predict mortality by heart failure. Most cardiovascular diseases can be prevented by addressing behavioural risk factors such as tobacco use, unhealthy diet and obesity, physical inactivity and harmful use of alcohol using population-wide strategies. People with cardiovascular disease or who are at high cardiovascular risk (due to the presence of one or more risk factors such as hypertension, diabetes, hyper lipidaemia or al-ready established disease) need early detection and management where in a machine learning model can be of great help

## Dataset
Dataset is obtained by Kaggle, Davide Chicco, Giuseppe Jurman: Machine learning canpredict survival of patients with heart failure from serum creatinine and ejection fractionalone. BMC Medical Informatics and Decision Making 20, 16 (2020).

### Datafields
- age: Age of the patient
- anaemia: Decrease of red blood cells or hemoglobin (boolean) creatinine_phosphokinase: Level of the CPK enzyme in the blood (mcg/L)
- diabetes: If the patient has diabetes (boolean)
- ejection_fraction: Percentage of blood leaving the heart at each contraction (percentage)
- high_blood_pressure: If the patient has hypertension (boolean)
- platelets: Platelets in the blood (kiloplatelets/mL)
- serum_creatinine: Level of serum creatinine in the blood (mg/dL)
- serum_sodium: Level of serum sodium in the blood (mEq/L)
- sex: Woman or man (binary)
- smoking: If the patient smokes or not (boolean)
- time: Follow-up period (days)
- DEATH_EVENT: If the patient deceased during the follow-up period (boolean)

## Methodology
### Preprocessing
By using following line I checked for the number of missing values in each column and verified that there is no missing values in the whole dataset

```
missing_rows = sum(ismissing(data)); %No missing data
```

All features have different ranges and if I use them like that they will affect the computer with different ratios. For solving this problem, I used two different scaling method which are standardization and normalization. Later on I saw that normalization method produces slightly better output than standardization. Performed normalization on each variable in training and test sets and saved the results in normalized datatables.

```
stand_age = (data.age - mean(data.age)) / std(data.age);
data.age = stand_age
```

### Partitioning and Classification

Partitioned the data into the training and the test sets. Created a variable named TestPartition at the beginning of the script to easily change the percentage of the test set and set it to 0.1 initially which means 10% of the dataset will be used for testing and 90% of it is used for training. 

```
cv=cvpartition(classification_model.NumObservations,'HoldOut',TestPartition);
cross_validated_model = crossval(classification_model, 'cvpartition', cv);
```
After performing partition I performed the prediction by using the model I created. I used confusion matrix to see how the results are distributed.

```
classification_model = fitctree(data, 'DEATH_EVENT~platelets+serum_creatinine+time+serum_sodium+sex+smoking+age+anaemia+creatinine_phosphokinase+diabetes+ejection_fraction+high_blood_pressure','MinParentSize',DecisionAttr ); 
```

### Experiments and Result
Used confusion matrix to calculate accuracy and analyze score.
```
    Results = confusionmat(cross_validated_model.Y(test(cv)),Predictions); 
    right_results = Results(1,1) + Results(2,2);
    wrong_results = Results(1,2) + Results(2,1);

    truth_score = right_results /(right_results + wrong_results);
    general_accuracy = general_accuracy + truth_score;
```

In first experiments I used all of the attributes provided in the dataset by using different customizations of decision tree algorithm like setting maximum number of splits or minimum size of the parent. During these experiments I obtained the best results by decreasing the number maximum splits or increasing the size of the minimum parents. When I analyzed the decision tree plot, I saw that in these customization algorithm decides only by checking at the follow-up period time. Even If provides the best resulted predictions approximately 82%, I wanted to analyze the effect of other attributes excluding the time. With the time attribute, accuracy of  trained model is up to 0.83034%, in the other hand if the model is trained without time attribute has up to 0.74586%. If the time attribute is included into calculations, it has the most effect on the result. If excluded serum creatinine has the most effect as can be seen on the following diagram.
