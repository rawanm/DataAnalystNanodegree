# Identify Fraud from Enron Email

In this project, you will play detective, and put your machine learning skills to use by building an algorithm to identify Enron Employees who may have committed fraud based on the public Enron financial and email dataset.

In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, a significant amount of typically confidential information entered into the public record, including tens of thousands of emails and detailed financial data for top executives. In this project, you will play detective, and put your new skills to use by building a person of interest identifier based on financial and email data made public as a result of the Enron scandal. To assist you in your detective work, we've combined this data with a hand-generated list of persons of interest in the fraud case, which means individuals who were indicted, reached a settlement or plea deal with the government, or testified in exchange for prosecution immunity.


## Dataset: 
[Enron Data](https://github.com/rawanm/DataAnalystNanodegree/blob/master/P5_Identify_Fraud_from_Enron_Email/final_project_dataset.pkl): As preprocessing to this project, we've combined the Enron email and financial data into a dictionary, where each key-value pair in the dictionary corresponds to one person. The dictionary key is the person's name, and the value is another dictionary, which contains the names of all the features and their values for that person. The features in the data fall into three major types, namely financial features, email features and POI labels.

## Files: 
* [Free-Response Questions Report](https://github.com/rawanm/DataAnalystNanodegree/blob/master/P5_Identify_Fraud_from_Enron_Email/EnronSubmissionFree-ResponseQuestions.pdf): documents and answers the project questions and machine learning proccess to obtain the final classifier. 
* [poi_id.py](https://github.com/rawanm/DataAnalystNanodegree/blob/master/P5_Identify_Fraud_from_Enron_Email/poi_id.py): python code to test ML algorithems and generate final classifier.  
* [Dataset](https://github.com/rawanm/DataAnalystNanodegree/blob/master/P5_Identify_Fraud_from_Enron_Email/final_project_dataset.pkl)
* [Dataset and Decsription](https://github.com/rawanm/DataAnalystNanodegree/blob/master/P5_Identify_Fraud_from_Enron_Email/enron61702insiderpay.pdf)
* [poi_names](https://github.com/rawanm/DataAnalystNanodegree/blob/master/P5_Identify_Fraud_from_Enron_Email/poi_names.txt)
* [tester.py](https://github.com/rawanm/DataAnalystNanodegree/blob/master/P5_Identify_Fraud_from_Enron_Email/tester.py): script provided by Udacity to test classifer.
* [Final Dataset](https://github.com/rawanm/DataAnalystNanodegree/blob/master/P5_Identify_Fraud_from_Enron_Email/my_dataset.pkl)
* [Final Classifier](https://github.com/rawanm/DataAnalystNanodegree/blob/master/P5_Identify_Fraud_from_Enron_Email/my_classifier.pkl)
* [Final Features List](https://github.com/rawanm/DataAnalystNanodegree/blob/master/P5_Identify_Fraud_from_Enron_Email/my_feature_list.pkl)


