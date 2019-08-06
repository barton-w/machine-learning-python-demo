###################################################
#Loading various objects and functions which will be used to analyze the data
###################################################
import pandas #Data analysis library, used here to import data and create a scatter_matrix
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt #Library that creates visualization figures
from sklearn import model_selection #Library of various functions and algorithms
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
print("All the imports are done")

###################################################
#Loading the data using pandas
###################################################
url = "./iris_flowers.csv"
#print(url)
names = ["sepal-length", "sepal-width", "petal-length", "petal-width", "class"]
#print(names)
dataset = pandas.read_csv(url, names=names)
#print(dataset)

###################################################
#Various ways to look at the data, and visualize it
###################################################
# #Dimensions of the dataset
# print("Dataset dimensions: " + str(dataset.shape))
# #Top 10 rows
# print(dataset.head(10))
# #Statistical summary
# print(dataset.describe())
# #Counts grouped by class
# print(dataset.groupby("class").size())
# #Box plots
# dataset.plot(kind="box", subplots=True, layout=(2,2), sharex=False, sharey=False)
# plt.show()
# #Histograms
# dataset.hist()
# plt.show()
# #Scatter plot matrix
# scatter_matrix(dataset)
# plt.show()

###################################################
#Splitting the data randomly into a training dataset and a validation dataset
###################################################
#creating a list of sub-lists for each row, with all the values (not column names) from the dataset
array = dataset.values
#print(array)
#slicing each row to include only the sepal and petal measurements
iris_attributes = array[:,0:4]
# print("iris_attributes array:")
# print(iris_attributes)
#slicing each row to inlcude only the Iris class
iris_species = array[:,4]
# print("iris_species array:")
# print(iris_species)
##establishing variables to be used to randomly divide iris_attributes and iris_species
validation_size = 0.20
seed = 7
#Running the train_test_split fuction, and assigning its return to 4 variables, each representing parts of the full dataset
iris_attributes_train, iris_attributes_validate, iris_species_train, iris_species_validate = model_selection.train_test_split(iris_attributes, iris_species, test_size=validation_size, random_state=seed)
# print("iris_attributes_train array:")
# print(iris_attributes_train)
# print("iris_attributes_validate array:")
# print(iris_attributes_validate)
# print("iris_species_train array:")
# print(iris_species_train)
# print("iris_species_validate array:")
# print(iris_species_validate)

###################################################
#Evaluating the performance of various models
###################################################
# #Populate various ML models into a list. The models list will have a label for each and the function definiing each model
# models = []
# models.append(("LR", LogisticRegression(solver="liblinear", multi_class="ovr"))) #liblinear and ovr are default, but throws a warning if not explicitly supplied
# models.append(("LDA", LinearDiscriminantAnalysis()))
# models.append(("KNN", KNeighborsClassifier()))
# models.append(("CART", DecisionTreeClassifier()))
# models.append(("NB", GaussianNB()))
# models.append(("SVM", SVC(gamma="auto"))) #auto is the default, but throws a warning if not explicitly supplied
# #print(models)
#
# #Loop over the list, evaluating each model
# #Passed into the scoring argument below, this tells the cross_val_score function to compute the percent of correct predictions
# scoring = "accuracy"
# for name, model in models:
#     #sets the number of sub-datasets the training sets will be divided into and evaluated, and the random_state used into a variable
#     #Training sets are divided into 10 sub-sets, and on of those is held as "truth" to compare evaluations to
#     kfold = model_selection.KFold(n_splits=10, random_state=seed)
#     #Cross Validation - cross_val_score is running the model agaist each split, comparing results to the "truth" split
#     cv_results = model_selection.cross_val_score(model, iris_attributes_train, iris_species_train, cv=kfold, scoring=scoring)
#     #Create a final "message" with the name of each model, average of its results, and standard deviation
#     msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
#     #print(cv_results)
#     print(msg)

###################################################
#Running Support Vector Machine model on the validation dataset and analyzing final accuracy
###################################################
svm = SVC(gamma="auto")
#Fit method trains the model. The algorithm is finding the mathematically relationships between the data and outcomes
svm.fit(iris_attributes_train, iris_species_train)
#Predict method is running the model against the validation dataset, and having it predict the outcomes
predictions = svm.predict(iris_attributes_validate)
print(predictions)
#Calculating an overall percent score of correct predictions compared to the actual answers
print(accuracy_score(iris_species_validate, predictions))
#Generating a breakdown by class of how accurate the predictions were
print(confusion_matrix(iris_species_validate, predictions))
#actual species are horizontal
#predictions are vertical
print(classification_report(iris_species_validate, predictions))
#precision is vertical correctness- for the class predictions the model made, what percent were correct
#recall is horizontal correctness- were all of the actual species correctly predicted
#f score is the mean of the two
