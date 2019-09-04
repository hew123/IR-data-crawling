import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels
import time
from pycm import *
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm,tree
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier as vc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,classification_report,confusion_matrix,make_scorer
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier,AdaBoostClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier


np.random.seed(500)
Corpus = pd.read_csv(r"labelled_data.csv",encoding='latin-1')
test_data = pd.read_csv(r"reviews_uncleaned.csv",encoding='latin-1')

pred_list = []


Tfidf_vect = TfidfVectorizer(max_features=5000,lowercase = True)

vectorizer = CountVectorizer(min_df = 0,lowercase = True,stop_words = 'english',ngram_range = (2,3))



Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(Corpus['Reviews'],Corpus['label'],test_size = 0)

Tfidf_vect.fit(Corpus['Reviews'])

Train_X_Count = Tfidf_vect.transform(Train_X)






print("----------------------------------------Model Training and fitting---------------------------------------------------")

class_names = ['-1', '0', '1']

#confusion matrix function
def plot_confusion_matrix(y_true, y_pred,classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
  


    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           title=title,
		   xticklabels=classes, yticklabels=classes,
		    xlabel='Predicted label',
           ylabel='True label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=0, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

	
	
#---------------------------------------------------------------------------------------------------------------------------	
Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(Corpus['Reviews'],Corpus['label'],test_size=0.3)

Tfidf_vect.fit(Corpus['Reviews'])
Train_X_Count = Tfidf_vect.transform(Train_X)

Test_X_Count = Tfidf_vect.transform(Test_X)


data_Count = Tfidf_vect.transform(test_data['Reviews'])

target_names = ['class -1' ,'class 0','class 1']

#NB
Naive = naive_bayes.MultinomialNB()

start = time.time()
Naive.fit(Train_X_Count,Train_Y)
end = time.time()
ms = float(round((end - start)*1000))
print("The time taken for Naive Bayes is ", ms)

start = time.time()
predictions_NB_valid = Naive.predict(Test_X_Count)
end = time.time()
ms = float(round((end - start)*1000))
print("The time taken for Naive Bayes is ", ms)


print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB_valid, Test_Y)*100)

print(cross_val_score(Naive,Test_X_Count,Test_Y,cv = 10))

print(classification_report(Test_Y,predictions_NB_valid,target_names = target_names))
#print(confusion_matrix(Test_Y,predictions_NB_valid))


#LR
classifier = LogisticRegression()
classifier.fit(Train_X_Count,Train_Y)

start = time.time()
classifier.fit(Train_X_Count,Train_Y)
end = time.time()
ms = float(round((end - start)*1000))
print("The time taken for LR is ", ms)




start = time.time()

predictions_LR_valid = classifier.predict(Test_X_Count)

end = time.time()
s = end - start
print("The time taken for LR is ", s)




print("Logistic Regression Accuracy Score -> ",accuracy_score(predictions_LR_valid, Test_Y)*100)

print(cross_val_score(classifier,Test_X_Count,Test_Y,cv = 10))

print(classification_report(Test_Y,predictions_LR_valid,target_names = target_names))

print(confusion_matrix(Test_Y,predictions_LR_valid))

# predictions_LR = classifier.predict(data_Count)
# test_data['label'] = predictions_LR

# test_data.to_csv("temp.csv",index = False)




#DT
clf_tree = tree.DecisionTreeClassifier()
clf_tree = clf_tree.fit(Train_X_Count, Train_Y)


start = time.time()
clf_tree = clf_tree.fit(Train_X_Count, Train_Y)
end = time.time()
ms = float(round((end - start)*1000))
print("The time taken for DT is ", ms)

start = time.time()

predictions_DT_valid = clf_tree.predict(Test_X_Count) 
end = time.time()
s = end - start

print("The time taken for DT is ", s)
print("DecisionTreeClassifier Score -> ",accuracy_score(predictions_DT_valid, Test_Y)*100)

print(cross_val_score(clf_tree,Test_X_Count,Test_Y,cv = 10))

print(classification_report(Test_Y,predictions_DT_valid,target_names = target_names))





#Knn
neigh = KNeighborsClassifier(n_neighbors = 5)
neigh.fit(Train_X_Count, Train_Y)

start = time.time()
neigh.fit(Train_X_Count, Train_Y)
end = time.time()
ms = float(round((end - start)*1000))
print("The time taken for Knn is ", ms)

start = time.time()

predictions_KNN_valid = neigh.predict(Test_X_Count) 

end = time.time()
s = end - start
print("The time taken for KNN is ", s)


print("KNN Score -> ",accuracy_score(predictions_KNN_valid, Test_Y)*100)
print(classification_report(Test_Y,predictions_KNN_valid,target_names = target_names))

print(cross_val_score(neigh,Test_X_Count,Test_Y,cv = 10))

#RF
rf_clf = RandomForestClassifier(n_estimators = 100,max_depth=5,random_state = 2)

rf_clf.fit(Train_X_Count, Train_Y)


start = time.time()
rf_clf.fit(Train_X_Count, Train_Y)
end = time.time()
ms = float(round((end - start)*1000))
print("The time taken for RF is ", ms)

start = time.time()

rf_predictions = rf_clf.predict(Test_X_Count)

end = time.time()
s = end - start
print("The time taken for RF is ", s)



print("Random Forest Score -> ",accuracy_score(rf_predictions, Test_Y)*100)
print(classification_report(Test_Y,rf_predictions,target_names = target_names))

print(cross_val_score(rf_clf,Test_X_Count,Test_Y,cv = 10))

plot_confusion_matrix(Test_Y, rf_predictions,class_names, title='Confusion matrix, without normalization')
plt.show()

#ANN

clf = MLPClassifier(solver  = 'lbfgs',alpha = 1e-5,hidden_layer_sizes = (5,2),random_state = 1)
clf.fit(Train_X_Count, Train_Y)
ann_predictions_valid = clf.predict(Test_X_Count)

print("Ann Score -> ",accuracy_score(ann_predictions_valid, Test_Y)*100)
print(classification_report(Test_Y,ann_predictions_valid,target_names = target_names))



#svc

# svc_clf = svm.SVC(gamma='scale')

# svc_clf.fit(Train_X_Count, Train_Y)
# svc_predictions = svc_clf.predict(Test_X_Count)


# print("Linear support vector classifier  machine Score -> ",accuracy_score(svc_predictions, Test_Y)*100)
# print(classification_report(Test_Y,svc_predictions,target_names = target_names))


#plot_confusion_matrix(Test_Y, svc_predictions,class_names, title='Confusion matrix, without normalization')
#plt.show()




#voting
# estimators = [('LR',classifier),('NB',Naive),('DT',clf),('KNN',neigh)]
# ensemble = vc(estimators,voting = 'hard')

# ensemble.fit(Train_X_Count, Train_Y)
# en_predict_valid = ensemble.predict(Test_X_Count)


# print("Ensemble Score -> ",accuracy_score(en_predict_valid, Test_Y)*100)
# print(classification_report(Test_Y,en_predict_valid,target_names = target_names))

# #Start timer
# start = time.time()



# #End Timer
# end = time.time()
# s = end - start
# print("The time taken for Ensemble is ", s)





# plot_confusion_matrix(Test_Y, en_predict_valid,class_names, title='Confusion matrix, without normalization')
# plt.show()

#boosting
ada = AdaBoostClassifier(n_estimators = 100,base_estimator = rf_clf,learning_rate = 1.0)
ada.fit(Train_X_Count, Train_Y)
ada_predictions_valid = ada.predict(Test_X_Count)

print("Ada Score -> ",accuracy_score(ada_predictions_valid, Test_Y)*100)
print(classification_report(Test_Y,ada_predictions_valid,target_names = target_names))

plot_confusion_matrix(Test_Y, ada_predictions_valid,class_names, title='Confusion matrix, without normalization')
plt.show()

#bagging
bag = BaggingClassifier(n_estimators = 100,base_estimator = clf,max_samples = 0.5,max_features = 1.0)
bag.fit(Train_X_Count, Train_Y)
bag_predictions_valid = bag.predict(Test_X_Count)

print("Bagging Score -> ",accuracy_score(bag_predictions_valid, Test_Y)*100)
print(classification_report(Test_Y,bag_predictions_valid,target_names = target_names))


plot_confusion_matrix(Test_Y, bag_predictions_valid,class_names, title='Confusion matrix, without normalization')
plt.show()

