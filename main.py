import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay






df = pd.read_csv(r'C:\Users\eliff\Desktop\VERİ BİLİMİ PROJE\Iris.csv')


df.columns = ['Id','sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
plantface_areacm2 = df['sepal_width']*df['sepal_length']+df['petal_width']*df['petal_length']
maxval = 10
minval = 4



df['living_temp'] =  minval+plantface_areacm2/5

for row in df['species']:
    if row == 'Iris-setosa':
        df['living_temp']+1


for row in df['species']:
    if row == 'Iris-virginica':
        df['living_temp']-2

for row in df['living_temp']:
    a= df['living_temp'].where(df.living_temp < 10, inplace=True)

df = df.replace(np.nan, 10)

#su ihtiyacının belirlenmesi

evopration_rate = df['living_temp']*0.1 #buharlaşma katsayısı sıcaklığa göre belirleniyor
                                        # ve her 1 c sıcaklık artışı buharlaşma hızını 0.1 artırıyor

df['water_need_ml'] = (plantface_areacm2+ evopration_rate)*10

df = df.replace('Iris-setosa', 1)
df = df.replace('Iris-versicolor', 2)
df = df.replace('Iris-virginica', 3)

#grafikler



sn.lmplot( x="sepal_length", y="living_temp", data=df, fit_reg=False, hue='species', legend=False,palette="rocket")
plt.legend(loc='lower right')
sn.lmplot( x="petal_length", y="water_need_ml", data=df, fit_reg=False, hue='species', legend=False,palette="rocket")
plt.legend(loc='lower right')
sn.lmplot(x="living_temp", y="water_need_ml", data=df, hue='species', legend=False,palette="rocket")
plt.legend(loc='lower right')
plt.show()






X = np.array(df.iloc[:,1:9])
Y = np.array([[df.species]])
Y = Y.reshape(150)




#değerleri train data ve test data olarak ayırıyoruz
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.5, random_state = 0)


seed = 20
scoring = 'accuracy'

modeller = []
modeller.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
modeller.append(('LDA', LinearDiscriminantAnalysis()))
modeller.append(('KNN', KNeighborsClassifier()))
modeller.append(('CART', DecisionTreeClassifier()))
modeller.append(('NB', GaussianNB()))
modeller.append(('SVM', SVC(gamma='auto')))
# her turnde modeli evluate edecek.
results = []
names = []
resultsgraph = []
for name, model in modeller:
	kfold = model_selection.KFold(n_splits=10, random_state=seed,shuffle = True)
	cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	resultsgraph.append(cv_results.mean())

	print(msg)

#en tutarlı ve düzgün LDA olduğu için confusion matrix onun için oluşturuldu
model = LinearDiscriminantAnalysis()
a = model.fit(X_train, y_train)

tahmin = model.predict(X_test)

y_pred = tahmin

cm = confusion_matrix(y_test, y_pred)
###############################################
model = LogisticRegression(random_state=0)
b = model.fit(X_train, y_train)

tahmin1 = model.predict(X_test)

y_pred1 = tahmin1

cm1 = confusion_matrix(y_test, y_pred1)
###############################################
model = KNeighborsClassifier()
c = model.fit(X_train, y_train)

tahmin2 = model.predict(X_test)

y_pred2 = tahmin2

cm2 = confusion_matrix(y_test, y_pred2)

###############################################
model = DecisionTreeClassifier()
d = model.fit(X_train, y_train)

tahmin3 = model.predict(X_test)

y_pred3 = tahmin3

cm3 = confusion_matrix(y_test, y_pred3)

###############################################
model = GaussianNB()
e = model.fit(X_train, y_train)

tahmin4 = model.predict(X_test)

y_pred4 = tahmin4

cm4 = confusion_matrix(y_test, y_pred4)

###############################################
model = SVC()
f = model.fit(X_train, y_train)

tahmin5 = model.predict(X_test)

y_pred5 = tahmin5

cm5 = confusion_matrix(y_test, y_pred5)

print(cm)
print(cm1)
print(cm2)
print(cm3)
print(cm4)
print(cm5)

plt.matshow(cm, cmap=plt.cm.Purples)
plt.title('Confusion matrix Doğrusal Diskriminant')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

plt.matshow(cm1, cmap=plt.cm.Purples)
plt.title('Confusion matrix Lojistik Regresyon')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

plt.matshow(cm2, cmap=plt.cm.Purples)
plt.title('Confusion matrix K En Yakın Komşu')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

plt.matshow(cm3, cmap=plt.cm.Purples)
plt.title('Confusion matrix Karar Ağacı')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

plt.matshow(cm4, cmap=plt.cm.Purples)
plt.title('Confusion matrix Gaussian Naive Bayes ')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

plt.matshow(cm5, cmap=plt.cm.Purples)
plt.title('Confusion matrix SVC')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

fig = plt.figure()
fig.suptitle('Algoritma Karşılaştırması')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


plt.bar(names,resultsgraph)
plt.xlabel("Algoritmalar")
plt.ylabel("Accuracy")
plt.title("Algoritma Karşılaştırması")
plt.show()



