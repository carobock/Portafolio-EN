import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

X= np.load("/Users/carol/Documents/MicroMaster/Data Analysis Statistical Modeling and Computation in Applications/Analysis2/data/p2_unsupervised/X.npy")
Y= np.load("/Users/carol/Documents/MicroMaster/Data Analysis Statistical Modeling and Computation in Applications/Analysis2/data/p1/Y.npy")
#print(X.shape)
#print(X[:,0].max())
#X_T= np.log2(X+np.ones([X.shape[0],X.shape[1]]))
#np.save("D.npy", np.log2(X+np.ones([X.shape[0],X.shape[1]])))
D= np.load("D.npy")

#kmeans= KMeans(n_clusters=3, n_init=100).fit(D)
#labels= kmeans.labels_
#np.save("Labels.npy", KMeans(n_clusters=3, n_init=100).fit(D).labels_)
L= np.load("Labels.npy")
#means= np.array([np.mean(D[np.where(labels==i)[0]],0) for i in range (3)])
#plt.scatter(means[:,0],means[:,1], c=[0,1,2,3])
#plt.show()

#pca = PCA()
#x = pca.fit_transform(D)
#a=pca.explained_variance_ratio_[10:2168]
#b= np.where(pca.explained_variance_ratio_.cumsum()>=0.33)[0][0]
#plt.plot(np.arange(1,101),pca.explained_variance_ratio_[0:2168])
#print(b)
#plt.show(b)
#plt.scatter(x[:,0],x[:,1],c=labels)
#plt.title("Scatter Plot of 1st and 2nd Component")
#plt.plot(a)
#plt.title("% Explained Variance by Component")
#plt.xlabel("Component")
#scale_factor = 0.01
#xmin, xmax = plt.xlim()
#ymin, ymax = plt.ylim()
#plt.xlim(xmin * scale_factor, xmax * scale_factor)
#plt.ylim(ymin * scale_factor, ymax * scale_factor)
#plt.show()

#all_kmeans= [KMeans(n_clusters=i+1, n_init=10) for i in range(10)]
#for i in range(10):
#    all_kmeans[i].fit(D)
#inertias= [all_kmeans[i].inertia_ for i in range(10)]
#print(inertias)
#plt.plot(np.arange(1,11,1), inertias)
#plt.xticks(np.arange(1,11,1.0))
#plt.title("KMeans Sum of Squares Criterion")
#plt.show()

#np.save("ztsne_100.npy",TSNE(n_components=2, perplexity=100).fit_transform(D))
#np.save("ztsne_80.npy",TSNE(n_components=2, perplexity=80).fit_transform(D))
#np.save("ztsne_50.npy",TSNE(n_components=2, perplexity=50).fit_transform(D))
#np.save("ztsne_30.npy",TSNE(n_components=2, perplexity=30).fit_transform(D))
#np.save("ztsne_10.npy",TSNE(n_components=2, perplexity=10).fit_transform(D))
#np.save("ztsne_5.npy",TSNE(n_components=2, perplexity=5).fit_transform(D))
#ztsne_100= np.load("ztsne_100.npy")
#ztsne_80= np.load("ztsne_80.npy")
#ztsne_50= np.load("ztsne_50.npy")
#ztsne_30= np.load("ztsne_30.npy")
#ztsne_10= np.load("ztsne_10.npy")
#ztsne_5= np.load("ztsne_5.npy")

#tsne= TSNE(n_components=2,verbose=1, perplexity=5  )
#z_tsne= tsne.fit_transform(D)
#plt.scatter(ztsne_50[:,0],ztsne_50[:,1], c=L)
#plt.title("TSNE, perplexity =50")
#plt.show()

#mds= MDS(n_components=2)
#t= mds.fit(D)
#plt.scatter(t.embedding_[:,0],t.embedding_[:,1], c=labels)
#plt.title("MDS Plot")
#plt.show()

"Logistic Regression"
#"Standarize the features"
#features, target = load(D(return_X_y=True))
#D_std = (features-np.mean(features,0))/np.std(features,0)
"Separe into train and test data"
X_train= np.load("/Users/carol/Documents/MicroMaster/Data Analysis Statistical Modeling and Computation in Applications/Analysis2/data/p2_evaluation/._X_train.npy"
y_train = np.load("/Users/carol/Documents/MicroMaster/Data Analysis Statistical Modeling and Computation in Applications/Analysis2/data/p2_evaluation/._Y_train.npy")
X_test = np.load("/Users/carol/Documents/MicroMaster/Data Analysis Statistical Modeling and Computation in Applications/Analysis2/data/p2_evaluation/._X_test.npy")
y_test = np.load("/Users/carol/Documents/MicroMaster/Data Analysis Statistical Modeling and Computation in Applications/Analysis2/data/p2_evaluation/._Y_test.npy")
#log_reg2 = LogisticRegression(penalty="l2", C=1,max_iter=5000, multi_class="ovr").fit(X_train,Y_train)
#print(log_reg2.score(X_test,Y_test))

#log_reg1 = LogisticRegression(penalty="l1", C=0.1, solver="liblinear", max_iter=5000, multi_class="ovr").fit(X_train,Y_train)
#print (log_reg1.score(X_test,Y_test))

log_reg = LogisticRegressionCV(cv=5, Cs=[0.01,0.1,10], max_iter=5000, penalty="l2", solver="liblinear")
print (log_reg.fit(X_train,Y_train))
print ("Mean accuracy", log_reg.score(X_train,Y_train))
print(log_reg.C_)
print(log_reg.scores_)