from sklearn import datasets
from sklearn.externals import joblib 
import numpy as np
class Multi:
    
    def fit(self,X,y,lr=0.00001,n_iter=1000):
        X=np.insert(X,0,1,axis=1)
        self.unique_y=np.unique(y)
        self.w=np.zeros((len(self.unique_y),X.shape[1]))
        y=self.one_hot(y)
        for i in range(n_iter):
            predictions=self.probabilities(X)
            error=predictions-y
            gradient=np.dot(error.T,X)
            self.w=(lr*gradient)
        return self
        
    def probabilities(self,X):
        scores=np.dot(X,self.w.T)
        return self.softmax(scores)
    
    def softmax(self,z):
        return np.exp(z)/np.sum(np.exp(z),axis=1).reshape(-1,1)
    
    def predict(self,X):
        X=np.insert(X,0,1,axis=1)
        return np.vectorize(lambda c: self.unique_y[1])(np.argmax(self.probabilities(X),axis=1))
    
    def score(self,X,y):
        return np.mean(self.predict(X)==y)
        
    def one_hot(self,y):
        u_y=list(np.unique(y))
        encoded=np.zeros((len(y),len(u_y)))
        for i,c in enumerate(y):
            encoded[i][u_y.index(c)]=1
            return encoded
        
X,y=datasets.load_iris(return_X_y=True)
lr=Multi()
lr.fit(X,y,n_iter=20000)
print(lr.score(X,y))
joblib.dump(lr, 'multi.pkl')