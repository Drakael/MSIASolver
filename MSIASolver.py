from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.datasets import make_classification

#fonction utile pour le débugging
def p(name,obj):
    if hasattr(obj,'shape'):
        print(name,type(obj),obj.shape,"\n",obj)
    else:
        print(name,type(obj),"\n",obj)

#classe abstraite pour les classifieurs ( = modèles prédictifs)
class MSIAClassifier(ABC):
    """Base class for Linear Models"""
    def __init__(self, learning_rate=0.5, max_iterations=500, starting_thetas = None, range_x = 1, nb_samples = 0):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.starting_thetas = starting_thetas
        self.predicted_thetas = None
        self.range_x = range_x
        self.nb_samples = nb_samples
        self.progression = None
        self.minima = None
        self.maxima = None
        self.mean = None
        self.std = None
        self.ptp = None
        self.scale_mode = 'ptp'
    
    @abstractmethod
    def fit(self, X, Y):
        """Fit model."""
        pass
        
    @abstractmethod
    def predict(self, X):
        """Predict using the linear model
        """
        pass

    @abstractmethod
    def regression_cost(self, model, theta, X, Y):
        """Calculate and return cost
        """
        pass

    @abstractmethod
    def randomize_model(self, theta, X, range_x, random_ratio=0.0, offsets=None):
        """Calculate target for initial training set and weights
        """
        pass

    @abstractmethod
    def plot_1_dimension(self, X, Y):
        """Plot visual for 1 dimentionnal problem
        """
        pass
     
    #fonction de normalisation par minimum et maximum
    def scale_(self, X):
        """Scale X values with min and max
        """
        self.minima = []
        self.maxima = []
        for i in range(X.shape[1]):
            min_ = np.min(X[:,i])
            self.minima.append(min_)
            max_ = np.max(X[:,i])
            self.maxima.append(max_)
            X[:,i]-=min_
            X[:,i]/=max_-min_
        return X
    
    #fonction de normalisation par moyenne (=mean) et plage de valeurs (=range)
    def scale(self, X, on='ptp'):
        """Scale X values with means and ranges
        """
        self.scale_mode = on
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.ptp = X.ptp(axis=0)
        if self.scale_mode == 'ptp':
            ar_std = (X - self.mean) / self.ptp
        else:
            ar_std = (X - self.mean) / self.std
        #verif
        #print('mean_',self.mean,' / ',np.mean(ar_std,axis=0))
        #print('std_',self.std,' / ',np.std(ar_std,axis=0))
        #print('ptp_',self.ptp,' / ',np.ptp(ar_std,axis=0))
        return ar_std
    
    #fonction de remise à l'échelle des poids prédis selon la normalisation initiale
    def rescale(self):
        """Rescale weights to original scale
        """
        array = []
        #p('self.predicted_thetas',self.predicted_thetas)
        theta_zero = self.predicted_thetas[0].copy()
        for col, mean, std, ptp in zip(self.predicted_thetas[1:], self.mean, self.std, self.ptp):
            if self.scale_mode == 'ptp':
                theta_i = float((col) / ptp)
            else:
                theta_i = float((col) / std)
            array.append(theta_i)
            theta_zero -= theta_i * mean
        #p('theta_zero',theta_zero)
        #p('array',array)
        #p('self.predicted_thetas before',self.predicted_thetas)
        self.predicted_thetas = np.array([float(theta_zero),]+array).reshape(len(self.predicted_thetas),1)
        #p('self.predicted_thetas after',self.predicted_thetas)
        return self

    def linear_regression(self, theta, x):
        """linear regression method
        """
        if isinstance(x, int):
            if theta.shape[0]==len(x)+1:
                x = np.concatenate([[1,],x])
        elif type(x).__module__ == np.__name__:
            if len(x.shape) == 1:
                x = x.reshape(1,x.shape[0])
            elif len(x.shape) == 0:
                x = np.array(x).reshape(1,1)
            if theta.shape[0]==x.shape[1]+1:
                x = np.column_stack((np.ones(len(x)),x))
        else:
            print('different type!!!!',type(x))
        return np.matmul(x,theta) 

    def get_cost_derivative(self, model, theta, X, Y):
        """cost derivative calculation
        """
        if theta.shape[0]==X.shape[1]+1:
            X = np.column_stack((np.ones(self.nb_samples),X))
        result = []
        diff = model(theta, X)-Y
        diff_reshaped = diff.reshape(1,X.shape[0])
        for i, t in enumerate(theta):
            #deriv = (1/self.nb_samples) * np.matmul((model(theta, X)-Y).reshape(1,X.shape[0]),X[:,i])
            result.append(np.matmul(diff_reshaped,X[:,i]) / self.nb_samples)
        return np.array(result).reshape(len(result),1)

    def get_cost_derivative_(self, model, theta, X, Y):
        """cost derivative calculation from Achille
        """
        if theta.shape[0]==X.shape[1]+1:
            X = np.column_stack((np.ones(self.nb_samples),X))
        diff_transpose = (model(theta, X)-Y).T
        return np.array([np.sum(np.matmul(diff_transpose, X[:,i]))/(X.shape[0]) for i, t in enumerate(theta)]).reshape(len(theta),1)

    def plot_progression(self):
        """plot learning progression
        """
        if self.progression is not None:
            plt.plot(self.progression, label='progression')
            plt.legend()
            plt.show()

    def gradient_descent(self, initial_model, X, Y, max_iteration, alpha, starting_thetas=None):
        """performs gradient descent
        """
        self.nb_samples = len(X)
        if starting_thetas is None:
            self.starting_thetas = np.random.random((X.shape[1]+1,1))
        self.predicted_thetas = self.starting_thetas
        self.progression = []
        cnt = max_iteration
        cout = 1
        while cnt > 0 and cout != 0.0:#np.abs(cout) > 0.00000001:
            iteration = self.get_cost_derivative(initial_model, self.predicted_thetas, X, Y)
            iteration*= alpha
            self.predicted_thetas = self.predicted_thetas - iteration
            cout = self.regression_cost(initial_model, self.predicted_thetas, X, Y)
            self.progression.append(cout)
            cnt-=1
        self.plot_progression()
        self.plot_1_dimension(X, Y)
        return self.predicted_thetas

#Classe de régression linéaire
class LinearRegression(MSIAClassifier):
    """Linear Regression Class
    """
    def __init__(self, learning_rate=3*10**-1, max_iterations=500, starting_thetas = None, range_x = 1, nb_samples = 0):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.starting_thetas = starting_thetas
        self.range_x = range_x
        self.nb_samples = nb_samples
        MSIAClassifier.__init__(self, self.learning_rate, self.max_iterations, self.starting_thetas, self.range_x, self.nb_samples)

    def fit(self, X, Y):
        """Linear Regression Fit
        """
        X = self.scale(X)
        self.predicted_thetas = self.gradient_descent(self.linear_regression, X, Y, self.max_iterations, self.learning_rate)
        self.rescale()
        return self

    def predict(self, X):
        """Linear Regression Prediction
        """
        return self.linear_regression(self.predicted_thetas, X)

    def regression_cost(self, model, theta, X, Y):
        """Linear Regression cost calculation
        """
        #return float(1/(2 * len(X)) * np.matmul((model(theta, X)-Y).T,(model(theta, X)-Y)))
        return float(np.sum(np.abs(model(theta,X)-Y))/ (self.nb_samples * X.shape[1] * self.range_x)) 

    def randomize_model(self, theta, X, range_x, random_ratio=0.0, offsets=None):
        """Linear Regression randomize function
        """
        if theta.shape[0]==X.shape[1]+1:
            X = np.column_stack((np.ones(self.nb_samples),X))
        produit = np.matmul(X,theta)
        if random_ratio != 0.0:
            produit+= (np.random.random(produit.shape)-0.5)*range_x*random_ratio
        return produit

    def plot_1_dimension(self, X, Y):
        """Linear Regression 1 dimensionnal plot
        """
        if(len(self.predicted_thetas)==2):
            plt.figure()
            plt.plot(X, Y  , 'o', label='original data')
            plt.plot(X, self.predicted_thetas[0] + self.predicted_thetas[1]*X, 'r', label='fitted line')
            plt.legend()
            plt.show()



class LogisticRegression(MSIAClassifier):
    """Logistic Regression Class
    """
    def __init__(self, learning_rate=0.5, max_iterations=500, predicted_thetas = None, range_x = 1, nb_samples = 0):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.predicted_thetas = predicted_thetas
        self.range_x = range_x
        self.nb_samples = nb_samples
        MSIAClassifier.__init__(self, self.learning_rate, self.max_iterations, self.predicted_thetas, self.range_x, self.nb_samples)

    def fit(self,X,Y):
        """Logistic Regression Fit
        """
        self.predicted_thetas = self.gradient_descent(self.sigmoid, X, Y, self.max_iterations, self.learning_rate)
        self.predicted_thetas/= np.absolute(self.predicted_thetas[:,0]).max()
        return self

    def predict(self,X):
        """Logistic Regression Prediction
        """
        return self.sigmoid(self.predicted_thetas, X)

    def sigmoid(self, theta, x):
        """Logistic Regression Sigmoid function
        """
        sigmoid = 1/(1+np.exp(self.linear_regression(theta, x)*-1))
        if sigmoid.shape == (1,1):
            sigmoid = sigmoid[0][0]
        return np.round(sigmoid,2)

    def regression_cost_(self, model, theta, X, Y):
        """Logistic Regression Cost calculation (heavy)
        """
        somme = 0
        for x, y in zip(X, Y):
            sig = np.clip(self.sigmoid(theta,x),0.00000001,0.99999999)
            somme+= (y * np.log(sig)) + ( (1-y) * np.log(1 - sig) )  
        somme/= -self.nb_samples    
        return float(somme) 
    
    def regression_cost__(self, model, theta, X, Y):
        """Logistic Regression Cost calculation (heavy)
        """
        somme = 0
        for x, y in zip(X, Y):
            sig = np.clip(self.sigmoid(theta,x),0.00000001,0.99999999)
            somme+= (y * ((1/sig)-1))+ ( (1-y) * ((1/(1-sig))-1) )  
        somme/= self.nb_samples    
        return float(somme) 

    def regression_cost(self, model, theta, X, Y):
        """Logistic Regression Cost calculation (regular)
        """
        #cout = float(np.sum(np.absolute(model(theta,X)-Y))/ self.nb_samples) 
        cout = float(np.absolute(model(theta,X)-Y).mean())
        return cout

    def randomize_model(self, theta, X, range_x, random_ratio=0.0, offsets=None):
        """Logistic Regression Randomize function
            TODO: test sur random_ratio qui doit être entre 0.0 et 1.0
        """
        #X = self.scale(X)
        #offsets_length = len(offsets)
        #if offsets_length==X.shape[1]:
            #for i in range(offsets_length):
                #p('X[:,i] before',X[:,i])
                #print('X[:,i].mean',X[:,i].mean())
                #p('offsets[i]',offsets[i])
                #X[:,i] += offsets[i]#X[:,i].mean()
                #X[:,i] -= X[:,i].mean()
                #p('X[:,i] after',X[:,i])
                #print('X[:,i].mean',X[:,i].mean())
        #elif offsets_length==X.shape[1]+1:
            #for i in range(offsets_length):
                #p('X[:,i+1] before',X[:,i+1])
                #print('X[:,i+1].mean',X[:,i+1].mean())
                #p('offsets[i]',offsets[i])
                #X[:,i+1] += offsets[i]#X[:,i+1].mean()
                #X[:,i+1] -= X[:,i+1].mean()
                #p('X[:,i+1] after',X[:,i+1])
                #print('X[:,i+1].mean',X[:,i+1].mean())
        if theta.shape[0]==X.shape[1]+1:
            X = np.column_stack((np.ones(self.nb_samples),X))
        produit = []
        for x in X:
            sig = self.sigmoid(theta,x.reshape(1,len(x)))
            val = 1 if sig > 0.5 else 0
            if random_ratio != 0.0:
                val = val if np.random.random() < random_ratio else 1 - val
            produit.append(val)
        return np.array(produit,order='F').reshape(self.nb_samples,1)

    def plot_1_dimension(self, X, Y):
        """Logistic Regression 1 dimensionnal plot
        """
        if(len(self.predicted_thetas)==2):
            plt.figure()
            plt.plot(X, Y, 'o', label='original data')
            x = np.linspace(-self.range_x/2,self.range_x/2,100)
            y = []
            for var in x:
                sig = self.sigmoid(self.predicted_thetas, var)
                y.append(sig)
            plt.plot(x,y,'r')
            plt.legend()
            plt.show()
            
            
class MSIASolver():
    """Solver class
    """
    def __init__(self):
        self.__clf = None
        self.__X = None
        self.__Y = None
        self.__nb_dimensions = None
        self.__nb_samples = None
        self.__use_classifier = None
        
    def fit(self,X,Y):
        """Solver Fit
        """
        self.__X = X.copy()
        nb_samples, nb_dimensions = X.shape
        self.set_nb_samples(nb_samples)
        self.set_nb_dimensions(nb_dimensions)
        self.__Y = Y.copy()
        #todo: tests sur les données
        self.__choose_classifier()
        if self.__use_classifier == 'LinearRegression':
            self.__clf = LinearRegression()
        elif self.__use_classifier == 'LogisticRegression':
            self.__clf = LogisticRegression()
        self.__clf.fit(self.__X,self.__Y)
        
        return self

    def predict(self,X):
        """Solver Prediction
        """
        return self.__clf.predict(self.X)
    
    def __choose_classifier(self):
        """Solver: automatic classifier choice
        """
        if(self.__Y.shape[1]==1):
            self.__use_classifier = 'LinearRegression'
            min_ = self.__Y.min(axis=0)
            p('self.__Y.dtype',self.__Y.dtype)
            if self.__Y.dtype == 'int32' and min_ >= 0:
                unique = self.__Y.astype(float).unique()
                test = True
                for item in unique:
                    if item.is_interger() == False:
                        test = False
                if test == True:
                    self.__use_classifier = 'LogisticRegression'
                    
    def set_learning_rate(self, learning_rate):
        if self.__clf != None:
            self.__clf.learning_rate = learning_rate
            
    def set_max_iterations(self, max_iterations):
        if self.__clf != None:
            self.__clf.max_iterations = max_iterations
            
    def set_predicted_thetas(self, predicted_thetas):
        if self.__clf != None:
            self.__clf.predicted_thetas = predicted_thetas
            
    def set_range_x(self, range_x):
        if self.__clf != None:
            self.__clf.range_x = range_x
            
    def set_nb_samples(self, nb_samples):
        self.__nb_samples = nb_samples
        if self.__clf != None:
            self.__clf.nb_samples = nb_samples
            
    def set_nb_dimensions(self, nb_dimensions):
        self.__nb_dimensions = nb_dimensions
        if self.__clf != None:
            self.__clf.nb_dimensions = nb_dimensions
            
    def get_starting_thetas(self):
        if self.__clf != None:
            return self.__clf.starting_thetas
        return None
    
    def get_predicted_thetas(self):
        if self.__clf != None:
            return self.__clf.predicted_thetas
        return None
            
                    
def severe_randomizer(clf, theta, n_samples, n_dimensions, range_x = 10000):
    X = []
    degre = np.floor(np.log10(range_x))
    if degre < 1:
        degre = 1
    rand_categories = []
    rand_offsets = []
    for i in range(n_dimensions):
        rand_category = np.random.randint(0,degre)
        rand_categories.append(rand_category)
        #rand_offset = np.random.randint(0,degre)-((degre-1)/2)
        rand_offset = (np.random.random()-0.5)*10**rand_category
        rand_offsets.append(rand_offset)
        
    for i in range(n_samples):
        row = []
        for j in range(n_dimensions):
            value = (np.random.random()*range_x)-(range_x/2)
            value*=10**rand_categories[j]
            value-= rand_offsets[j]
            row.append(value)
        X.append(row)
    X = np.array(X)
    Y = clf.randomize_model(theta, X, range_x, 0.0, rand_offsets) 
    return X, Y




#on enregistre le temps courant
# = on démarre le chronomètre
tic_time = datetime.now()

#variables de base
n_dimensions = 36
n_samples = 1500
range_x = 150000  
 
#calcul aléatoire de poids pour le modèle théorique
theta_original = (np.random.random((n_dimensions+1,1))*range_x)-(range_x/2)
#calcul aléatoire de poids de départ pour le modèle prédictif
#theta_initial = (np.random.random((n_dimensions+1,1))*range_x)-(range_x/2)
#theta_initial = np.zeros((n_dimensions+1,1))

#déclaration du modèle prédictif
solver = MSIASolver()
#clf = LinearRegression(0.3, 4000, theta_initial, range_x, n_samples)
#clf = LogisticRegression(0.5, 2000, theta_initial, range_x, n_samples)

#initialisation aléatoire du set d'entrainement
#X, Y = severe_randomizer(clf, theta_original, n_samples, n_dimensions, range_x)
#X = np.array([(np.random.random(n_dimensions)*range_x)-(range_x/2) for x in range(n_samples)])
#X = np.array([(np.random.random(n_dimensions)) for x in range(n_samples)])

#from sklearn.datasets import fetch_california_housing
#dataset = fetch_california_housing()
#X, Y = dataset.data, dataset.target
#n_samples, n_dimensions = X.shape
#Y = Y.reshape(len(Y),1)

degre = np.floor(np.log10(range_x))
if degre < 1:
    degre = 1
rand_categories = []
rand_offsets = []
for i in range(n_dimensions):
    rand_category = np.random.randint(0,degre)
    rand_categories.append(rand_category)
    #rand_offset = np.random.randint(0,degre)-((degre-1)/2)
    rand_offset = (np.random.random()-0.5)#*10**rand_category
    rand_offsets.append(rand_offset)
X, Y = make_classification(n_samples=n_samples,
                           n_features=n_dimensions,
                           n_informative=n_dimensions,
                           #scale=range_x,
                           shift=rand_offsets,
                           n_redundant=0,
                           n_repeated=0,
                           n_classes=2,
                           random_state=np.random.randint(100),
                           shuffle=False)
Y = Y.reshape(len(Y),1)

#on normalise les poids du set initial 
#(=met la valeur maximale à 1 et toutes les autres à l'échelle en dessous) 
#pour comparaison finale avec le modèle prédit
#if type(clf).__name__ == 'LogisticRegression':
#    print('LogisticRegression!!!!!!!!!!!!!!!!!')
#    p('theta_original',theta_original)
#    theta_original/= np.absolute(theta_original[:,0]).max()
#    p('theta_original after scaling',theta_original)

#affichages préliminaires
#p('X',X)
#p('theta_original',theta_original)
#p('theta_initial',theta_initial)

#calcule bruité des cibles selon le modèle théorique
#Y = clf.randomize_model(theta_original,X,range_x)  
#p('Y',Y)

#entrainement du modèle sur les données
solver.fit(X, Y)
#clf.fit(X, Y)

#affichage finaux
predicted_thetas = solver.get_predicted_thetas()
print('Theta start',"\n",solver.get_starting_thetas())   
print("Theta target\n",theta_original) 
print('Theta end : ',"\n",predicted_thetas)
#if type(clf).__name__ == 'LinearRegression':
print('Means : ',"\n",clf.mean)
print('StDs : ',"\n",clf.std)
print('Ranges : ',"\n",clf.ptp)
print('Erreurs : ',"\n",theta_original-predicted_thetas)
print('Erreur globale : ',"\n",np.sum(theta_original-predicted_thetas))
print('Erreur moyenne : ',"\n",np.sum(theta_original-predicted_thetas)/len(X))
#arrêt du chronomètre
delta_time = (datetime.now()) - tic_time
#affichage du temps de calcul global
print('Script executed in',delta_time.days,'d',delta_time.seconds,'s',delta_time.microseconds,'µs')
