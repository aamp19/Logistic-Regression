import numpy as np
import util
import sys
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split

def sigmoid(x): 
    return 1.0/(1 + np.exp(-x))


### NOTE : You need to complete logreg implementation first!
class LogisticRegression:
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=1000000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    
    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        v = 0
        h = 0
        m = len(y)
        for i in range (len(y)):
            v += (1/m)*(sigmoid(np.dot(self.theta.transpose(),x[i])) - y[i]) * x[i]
                          
            h += (1/m)*(sigmoid(np.dot(self.theta.transpose(),x[i]))) * (1 - (sigmoid(np.dot(self.theta.transpose(),x[i])))) * x[i] * x[i].transpose()
        self.theta = (self.theta - ((1/h)*v))
        
        
            
        # *** END CODE HERE ***

    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***

        y_hat= np.round(sigmoid(np.dot(self.theta.reshape(2, 1).transpose(),x.transpose())))
        return y_hat
        
        # *** END CODE HERE ***


# Character to replace with sub-problem letter in plot_path/save_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, save_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on t-labels,
        2. on y-labels,
        3. on y-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        save_path: Path to save predictions.
    """
    output_path_true = save_path.replace(WILDCARD, 'true')
    output_path_naive = save_path.replace(WILDCARD, 'naive')
    output_path_adjusted = save_path.replace(WILDCARD, 'adjusted')

    # *** START CODE HERE ***
    # Part (a): Train and test on true labels
    
    #split dataset into train and test
    #call fit on train
    #call predict on test
    Newtonplot = LogisticRegression(theta_0 = np.array([0.00001,0.0001]))
    Validplot = LogisticRegression(theta_0 = np.array([0.00001,0.0001]))
    train = pd.read_csv(train_path) #read train set into the program
    valid = pd.read_csv(valid_path) #read valid set into the program
    test = pd.read_csv(test_path) #read test set into the program
    
    train_x = np.array(train[["x_1", "x_2"]].values) #identify input data in train set
    train_y = np.array(train["y"].values) #identify output data in train set
    
    
    
    test_x = np.array(test[["x_1","x_2"]].values) #identify input data in test set
    test_y = np.array(test["y"].values) #identify output data in test set
    
    valid_x = np.array(valid[["x_1", "x_2"]].values) #identify input data in valid set
    valid_y = np.array(valid["y"].values) #identify output data in valid set
    
    c1=np.array([j for i,j in enumerate(test_x) if test_y[i]==0]) #group elements in the test output that are 0
    x1_c1=c1[:,0]
    x2_c1=c1[:,1]
    
    c2= np.array([j for i,j in enumerate(test_x) if test_y[i]==1])#group elements in the test output that are 1
    x1_c2=c2[:,0]
    x2_c2=c2[:,1]
    
    c5 = np.array([j for i,j in enumerate(valid_x) if valid_y[i]==0])
    x1_c5=c5[:,0]
    x2_c5=c5[:,0]
    
    c6 = np.array([j for i,j in enumerate(valid_x) if valid_y[i]==1])
    x1_c6=c6[:,0]
    x2_c6=c6[:,0]
    
    Newtonplot.fit(train_x, train_y) #train the logistic regression object
    Newtonpredict = Newtonplot.predict(test_x) #see future values of the regression object based on the trained data
    Validplot.fit(valid_x,valid_y)
    Validpredict = Validplot.predict(valid_x)
    #print(Newtonpredict)
    c3=np.array([j for i,j in enumerate(test_x) if Newtonpredict[0][i]==0])  #group elements in the future prediction output that are 0
    x1_c3 = c3[:,0]
    x2_c3 = c3[:,1]
    
    c4=np.array([j for i,j in enumerate(test_x) if Newtonpredict[0][i]==1]) #group elements in the future prediction output that are 1
    x1_c4 = c4[:,0]
    x2_c4 = c4[:,1]
    
    c7=np.array([j for i,j in enumerate(valid_x) if 0.5*Validpredict[0][i]==0])
    x1_c7 = c7[:,0]
    x2_c7 = c7[:,1]
    
    c8=np.array([j for i,j in enumerate(valid_x) if 0.5*Validpredict[0][i]==0])
    x1_c8 = c8[:,0]
    x2_c8 = c8[:,1]
    #plt.scatter(x1_c1,x2_c1, label = 'outcome is 0')
    #plt.scatter(x1_c2,x2_c2, label = 'outcome is 1',color = 'red')
    #plt.scatter(x1_c1, x2_c1, label = 'predicted outcome is 0',color = 'red')
    #plt.scatter(x1_c2, x2_c2, label = 'predicted outcome is 1',color = 'green')
    plt.scatter(x1_c7, x2_c7, label = 'predicted outcome is 0',color = 'red')
    plt.scatter(x1_c8, x2_c8, label = 'predicted outcome is 1',color = 'green')
    
    plt.legend()
    # Make sure to save predicted probabilities to output_path_true using np.savetxt()
    # Part (b): Train on y-labels and test on true labels
    # Make sure to save predicted probabilities to output_path_naive using np.savetxt()
    # Part (f): Apply correction factor using validation set and test on true labels
    # Plot and use np.savetxt to save outputs to output_path_adjusted
    # *** END CODER HERE

if __name__ == '__main__':
    main(train_path='train.csv',
        valid_path='valid.csv',
        test_path='test.csv',
        save_path='posonly_X_pred.txt')
