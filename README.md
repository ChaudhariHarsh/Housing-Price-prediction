# ex1
> # Machine Learning Assignments of Linear Regression for Housing Price

  In this Project we will make Linear Regression model for simple Housing prizing problem using Python. In this Project we are not using any machine learning but insteed we develop function by self. This way we can learn very basic way to implement machine learning. 

  In this project first we are developing machine learning function for linear regression named as LinearRegression(). In Next step we Load training data in function to train and then we check how training being done. To learn basic machine learning here is some discreption,

 X = Features,
 Y = Resposes,
 W = weights - Learning Parameter,
 b = bias - Learning Parameter
 α = Learning Rate,
 m = Training set size
 
 ## Hyponthsis function:
 > h(X) = b + W(1) * X(1) + W(2) * X(2) + W(3) * X(3) ...
 
 ## Cost Functoin : 
 > Cost_Function = ( h(X) - Y )**2
 
 ## Gradient Decent :
 > grad J = (1/2) * (h(X) - Y)

 ## Weight Update rule : 
 >  b := b - (α / 2 * m) (**∑** (h(X) - Y) * X(i))
 
 ## Weight Update rule : 
 >  W[i] := w[i] - (α / 2 * m) (**∑** (h(X) - Y) * X(i))
 
  By implementing this function we make LinearRegression function and training to get optimal Learning Parameters b and W.
 
 # Visualization :  
  I have implemented simple visualization with iteration = 1000 and learning_rate = 0.1
  
 
![Alt text](https://github.com/ChaudhariHarsh/ex1/blob/master/LinearRe.png?raw=true "Linear Regression Learning Visualization")

this way we can train simple linear regression model.
