# ex1
> # Machine Learning Assignments of Linear Regression for Housing Price

  In this Project we will make Linear Regression model for simple Housing prizing problem using Python. In this Project we are not using any machine learning but insteed we develop function by self. This way we can learn very basic way to implement machine learning. 

  In this project first we are developing machine learning function for linear regression named as LinearRegression(). In Next step we Load training data in function to train and then we check how training being done. To learn basic machine learning here is some discreption,

 X = Features,
 Y = Resposes,
 θ = Learning Parameter,
 α = Learning Rate,
 m = Training set size
 
 ## Hyponthsis function:
 > h(X) = θ(0) + θ(1) * X(1) + θ(2) * X(2) + θ(3) * X(3) ...
 
 ## Cost Functoin : 
 > J = ( h(X) - Y )**2
 
 ## Gradient Decent :
 > grad J = (1/2) * (h(X) - Y)
 
 ## Theta Update rule : 
 >  θ(i) := θ(i) - (α / 2 * m) (**∑** (h(X) - Y) * X(i))
 
  By implementing this function we make LinearRegression function and training to get optimal thetas.
 
