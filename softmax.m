function [w] = softmax(X)
%calculate Softmax
mx=max(X);
X=X-mx;
w=exp(X)/sum(exp(X));

end