function [w, train_acc, train_confmat] = train_mlr(X,t)
N=size(X,1);
w=zeros(size(X,2)+1,size(t,2));
append=ones(N,1);
bisedinput=[X append];
Divergence_Criteria = 10^-5;
eta = 0.9;
for i=1:25
    wold = w;
    selection = randperm(N);
    for k=1:N
        y=softmax(bisedinput(selection(k),:)*w);
        input=bisedinput(selection(k),:)';
        error = y-t(selection(k),:);
        
        w=w-((eta^i)*(input*error));
    end
    if sumsqr(w - wold) <= Divergence_Criteria, 
                break;
        end
    disp(i)
end
 [train_acc,train_confmat]=test_mlr(w, X,t);
end