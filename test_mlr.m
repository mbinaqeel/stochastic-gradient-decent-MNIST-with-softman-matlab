function [test_acc,test_confmat] = test_mlr(W,testX,testt )
    x_size=size(testX,1);
    append=ones(x_size,1);
    testX=[testX  append];
    [m,n]=size(testt);
    y=zeros(m,n);
    for i=1:x_size
        y(i,:)=softmax(testX(i,:)*W);
    end

[~,indexy]=max(y');
[~,indext]=max(testt');
match=(indexy==indext);

    test_acc = mean(match)*100;
    
    [order,test_confmat] = confusion(testt',y');
    
end

