load Data\mnist_uint8.mat
train_x=double(train_x);
train_y=double(train_y);
test_x=double(test_x);
test_y=double(test_y);


%training data
  [w, train_acc, train_confmat]=train_mlr(train_x,train_y);
  disp(train_confmat);
  disp(train_acc);

%testing data
 [test_acc, test_confmat]=test_mlr(w,test_x,test_y);
 disp(test_confmat);
 disp(test_acc);