def calculate_accuracy(test_predict, testing_label):
  count = 0
  total = len(test_predict)
  for i in test_predict:
    if test_predict[i] == testing_label[i]:
      count+=1;
  accuracy = count/total
  return accuracy
#
# m = calculate_accuracy(test_predict,testing_label)
# print('Test accuracy : ', m)