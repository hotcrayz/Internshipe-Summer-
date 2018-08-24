import pickle

predictions = pickle.load(open("WISDM_predictions_seg200.p", "rb"))
y_test = pickle.load(open("WISDM_test_seg200.p", "rb"))

#print(predictions)
#print('\n')
#print(predictions.shape)

#print(y_test)
#print('\n')
#print(y_test.shape)

#for i in range(0, 1648) :
#	print(predictions[i][0], predictions[i][1], predictions[i][2], predictions[i][3], predictions[i][4], predictions[i][5])

for i in range(0, 1648) :
	print(y_test[i][0], y_test[i][1], y_test[i][2], y_test[i][3], y_test[i][4], y_test[i][5])

	


