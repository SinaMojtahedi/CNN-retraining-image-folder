def Retrain_CNN_Model(Model, data, labels,  Split = .2, Class_Nr = 20, EPCH = 10):

  # I assume that data has 4D, (sample Nr, image length, image width, color channels)
  # Split --> which fraction of dataset is taken as test and train 
  # Class_Nr --> number of classes 
  # EPCH --> epoch numbers

  model = Model
  s, x, y, z = data.shape
  data = data / np.max(data)
  # randomizing data order
  #Data = data[np.random.permutation(s)]
  SP = np.int(np.floor(s * (1- Split)))
  train_data = data[0:SP]
  test_data = data[SP+1:s]
  train_labels = labels[0:SP]
  test_labels = labels[SP+1:s]
  #pdb.set_trace()

  from keras.utils import to_categorical
  #one-hot encode target column
  train_labels = to_categorical(train_labels)
  test_labels = to_categorical(test_labels)

  model.summary()

  #train the model
  model.fit(train_data, train_labels, validation_data=(test_data, test_labels), epochs=EPCH, shuffle=True)

  return model