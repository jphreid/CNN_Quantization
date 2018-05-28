import tensorflow as tf
import numpy as np
import Option
from scipy.io import loadmat  # To load SVHN (Iban)
from sklearn.preprocessing import OneHotEncoder # For preprocessing SVHN labels (Iban)

from tensorflow.examples.tutorials.mnist import input_data  # added for MNIST

def preprocess(x, train=False):
  dataSet = Option.dataSet
  if dataSet == 'CIFAR10':
    if train:
      x = tf.image.resize_image_with_crop_or_pad(x, 40, 40)
      x = tf.random_crop(x, [32, 32, 3])
      x = tf.image.random_flip_left_right(x)
  elif dataSet == 'MNIST':
     if train:
       x = tf.reshape(x, [28, 28, 1])
  elif dataSet == 'SVHN':
    if train:
      x = tf.image.resize_image_with_crop_or_pad(x, 40, 40)
      x = tf.random_crop(x, [32, 32, 3])
  else:
    print('Unkown dataset',dataSet,'no preprocess')
  x = tf.transpose(x, [2, 0, 1]) # from HWC to CHW
  return x

# get dataset from NPZ files ; this is for CIFAR10 only
def loadNPZ(pathNPZ):
  data = np.load(pathNPZ)

  trainX = data['trainX']
  trainY = data['trainY']

  testX = data['testX']
  testY = data['testY']

  label = data['label']
  return trainX, trainY, testX, testY, label  
  
def data2Queue(dataX, dataY, batchSize, numThreads, shuffle=False, isTraining=True, seed=None):

  q = tf.FIFOQueue(capacity=dataX.shape[0], dtypes=[dataX.dtype, dataY.dtype],shapes=[dataX.shape[1:],dataY.shape[1:]])
  enqueue_op = q.enqueue_many([dataX, dataY])
  sampleX, sampleY = q.dequeue()
  qRunner = tf.train.QueueRunner(q, [enqueue_op])
  tf.train.add_queue_runner(qRunner)

  sampleX_ = preprocess(sampleX, isTraining)

  if shuffle:
    batchX, batchY = tf.train.shuffle_batch([sampleX_, sampleY],
                                            batch_size=batchSize,
                                            num_threads=numThreads, capacity=dataX.shape[0],
                                            min_after_dequeue=dataX.shape[0] // 2,     # '/' replaced by '//' (Iban)
                                            seed=seed)
  else:
    batchX, batchY = tf.train.batch([sampleX_, sampleY],
                                    batch_size=batchSize,
                                    num_threads=numThreads,
                                    capacity=dataX.shape[0])

  return batchX, batchY
  
def loadData(dataSet,batchSize,numThread):

  if dataSet == 'CIFAR10':
  
    pathNPZ = '../dataSet/' + dataSet + '.npz'
    numpyTrainX, numpyTrainY, numpyTestX, numpyTestY, label = loadNPZ(pathNPZ)
    numTrain = numpyTrainX.shape[0]
    numTest = numpyTestX.shape[0]

  elif dataSet == 'MNIST':
  
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    numpyTrainX = mnist.train.images[:50000,:].reshape(50000,28,28,1)
    numpyTrainY = mnist.train.labels[:50000,:].astype(int)
    numpyTestX = mnist.test.images[:10000,:].reshape(10000,28,28,1)
    numpyTestY = mnist.test.labels[:10000,:].astype(int)
    label = ['zero', 'one','two','three','four', 'five','six','seven','eight','nine']  
    numTrain = numpyTrainX.shape[0]
    numTest = numpyTestX.shape[0]
	
  elif dataSet == "SVHN":

    numpyTrainX = loadmat('dataSVHN/train_32x32.mat')['X']
    numpyTrainX = np.transpose(numpyTrainX, (3,0,1,2))
    numpyTestX = loadmat('dataSVHN/test_32x32.mat')['X']
    numpyTestX = np.transpose(numpyTestX, (3,0,1,2))
    numpyExtraX = loadmat('dataSVHN/extra_32x32.mat')['X']
    numpyExtraX = np.transpose(numpyExtraX, (3,0,1,2))
  
    numpyTrainY = loadmat('dataSVHN/train_32x32.mat')['y']
    enc = OneHotEncoder().fit(numpyTrainY.reshape(-1, 1))
    numpyTrainY = enc.transform(numpyTrainY.reshape(-1, 1)).toarray().astype(np.uint8) 
    numpyTestY = loadmat('dataSVHN/test_32x32.mat')['y']
    numpyTestY = enc.transform(numpyTestY.reshape(-1, 1)).toarray().astype(np.uint8)
    numpyExtraY = loadmat('dataSVHN/extra_32x32.mat')['y']
    numpyExtraY = enc.transform(numpyExtraY.reshape(-1, 1)).toarray().astype(np.uint8)

    numpyTrainX = np.vstack((numpyTrainX,numpyExtraX))
    numpyTrainY = np.vstack((numpyTrainY,numpyExtraY))

    label = ['zero', 'one','two','three', 'four', 'five','six','seven','eight','nine']  
    numTrain = numpyTrainX.shape[0]
    numTest = numpyTestX.shape[0] 

  trainX, trainY = data2Queue(numpyTrainX, numpyTrainY, batchSize,numThread, True, True)
  testX, testY = data2Queue(numpyTestX, numpyTestY, 100, 1, False,False)

  return trainX,trainY,testX,testY,numTrain,numTest,label





