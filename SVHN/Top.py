from __future__ import division
import numpy as np
import time
import tensorflow as tf
import NN
import Option
import Log
import getData
import Quantize
from tqdm import tqdm
import pickle

# with tf.Session() as session:
# 	c = session.run(Quantize.C(q))
# 	q = session.run(Quantize.Q(-1, 2))

# for single GPU quanzation
def quantizeGrads(Grad_and_vars):
  if Quantize.bitsG <= 16:
    grads = []
    for grad_and_vars in Grad_and_vars:
      grads.append([Quantize.G(grad_and_vars[0]), grad_and_vars[1]])
    return grads
  return Grad_and_vars

def showVariable(keywords=None):
  Vars = tf.global_variables()
  Vars_key = []
  for var in Vars:
    print(var.device,var.name,var.shape,var.dtype)
    if keywords is not None:
      if var.name.lower().find(keywords) > -1:
        Vars_key.append(var)
    else:
      Vars_key.append(var)
  return Vars_key

def main(nb_epoch):
  # get Option
  GPU = Option.GPU
  batchSize = Option.batchSize
  pathLog = '../log/' + Option.Time + '(' + Option.Notes + ')' + '.txt'
  Log.Log(pathLog, 'w+', 1) # set log file
  print(time.strftime('%Y-%m-%d %X', time.localtime()), '\n')
  print(open('Option.py').read())

  # get data
  numThread = 4*len(GPU)
  assert batchSize % len(GPU) == 0, ('batchSize must be divisible by number of GPUs')

  with tf.device('/cpu:0'):
    batchTrainX,batchTrainY,batchTestX,batchTestY,numTrain,numTest,label = getData.loadData(Option.dataSet,batchSize,numThread)
  batchNumTrain = int(numTrain / batchSize)
  batchNumTest = int(numTest / 100)

  optimizer = Option.optimizer
  global_step = tf.get_variable('global_step', [], dtype=tf.int32, initializer=tf.constant_initializer(0), trainable=False)
  Net = []


  # on my machine, alexnet does not fit multi-GPU training
  # for single GPU
  with tf.device('/gpu:%d' % GPU[0]):
    Net.append(NN.NN(batchTrainX, batchTrainY, training=True, global_step=global_step))
    lossTrainBatch, errorTrainBatch = Net[-1].build_graph()
    update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # batchnorm moving average update ops (not used now)

    # since we quantize W at the beginning and the update delta_W is quantized,
    # there is no need to quantize W every iteration
    # we just clip W after each iteration for speed
    update_op += Net[0].W_clip_op

    gradTrainBatch = optimizer.compute_gradients(lossTrainBatch)

    gradTrainBatch_quantize = quantizeGrads(gradTrainBatch)
    with tf.control_dependencies(update_op):
      train_op = optimizer.apply_gradients(gradTrainBatch_quantize, global_step=global_step)

    tf.get_variable_scope().reuse_variables()
    Net.append(NN.NN(batchTestX, batchTestY, training=False))
    _, errorTestBatch = Net[-1].build_graph()



  showVariable()

  # Build an initialization operation to run below.
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  config.allow_soft_placement = True
  config.log_device_placement = False
  sess = Option.sess = tf.InteractiveSession(config=config)
  sess.run(tf.global_variables_initializer())
  saver = tf.train.Saver(max_to_keep=None)
  # Start the queue runners.
  tf.train.start_queue_runners(sess=sess)


  def getErrorTest():
    errorTest = 0.
    for i in tqdm(range(batchNumTest),desc = 'Test', leave=False):
      errorTest += sess.run([errorTestBatch])[0]
    errorTest /= batchNumTest
    return errorTest

  if Option.loadModel is not None:
    print('Loading model from %s ...' % Option.loadModel),
    saver.restore(sess, Option.loadModel)
    print('Finished'),
    errorTestBest = getErrorTest()
    print('Test:', errorTestBest)

  else:
    # at the beginning, we discrete W
    sess.run([Net[0].W_q_op])

  print('\nOptimization Start!\n')
  
  lossTotal_ = []
  errorTotal_ = []
  errorTest_ = []
  #####################################################################################################################################################################
  gradH_ = []
  gradW_ = []
  gradWq_ = []
  
  for epoch in range(nb_epoch):
    # check lr_schedule
    if len(Option.lr_schedule) / 2:
      if epoch == Option.lr_schedule[0]:
        Option.lr_schedule.pop(0)
        lr_new = Option.lr_schedule.pop(0)
        if lr_new == 0:
          print('Optimization Ended!')
          exit(0)
        lr_old = sess.run(Option.lr)
        sess.run(Option.lr.assign(lr_new))
        print('lr: {} -> {}'.format(lr_old, lr_new))

    print('Epoch: {}'.format(epoch)),


    lossTotal = 0.
    errorTotal = 0
    t0 = time.time()
    for batchNum in tqdm(range(batchNumTrain), desc='Epoch: %03d' % epoch, leave=False, smoothing=0.1):
      if Option.debug is False:
        _, loss_delta, error_delta = sess.run([train_op, lossTrainBatch, errorTrainBatch])
      else:
        _, loss_delta, error_delta, H, W, W_q, gradH, gradW, gradW_q=\
        sess.run([train_op, lossTrainBatch, errorTrainBatch, Net[0].H, Net[0].W, Net[0].W_q, Net[0].gradsH, Net[0].gradsW, gradTrainBatch_quantize])

      lossTotal += loss_delta
      errorTotal += error_delta

    lossTotal /= batchNumTrain
    errorTotal /= batchNumTrain

    print('Loss: {} Train: {}'.format(lossTotal, errorTotal)),

    # get test error
    errorTest = getErrorTest()
    print('Test: {} FPS: {}'.format(errorTest, numTrain / (time.time() - t0))),

    if epoch == 0:
      errorTestBest = errorTest
    if errorTest < errorTestBest:
      if Option.saveModel is not None:
        saver.save(sess, Option.saveModel)
        print('S'),
    if errorTest < errorTestBest:
      errorTestBest = errorTest
      print('BEST')

    lossTotal_.append(lossTotal)
    errorTotal_.append(errorTotal)
    errorTest_.append(errorTest)


    if epoch%5 == 0: 
      dd = {}
      dd['lossTotal'] = lossTotal_
      dd['errorTotal'] = errorTotal_
      dd['errorTest'] = errorTest_
      filename = '../savedata/temp_savedata.pkl'
      with open(filename, 'wb') as  f: 
        pickle.dump(dd, f)

  
  # _, loss_delta, error_delta, H, W, W_q, gradH, gradW, gradW_q=\
  # 	sess.run([train_op, lossTrainBatch, errorTrainBatch, Net[0].H, Net[0].W, Net[0].W_q, Net[0].gradsH, Net[0].gradsW, gradTrainBatch_quantize])

  return lossTotal_, errorTotal_, errorTest_ # , gradH, gradW, gradW_q

##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################

if __name__ == '__main__':

  name_file = 'SVHN_2888'
  nb_epoch = 40
  
  # lossTotal, errorTotal, errorTest, gradH, gradW, gradWq = main(nb_epoch)
  lossTotal, errorTotal, errorTest = main(nb_epoch)
  
  dd = {}
  dd['lossTotal'] = lossTotal
  dd['errorTotal'] = errorTotal
  dd['errorTest'] = errorTest    
  # dd['gradH'] = gradH
  # dd['gradW'] = gradW
  # dd['gradWq'] = gradWq

  filename = '../savedata/{}_epoch_{}_extra_data_v2.pkl'.format(name_file, nb_epoch)
  with open(filename, 'wb') as  f: 
    pickle.dump(dd, f)

