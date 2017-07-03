import time
import logging

import helper

import tensorflow as tf
import numpy as np


def aver_gradients(tower_grads, clip):
    aver_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)
        grad = tf.clip_by_value(grad, -clip, clip)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        aver_grads.append(grad_and_var)
    return aver_grads

def viterbi(logits, transitions, seqLen):
    '''decoding in tensorflow
       logits: (batch_size, time_step, nb_tags)
       transitions: (nb_tags, nb_tags)
       seqLen: (batch_size, )
       '''
    def _forward(prev, step):
        last_score, _, _ = prev
        path_score = tf.expand_dims(last_score, axis=-1) + trans # B x N x N
        cur_score = step + tf.reduce_max(path_score, reduction_indices=1) # B x N
        backtrace = tf.argmax(path_score, axis=1) # B x N
        best_path = tf.argmax(cur_score, axis=1) # B
        return cur_score, backtrace, best_path

    def _backward(last, step_backtrace):
        indices = tf.stack([ranges, tf.to_int32(last)], axis=1)
        last = tf.gather_nd(step_backtrace, indices)
        return last
        
    logits = tf.transpose(logits, [1,0,2]) #T x B x N
    trans = tf.expand_dims(transitions, axis=0) #1 x N x N
    
    shape = tf.shape(logits)
    batch_sz = shape[1]
    nb_tags = shape[2]
    ranges = tf.range(batch_sz)

    _, backtrace, best_path = tf.scan(fn=_forward,
                                      elems=logits[1:],
                                      initializer=(logits[0], 
                                                   tf.zeros((batch_sz, nb_tags), dtype=tf.int64), 
                                                   tf.zeros((batch_sz,),dtype=tf.int64))
                                      )

    last = tf.gather_nd(best_path, tf.stack([seqLen-2, ranges], axis=1)) # B
    backtrace = tf.reverse_sequence(backtrace, seqLen-1, seq_dim=0, batch_dim=1)
    
    viterbi = tf.scan(fn=_backward,
                      elems=backtrace,
                      initializer=last)

    last = tf.expand_dims(last, axis=0)
    viterbi = tf.concat([last, viterbi], axis=0)
    viterbi = tf.reverse_sequence(viterbi, seqLen, seq_dim=0, batch_dim=1)
    return tf.transpose(viterbi, [1,0])
        
class Model(object):
    
    def __init__(self, devices):
                
        self.nb_device = len(devices)
        self.placeholders = []
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
       
        gradients = []
        losses = []
        outputs = [None]*self.nb_device

        with tf.device('/cpu:0'):
            # add placeholder
            for _ in range(self.nb_device):
                self.placeholders.append(self.add_placeholder())
            #preload input data
            #self.placeholders = list(self.add_placeholder())
            #preloaded = [tf.Variable(placeholder, trainable=False, collections=[]) for placeholder in self.placeholders]
            #preloaded = tf.train.slice_input_producer(preloaded, shuffle=False)
            #data = tf.train.batch(preloaded, 

            # add optimizer
            self.opt = self.add_optimizer()
            # built model on cpu device 
            self.model_fn(self.placeholders[0])

            tf.get_variable_scope().reuse_variables()
            # copy model replicas in gpu
            for idx, gpu in enumerate(devices):
                with tf.name_scope('tower_%d'%idx), tf.device(gpu):
                    output, loss = self.model_fn(self.placeholders[idx])                                                            
                    outputs[idx] = output
                    losses.append(loss)
                    gradients.append(self.opt.compute_gradients(loss))

            with tf.variable_scope(tf.get_variable_scope(), reuse=False):
                aver_gradient = aver_gradients(gradients, clip=10)
                self.train_op = self.opt.apply_gradients(aver_gradient, global_step=self.global_step)
            
            self.loss = tf.reduce_sum(losses)
            self.outputs = tf.concat(outputs, axis=0)
            self.summary = tf.summary.scalar('loss', self.loss)

    def add_placeholder(self):
        raise NotImplementedError()

    def model_fn(self, inputs):
        '''define model structure here
           inputs: placeholders defined by the add_placeholder function
           return: predicts 
                   loss'''
        raise NotImplementedError()
    
    def add_optimizer(self):
        '''define optimizer
        return an tf optimizer'''
        raise NotImplementedError()

    def train(self, sess, saver, 
              x_train, y_train, seqLen_train, 
              x_val, y_val, seqLen_val,
              batch_sz, 
              epoch, 
              patience,
              dropout,
              save_to,
              verbose_step=10):
   
        summary_writer_train = tf.summary.FileWriter(save_to + '/summary/train_loss', sess.graph)  
        summary_writer_val = tf.summary.FileWriter(save_to + '/summary/val_loss', sess.graph)     
        max_f1 = -1000
        max_epoch = -1

        logging.info('='*10+'Training..'+'='*10)
        for i in range(epoch):
            print '='*10+'current epoch: %d'%i+'='*10
            #shuffle data and generate batch
            generator, total_step, _ = helper.batch_generator(x_train, seqLen_train, y_train, [dropout], batch_sz, self.nb_device, shuffle=True)
            start_t = time.time()       
            for step, batch_data in enumerate(generator):
                _, loss_train, train_summary =\
                    sess.run([
                        self.train_op,
                        self.loss,
                        self.summary
                    ],
                    #feed_dict=dict(zip([holder for replica in self.placeholders for holder in replica], batch_data)))
                    feed_dict = dict(zip(self.placeholders, batch_data)))
                if step % verbose_step == 0:
                    summary_writer_train.add_summary(train_summary, step+total_step*i)
                    print "step: %d / %d, loss: %f, time: %s" %(step, total_step, loss_train, time.time()-start_t)
                    start_t = time.time()
            print 'validating..'
            start_t = time.time()
            seg_val, joint_val = self.test(sess, x_val, y_val, seqLen_val)
            print 'validate time cost %s'%(time.time() - start_t)
            logging.info('epoch: %d, segment===valid precision: %.2f, valid recall: %.2f, valid f1: %.2f'%((i,)+seg_val))
            logging.info('epoch: %d, joint===valid precision: %.2f, valid recall: %.2f, valid f1: %.2f'%((i,)+joint_val))
            print 'epoch: %d, segment===valid precision: %.2f, valid recall: %.2f, valid f1: %.2f' %((i,)+seg_val)
            print 'epoch: %d, joint===valid precision: %.2f, valid recall: %.2f, valid f1: %.2f' %((i,)+joint_val)
            if seg_val[-1] > max_f1:
                max_f1 = seg_val[-1]
                best_epoch = i
                save_path = saver.save(sess, save_to)
                print "saved the best model with f1: %.2f" % max_f1
            else:
                if i - best_epoch > patience:
                    logging.info('Early stop')
                    print 'stop training due to early stop'
                    break
        print '='*20
        print 'best model in epoch %d with f1: %.2f'%(best_epoch, max_f1)
    
    def test(self, sess, inputs, targets, seqLen, batch_sz=128):
        y_pred = self.predict(sess, inputs, seqLen, batch_sz)
        seg, joint = self.evaluate(targets, y_pred, seqLen)
        return seg, joint
    
    def predict(self, sess, inputs, seqLen, batch_sz=128):
        generator, _, idx = helper.batch_generator(inputs, seqLen, np.zeros_like(inputs), [0.0],  batch_sz, self.nb_device, False)
        predicts = []
        #transitions = sess.run(self.transitions)
        for step, batch_data in enumerate(generator):
            predict = sess.run(self.outputs, 
                            feed_dict=dict(zip(self.placeholders, batch_data)))
            predicts.extend(predict)

        predicts = [x[0] for x in sorted(zip(predicts, idx), key=lambda x:x[1])]
        predicts = [predict[:Len] for predict, Len in zip(predicts, seqLen)]
        return predicts

    def evaluate(self, y_true, y_pred, seqLen):
        precision = -1.0
        recall = -1.0
        f1 = -1.0
        seg_hit = 0
        joint_hit = 0
        pred_num = 0
        true_num = 0
        
        for i in range(len(y_pred)):
            true_words, true_joint = helper.labelSeq2chunks(y_true[i][:seqLen[i]])
            pred_words, pred_joint = helper.labelSeq2chunks(y_pred[i])

            seg_hit += len(set(true_words) & set(pred_words))
            joint_hit += len(set(true_joint) & set(pred_joint))
            pred_num += len(set(pred_words))
            true_num += len(set(true_words))

        if pred_num != 0:
            precision_seg = 1.0 * seg_hit / pred_num
            precision_joint = 1.0 * joint_hit / pred_num
        if true_num != 0:
            recall_seg = 1.0 * seg_hit / true_num
            recall_joint = 1.0 * joint_hit / true_num
        if seg_hit != 0:
            f1_seg = 2.0 * (precision_seg * recall_seg) / (precision_seg + recall_seg)
        else:
            f1_seg = 0
        if joint_hit != 0:
            f1_joint = 2.0 * (precision_joint * recall_joint) / (precision_joint + recall_joint)
        else:
            f1_joint = 0
        return (precision_seg*100, recall_seg*100, f1_seg*100), (precision_joint*100, recall_joint*100, f1_joint*100)


class bilstm_crf(Model):

    def __init__(self, nb_tags, nb_layers, maxLen, h_dim, embedding, devices):
        
        self.nb_tags = nb_tags
        self.nb_layers = nb_layers
        self.h_dim = h_dim
        self.embedding = embedding
        super(bilstm_crf, self).__init__(devices)
            
    def model_fn(self, x):
        input_x = x[0]
        seqLen = x[1]
        target = x[2]
        dropout = x[3]
        maxLen = tf.shape(input_x)[1]
        #transiton_params
        transitions = tf.get_variable('transitions', [self.nb_tags, self.nb_tags])
        # char embedding
        embedding = tf.get_variable('embedding', 
                                    shape=self.embedding.shape, 
                                    initializer=tf.constant_initializer(self.embedding), 
                                    trainable=True, 
                                    dtype=tf.float32)

        inputs_emb = tf.nn.embedding_lookup(embedding, input_x)

        # lstm cell
        lstm_cell_fw = tf.contrib.rnn.GRUCell(self.h_dim)
        lstm_cell_bw = tf.contrib.rnn.GRUCell(self.h_dim)

        # dropout
        lstm_cell_fw = tf.contrib.rnn.DropoutWrapper(lstm_cell_fw, output_keep_prob=(1 - dropout))
        lstm_cell_bw = tf.contrib.rnn.DropoutWrapper(lstm_cell_bw, output_keep_prob=(1 - dropout))

        lstm_cell_fw = tf.contrib.rnn.MultiRNNCell([lstm_cell_fw] * self.nb_layers)
        lstm_cell_bw = tf.contrib.rnn.MultiRNNCell([lstm_cell_bw] * self.nb_layers)

        # forward and backward
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(
            lstm_cell_fw, 
            lstm_cell_bw,
            inputs_emb,
            seqLen,
            dtype=tf.float32,
            scope='bilstm'
        )
        
        outputs = tf.reshape(tf.concat(outputs, 2), [-1, self.h_dim * 2])
        hidden = tf.contrib.layers.fully_connected(outputs, self.h_dim*2, tf.tanh, reuse=True, scope='hidden')
        logits = tf.contrib.layers.fully_connected(hidden, self.nb_tags, None, reuse=True, scope='logits')
        logits = tf.reshape(logits, (-1, maxLen, self.nb_tags))

        log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(logits, target, seqLen, transitions)
        loss = -tf.reduce_sum(log_likelihood / tf.cast(seqLen, tf.float32))

        predicts = viterbi(logits, transitions, seqLen)

        return predicts, loss

    def add_placeholder(self):
        inputs = tf.placeholder(tf.int32, [None, None])
        seqLen = tf.placeholder(tf.int32)
        targets = tf.placeholder(tf.int32, [None, None])
        dropout = tf.placeholder(tf.float32)
        return inputs, seqLen, targets, dropout

    def add_optimizer(self):
        optimizer = tf.train.AdamOptimizer(0.001)
        return optimizer
