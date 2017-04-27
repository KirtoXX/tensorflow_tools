import tensorflow as tf

class mutul_gpu:

    def __init__(self):
        self.device_list = ['/gpu:0']
        self.MOVING_AVERAGE_DECAY = 0.99
        self.network = None   #自定义的网络结构
        self.trian_op = None  #优化方法
        self.loss = None      #损失函数
        self.real_target = None  #z真实标签

    def set_device(self,device_list):
        self.device_list = device_list

    def set_paramater(self,input,real_target,network,loss,train_op):
        self.network = network
        self.loss = loss
        self.trian_op = train_op
        self.input = input
        self.real_target = real_target

    def average_gradients(self,tower_grads):
        average_grads = []

        # 枚举所有的变量和变量在不同GPU上计算得出的梯度。
        for grad_and_vars in zip(*tower_grads):
            # 计算所有GPU上的梯度平均值。
            grads = []
            for g, _ in grad_and_vars:
                expanded_g = tf.expand_dims(g, 0)
                grads.append(expanded_g)
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)

            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            # 将变量和它的平均梯度对应起来。
            average_grads.append(grad_and_var)
        # 返回所有变量的平均梯度，这个将被用于变量的更新。
        return average_grads


    def _get_loss(self,x, y_,network,loss_function,scope, reuse_variables=True):
        with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
            y = network(x)
        loss = loss_function(y,y_)
        tf.add_to_collection('losses',loss)
        #loss = cross_entropy + regularization_loss
        return loss


    def get_trainop(self):
        tower_grads = []
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        # 将运算绑定在设备上
        reuse_variables = False
        for i,device in enumerate(self.device_list):
            print(device)
            with tf.device(device):    #name
                with tf.name_scope('GPU_%d' % i) as scope:
                    cur_loss = self._get_loss(x=self.input,
                                             y_=self.real_target,
                                             network=self.network,
                                             loss_function=self.loss,
                                             scope=scope,
                                             reuse_variables=reuse_variables)
                    reuse_variables = True
                    grads = self.train_op.compute_gradients(cur_loss)

                    tower_grads.append(grads)
        #计算总梯度
        grads = self.average_gradients(tower_grads=tower_grads)
        for grad, var in grads:
            if grad is not None:
                tf.histogram_summary('gradients_on_average/%s' % var.op.name, grad)
                # 使用平均梯度更新参数。
                apply_gradient_op = self.train_op.apply_gradients(grads, global_step=global_step)
                for var in tf.trainable_variables():
                    tf.histogram_summary(var.op.name, var)

                # 计算变量的滑动平均值。
                variable_averages = tf.train.ExponentialMovingAverage(self.MOVING_AVERAGE_DECAY, global_step)
                variables_to_average = (tf.trainable_variables() + tf.moving_average_variables())
                variables_averages_op = variable_averages.apply(variables_to_average)
                # 每一轮迭代需要更新变量的取值并更新变量的滑动平均值。
                train_op2 = tf.group(apply_gradient_op, variables_averages_op)
        return train_op2

    def get_total_loss(self):
        losses = tf.get_collection('losses',scope=None)
        return tf.add_n(losses,name='total_loss')










