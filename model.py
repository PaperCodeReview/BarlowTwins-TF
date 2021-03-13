import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Lambda
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import Constant

from resnet import ResNet18
from layer import _conv2d
from layer import _batchnorm
from layer import _dense
from augment import Augment


MODEL_DICT = {
    'resnet18' : ResNet18,
    'resnet50' : tf.keras.applications.ResNet50,}
FAMILY_DICT = {
    'resnet18' : tf.python.keras.applications.resnet,
    'resnet50' : tf.python.keras.applications.resnet,}


def set_lincls(args, backbone):
    DEFAULT_ARGS = {
        "use_bias": args.use_bias,
        "kernel_regularizer": l2(args.weight_decay)}
    
    if args.freeze:
        backbone.trainable = False
        
    x = backbone.get_layer(name='avg_pool').output
    x = _dense(**DEFAULT_ARGS)(args.classes, name='predictions')(x)
    model = tf.keras.Model(backbone.input, x, name='lincls')
    return model


class BarlowTwins(tf.keras.Model):
    def __init__(self, args, logger, num_workers=1, **kwargs):
        super(BarlowTwins, self).__init__(**kwargs)
        self.args = args
        self._num_workers = num_workers
        norm = 'bn' if self._num_workers == 1 else 'syncbn'

        # preprocess
        augment = Augment(args)
        self.preprocess = tf.keras.Sequential(name='preprocess')
        self.preprocess.add(Lambda(lambda x: augment._random_color_jitter(x)))
        self.preprocess.add(Lambda(lambda x: augment._random_grayscale(x)))
        if self.args.dataset == 'imagenet':
            self.preprocess.add(Lambda(lambda x: augment._random_gaussian_blur(x)))
        self.preprocess.add(Lambda(lambda x: augment._random_hflip(x)))
        self.preprocess.add(Lambda(lambda x: augment._standardize(x)))

        # encoder
        DEFAULT_ARGS = {
            "use_bias": self.args.use_bias,
            "kernel_regularizer": l2(self.args.weight_decay)}
        FAMILY_DICT[self.args.backbone].Conv2D = _conv2d(**DEFAULT_ARGS)
        FAMILY_DICT[self.args.backbone].BatchNormalization = _batchnorm(norm=norm)
        FAMILY_DICT[self.args.backbone].Dense = _dense(**DEFAULT_ARGS)

        DEFAULT_ARGS.update({'norm': norm}) # for resnet18
        self.encoder = MODEL_DICT[self.args.backbone](
            include_top=False,
            weights=None,
            input_shape=(self.args.img_size, self.args.img_size, 3),
            pooling='avg',
            **DEFAULT_ARGS if self.args.backbone == 'resnet18' else {})
        DEFAULT_ARGS.pop('norm') # for resnet18

        # projector
        num_mlp = 3
        self.projector = tf.keras.Sequential(name='projector')
        for i in range(num_mlp-1):
            self.projector.add(_dense(**DEFAULT_ARGS)(self.args.proj_dim, name=f'proj_fc{i+1}'))
            self.projector.add(_batchnorm(norm=norm)(epsilon=1.001e-5, name=f'proj_bn{i+1}'))
            self.projector.add(Activation('relu', name=f'proj_relu{i+1}'))

        self.projector.add(_dense(**DEFAULT_ARGS)(self.args.proj_dim, name=f'proj_fc{i+2}'))

    def call(self, x):
        y = self.preprocess(x)
        e = self.encoder(y)
        z = self.projector(e)
        return x, y, e, z

    def compile(
        self,
        optimizer,
        loss,
        run_eagerly=None):

        super(BarlowTwins, self).compile(
            optimizer=optimizer, run_eagerly=run_eagerly)

        self._loss = loss        

    def train_step(self, data):
        img1, img2 = data
        with tf.GradientTape() as tape:
            _, _, _, z_a = self(img1, training=True)
            _, _, _, z_b = self(img2, training=True)

            N = tf.shape(z_a)[0]
            D = tf.shape(z_a)[1]
            
            # normalize repr. along the batch dimension
            z_a_norm = (z_a - tf.reduce_mean(z_a, axis=0)) / tf.math.reduce_std(z_a, axis=0) # (b, i)
            z_b_norm = (z_b - tf.reduce_mean(z_b, axis=0)) / tf.math.reduce_std(z_b, axis=0) # (b, j)

            # cross-correlation matrix
            c_ij = tf.einsum(
                'bi,bj->ij', 
                tf.math.l2_normalize(z_a_norm, axis=0), 
                tf.math.l2_normalize(z_b_norm, axis=0)) / tf.cast(N, tf.float32) # (i, j)

            # loss
            eye = tf.eye(D, D, dtype=tf.float32)
            loss_invariance = tf.reduce_sum(tf.square(1. - tf.boolean_mask(c_ij, tf.cast(eye, tf.bool))))
            loss_reduction = tf.reduce_sum(tf.square(tf.boolean_mask(c_ij, tf.cast(1-eye, tf.bool))))

            loss_barlowtwins = loss_invariance + self.args.loss_weight * loss_reduction
            loss_decay = sum(self.losses)

            loss = loss_barlowtwins + loss_decay
            total_loss = loss / self._num_workers

        trainable_vars = self.trainable_variables
        grads = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))

        results = {
            'loss': loss, 
            'loss_barlowtwins': loss_barlowtwins, 
            'loss_decay': loss_decay, 
            'loss_invariance': loss_invariance,
            'loss_reduction': loss_reduction}
        return results