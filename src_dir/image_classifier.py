# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generic training script that trains a model using a given dataset."""

from __future__ import absolute_import, division, print_function

import argparse
import codecs
import glob
import io
import json
import logging
import math
import os
import re
import subprocess
import sys

import numpy as np
import PIL

import freeze_graph as fg
import tensorflow as tf
from datasets import dataset_factory
from deployment import model_deploy
from nets import nets_factory
from preprocessing import preprocessing_factory
from tensorflow.contrib import quantize as contrib_quantize
from tensorflow.contrib import slim as contrib_slim
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.platform import gfile

slim = contrib_slim


def parse_args():
    parser = argparse.ArgumentParser()

    ###############################
    # SageMaker Default Arguments #
    ###############################
    parser.add_argument('--model_dir', type=str,
                        default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train_dir', type=str, default=os.environ.get('SM_MODEL_DIR'),
                        help='Directory where checkpoints and event logs are written to.')
    parser.add_argument('--dataset_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--hosts', type=list,
                        default=json.loads(os.environ.get('SM_HOSTS')))
    parser.add_argument('--current-host', type=str,
                        default=os.environ.get('SM_CURRENT_HOST'))
    parser.add_argument('--data-config', type=json.loads,
                        default=os.environ.get('SM_INPUT_DATA_CONFIG'))
    parser.add_argument('--output_data_dir', type=str,
                        default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    parser.add_argument('--num_gpus', type=str,
                        default=os.environ.get('SM_NUM_GPUS'))
    parser.add_argument('--output-dir', type=str,
                        default=os.environ.get('SM_OUTPUT_DIR'))
    parser.add_argument('--fw-params', type=json.loads,
                        default=os.environ.get('SM_FRAMEWORK_PARAMS'))

    ###############################
    #  Default Arguments #
    ###############################
    parser.add_argument('--master', type=str, default='',
                        help='The address of the TensorFlow master to use.')
    parser.add_argument('--warmup_epochs', type=float, default=0,
                        help='Linearly warmup learning rate from 0 to learning_rate over this many epochs.')
    parser.add_argument('--num_clones', type=int, default=1,
                        help='Number of model clones to deploy. Note For historical reasons'
                        'loss from all clones averaged out and learning rate decay happen per clone epochs')
    parser.add_argument('--clone_on_cpu', type=bool,
                        default=False, help='Use CPUs to deploy clones.')
    parser.add_argument('--worker_replicas', type=int,
                        default=1, help='Number of worker replicas.')
    parser.add_argument('--num_ps_tasks', type=int, default=0,
                        help='The number of parameter servers. If the value is 0,'
                        'then the parameters are handled locally by the worker.')
    parser.add_argument('--task', type=int, default=0,
                        help='Task id of the replica running the training.')
    parser.add_argument('--save_interval_secs', type=int, default=600,
                        help='The frequency with which the model is saved, in seconds.')
    parser.add_argument('--save_summaries_secs', type=int, default=600,
                        help='The frequency with which summaries are saved, in seconds.')
    parser.add_argument('--log_every_n_steps', type=int, default=10,
                        help='The frequency with which logs are print.')
    parser.add_argument('--num_preprocessing_threads', type=int, default=4,
                        help='The number of threads used to create the batches.')
    parser.add_argument('--num_readers', type=int, default=4,
                        help='The number of parallel readers that read data from the dataset.')

    ##########################
    # Optimization Arguments #
    ##########################
    parser.add_argument('--adam_beta1', type=float, default=0.9,
                        help='The exponential decay rate for the 1st moment estimates.')
    parser.add_argument('--adam_beta2', type=float, default=0.999,
                        help='The exponential decay rate for the 2nd moment estimates.')
    parser.add_argument('--adagrad_initial_accumulator_value', type=float,
                        default=0.1, help='Starting value for the AdaGrad accumulators.')
    parser.add_argument('--adadelta_rho', type=float,
                        default=0.95, help='The decay rate for adadelta.')
    parser.add_argument('--optimizer', type=str, default='rmsprop',
                        help='The name of the optimizer, one of "adadelta", "adagrad",'
                        '"adam","ftrl", "momentum", "sgd" or "rmsprop".')
    parser.add_argument('--weight_decay', type=float, default=0.00004,
                        help='The weight decay on the model weights.')
    parser.add_argument('--opt_epsilon', type=float,
                        default=1.0, help='Epsilon term for the optimizer.')
    parser.add_argument('--ftrl_learning_rate_power', type=float,
                        default=-0.5, help='The learning rate power.')
    parser.add_argument('--ftrl_initial_accumulator_value', type=float,
                        default=0.1, help='Starting value for the FTRL accumulators.')
    parser.add_argument('--ftrl_l1', type=float, default=0.0,
                        help='The FTRL l1 regularization strength.')
    parser.add_argument('--ftrl_l2', type=float, default=0.0,
                        help='The FTRL l2 regularization strength.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='The momentum for the MomentumOptimizer and RMSPropOptimizer.')
    parser.add_argument('--rmsprop_momentum', type=float,
                        default=0.9, help='Momentum.')
    parser.add_argument('--rmsprop_decay', type=float,
                        default=0.9, help='Decay term for RMSProp.')
    parser.add_argument('--quantize_delay', type=int, default=-1,
                        help='Number of steps to start quantized training.'
                        'Set to -1 would disable quantized training.')

    ###########################
    # Learning Rate Arguments #
    ###########################
    parser.add_argument('--learning_rate_decay_type', type=str, default='exponential',
                        help='Specifies how the learning rate is decayed. One of "fixed", "exponential" or "polynomial"')
    parser.add_argument('--learning_rate', type=float,
                        default=0.01, help='Initial learning rate.')
    parser.add_argument('--end_learning_rate', type=float,
                        default=0.01, help='Initial learning rate.')
    parser.add_argument('--label_smoothing', type=float,
                        default=0.0, help='The amount of label smoothing.')
    parser.add_argument('--learning_rate_decay_factor', type=float,
                        default=0.94, help='Learning rate decay factor.')
    parser.add_argument('--num_epochs_per_decay', type=float, default=2.0,
                        help='Number of epochs after which learning rate decays. '
                        'Note: this flag counts epochs per clone but aggregates per sync replicas.'
                        'So 1.0 means that each clone will go over full epoch individually, '
                        'but replicas will go once across all replicas.')
    parser.add_argument('--sync_replicas', type=bool,
                        default=False, help='Whether or not to synchronize the replicas during training.')
    parser.add_argument('--replicas_to_aggregate', type=int,
                        default=1, help='The Number of gradients to collect before updating params.')
    parser.add_argument('--moving_average_decay', type=float,
                        default=None, help='The decay to use for the moving average.'
                        ' If left as None, then moving averages are not used.')

    #####################
    # Dataset Arguments #
    #####################
    parser.add_argument('--dataset_name', type=str,
                        default='imagenet', help='The name of the dataset to load.')
    # parser.add_argument('--dataset_split_name', type=str,
    #                     default='train', help='The name of the train/test split.')
    parser.add_argument('--labels_offset', type=int,
                        default=0, help='An offset for the labels in the dataset.'
                        'This flag is primarily used to evaluate the VGG and ResNet '
                        'architectures which do not use a background class for the ImageNet dataset.')
    parser.add_argument('--model_name', type=str,
                        default='inception_v3', help='The name of the architecture to train.')
    parser.add_argument('--preprocessing_name', type=str,
                        default=None, help='The name of the preprocessing to use.'
                        'If left as `None`, then the model_name flag is used.')
    parser.add_argument('--batch_size', type=int,
                        default=32, help='The number of samples in each batch.')
    parser.add_argument('--image_size', type=int,
                        default=None, help='Train image size')
    parser.add_argument('--max_number_of_steps', type=int,
                        default=None, help='The maximum number of training steps.')
    parser.add_argument('--use_grayscale', type=bool, default=False,
                        help='Whether to convert input images to grayscale.')

    #########################
    # Fine-Tuning Arguments #
    #########################
    parser.add_argument('--finetune_checkpoint_path', type=str,
                        default=None, help='The path to a checkpoint from which to fine-tune.')
    parser.add_argument('--checkpoint_exclude_scopes', type=str,
                        default=None, help='Comma-separated list of scopes of variables'
                        'to exclude when restoring from a checkpoint.')
    parser.add_argument('--trainable_scopes', type=str,
                        default=None, help='Comma-separated list of scopes to filter the set'
                        'of variables to train. By default, None would train all the variables.')
    parser.add_argument('--ignore_missing_vars', type=bool,
                        default=False, help='When restoring a checkpoint would ignore missing variables.')

    #########################
    # evaluation Arguments #
    #########################
    parser.add_argument('--eval_batch_size', type=int,
                        default=100, help='The number of samples in each batch in evaluation.')
    parser.add_argument('--train_num_data', type=int,
                        help='The number of train samples')
    parser.add_argument('--test_num_data', type=int,
                        help='The number of test samples')
    parser.add_argument('--max_eval_num_batches', type=int,
                        default=None, help='Max number of batches to evaluate by default use all.')
    # parser.add_argument('--num_preprocessing_threads', type=int,
    #                     default=4, help='The number of threads used to create the batches.')
    # parser.add_argument('--eval_image_size', type=int,
    #                     default=None, help='Eval image size')
    parser.add_argument('--quantize', type=bool,
                        default=False, help='Eval image size')

    parser.add_argument('--is_training', type=bool,
                        default=False, help='Whether to save out a training-focused version of the model.')

    parser.add_argument('--is_video_model', type=bool,
                        default=False, help='whether to use 5-D inputs for video model.')

    parser.add_argument('--num_frames', type=int,
                        default=None, help='The number of frames to use. Only used if is_video_model is True.')
    parser.add_argument('--write_text_graphdef', type=bool,
                        default=False, help='Whether to write a text version of graphdef.')
    return_value = parser.parse_known_args()
    print("parser.parse_known_args() : {}".format(return_value))
    return return_value


def _configure_learning_rate(args, num_samples_per_epoch, global_step):
    """Configures the learning rate.

    Args:
      num_samples_per_epoch: The number of samples in each epoch of training.
      global_step: The global_step tensor.

    Returns:
      A `Tensor` representing the learning rate.

    Raises:
      ValueError: if
    """
    # Note: when num_clones is > 1, this will actually have each clone to go
    # over each epoch args.num_epochs_per_decay times. This is different
    # behavior from sync replicas and is expected to produce different results.
    steps_per_epoch = num_samples_per_epoch / args.batch_size
    if args.sync_replicas:
        steps_per_epoch /= args.replicas_to_aggregate

    decay_steps = int(steps_per_epoch * args.num_epochs_per_decay)

    if args.learning_rate_decay_type == 'exponential':
        learning_rate = tf.train.exponential_decay(
            args.learning_rate,
            global_step,
            decay_steps,
            args.learning_rate_decay_factor,
            staircase=True,
            name='exponential_decay_learning_rate')
    elif args.learning_rate_decay_type == 'fixed':
        learning_rate = tf.constant(
            args.learning_rate, name='fixed_learning_rate')
    elif args.learning_rate_decay_type == 'polynomial':
        learning_rate = tf.train.polynomial_decay(
            args.learning_rate,
            global_step,
            decay_steps,
            args.end_learning_rate,
            power=1.0,
            cycle=False,
            name='polynomial_decay_learning_rate')
    else:
        raise ValueError('learning_rate_decay_type [%s] was not recognized' %
                         args.learning_rate_decay_type)

    if args.warmup_epochs:
        warmup_lr = (
            args.learning_rate * tf.cast(global_step, tf.float32) /
            (steps_per_epoch * args.warmup_epochs))
        learning_rate = tf.minimum(warmup_lr, learning_rate)
    return learning_rate


def _configure_optimizer(args, learning_rate):
    """Configures the optimizer used for training.

    Args:
      learning_rate: A scalar or `Tensor` learning rate.

    Returns:
      An instance of an optimizer.

    Raises:
      ValueError: if args.optimizer is not recognized.
    """
    if args.optimizer == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(
            learning_rate,
            rho=args.adadelta_rho,
            epsilon=args.opt_epsilon)
    elif args.optimizer == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(
            learning_rate,
            initial_accumulator_value=args.adagrad_initial_accumulator_value)
    elif args.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(
            learning_rate,
            beta1=args.adam_beta1,
            beta2=args.adam_beta2,
            epsilon=args.opt_epsilon)
    elif args.optimizer == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(
            learning_rate,
            learning_rate_power=args.ftrl_learning_rate_power,
            initial_accumulator_value=args.ftrl_initial_accumulator_value,
            l1_regularization_strength=args.ftrl_l1,
            l2_regularization_strength=args.ftrl_l2)
    elif args.optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(
            learning_rate,
            momentum=args.momentum,
            name='Momentum')
    elif args.optimizer == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate,
            decay=args.rmsprop_decay,
            momentum=args.rmsprop_momentum,
            epsilon=args.opt_epsilon)
    elif args.optimizer == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    else:
        raise ValueError('Optimizer [%s] was not recognized' % args.optimizer)
    return optimizer


def _get_init_fn(args):
    """Returns a function run by the chief worker to warm-start the training.

    Note that the init_fn is only run when initializing the model during the very
    first global step.

    Returns:
      An init function run by the supervisor.
    """
    if args.finetune_checkpoint_path is None:
        return None

    # Warn the user if a checkpoint exists in the train_dir. Then we'll be
    # ignoring the checkpoint anyway.
    # if tf.train.latest_checkpoint(args.finetune_checkpoint_path):
    #     tf.logging.info(
    #         'Ignoring --checkpoint_path because a checkpoint already exists in %s'
    #         % args.finetune_checkpoint_path)
    #     return None

    exclusions = []
    if args.checkpoint_exclude_scopes:
        exclusions = [scope.strip()
                      for scope in args.checkpoint_exclude_scopes.split(',')]

    # TODO(sguada) variables.filter_variables()
    variables_to_restore = []
    for var in slim.get_model_variables():
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                break
        else:
            variables_to_restore.append(var)

    if tf.gfile.IsDirectory(args.finetune_checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(
            args.finetune_checkpoint_path)
    else:
        checkpoint_path = args.finetune_checkpoint_path

    tf.logging.info('Fine-tuning from %s' % checkpoint_path)

    return slim.assign_from_checkpoint_fn(
        checkpoint_path,
        variables_to_restore,
        ignore_missing_vars=args.ignore_missing_vars)


def _get_variables_to_train(args):
    """Returns a list of variables to train.

    Returns:
      A list of variables to train by the optimizer.
    """
    if args.trainable_scopes is None:
        return tf.trainable_variables()
    else:
        scopes = [scope.strip() for scope in args.trainable_scopes.split(',')]

    variables_to_train = []
    for scope in scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.extend(variables)
    return variables_to_train


def evaluation(args):
    if not args.dataset_dir:
        raise ValueError(
            'You must supply the dataset directory with --dataset_dir')

    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
        tf_global_step = slim.get_or_create_global_step()

        ######################
        # Select the dataset #
        ######################
        dataset = dataset_factory.get_dataset(
            args.dataset_name, 'val', args.dataset_dir, args.test_num_data)

        ####################
        # Select the model #
        ####################
        network_fn = nets_factory.get_network_fn(
            args.model_name,
            num_classes=(dataset.num_classes - args.labels_offset),
            is_training=False)

        ##############################################################
        # Create a dataset provider that loads data from the dataset #
        ##############################################################
        provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            shuffle=False,
            common_queue_capacity=2 * args.eval_batch_size,
            common_queue_min=args.eval_batch_size)
        [image, label] = provider.get(['image', 'label'])
        label -= args.labels_offset

        #####################################
        # Select the preprocessing function #
        #####################################
        preprocessing_name = args.preprocessing_name or args.model_name
        
        print("eval_args.use_grayscale : {}".format(args.use_grayscale))
        
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            preprocessing_name,
            is_training=False,
            use_grayscale=args.use_grayscale)

        eval_image_size = args.image_size or network_fn.default_image_size

        image = image_preprocessing_fn(image, eval_image_size, eval_image_size)

        images, labels = tf.train.batch(
            [image, label],
            batch_size=args.eval_batch_size,
            num_threads=args.num_preprocessing_threads,
            capacity=5 * args.eval_batch_size)

        ####################
        # Define the model #
        ####################
        logits, _ = network_fn(images)

        if args.quantize:
            contrib_quantize.create_eval_graph()

        if args.moving_average_decay:
            variable_averages = tf.train.ExponentialMovingAverage(
                args.moving_average_decay, tf_global_step)
            variables_to_restore = variable_averages.variables_to_restore(
                slim.get_model_variables())
            variables_to_restore[tf_global_step.op.name] = tf_global_step
        else:
            variables_to_restore = slim.get_variables_to_restore()

        predictions = tf.argmax(logits, 1)
        labels = tf.squeeze(labels)

        # Define the metrics:
        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
            'Accuracy': slim.metrics.streaming_accuracy(predictions, labels),
            'Recall_5': slim.metrics.streaming_recall_at_k(
                logits, labels, 5),
        })

        # Print the summaries to screen.
        for name, value in names_to_values.items():
            summary_name = 'eval/%s' % name
            op = tf.summary.scalar(summary_name, value, collections=[])
            op = tf.Print(op, [value], summary_name)
            tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

        # TODO(sguada) use num_epochs=1
        if args.max_eval_num_batches:
            num_batches = args.max_eval_num_batches
        else:
            # This ensures that we make a single pass over all of the data.
            num_batches = math.ceil(
                dataset.num_samples / float(args.eval_batch_size))

        if tf.gfile.IsDirectory(args.train_dir):
            checkpoint_path = tf.train.latest_checkpoint(args.train_dir)
        else:
            checkpoint_path = args.train_dir

        tf.logging.info('Evaluating %s' % checkpoint_path)

        slim.evaluation.evaluate_once(
            master=args.master,
            checkpoint_path=checkpoint_path,
            logdir=args.train_dir,
            num_evals=num_batches,
            eval_op=list(names_to_updates.values()),
            variables_to_restore=variables_to_restore)


def export_inference_graph(args):
    if not args.train_dir:
        raise ValueError(
            'You must supply the path to save to with --train_dir')
    if args.is_video_model and not args.num_frames:
        raise ValueError(
            'Number of frames must be specified for video models with --num_frames')
    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default() as graph:
        dataset = dataset_factory.get_dataset(args.dataset_name, 'train',
                                              args.dataset_dir, args.train_num_data)
        network_fn = nets_factory.get_network_fn(
            args.model_name,
            num_classes=(dataset.num_classes - args.labels_offset),
            is_training=args.is_training)
        image_size = args.image_size or network_fn.default_image_size
        num_channels = 1 if args.use_grayscale else 3
        if args.is_video_model:
            input_shape = [
                1, args.num_frames, image_size, image_size,
                num_channels
            ]
        else:
            input_shape = [1,
                           image_size, image_size, num_channels]
        placeholder = tf.placeholder(name='input', dtype=tf.float32,
                                     shape=input_shape)
        network_fn(placeholder)

        if args.quantize:
            contrib_quantize.create_eval_graph()

        graph_def = graph.as_graph_def()
        if args.write_text_graphdef:
            tf.io.write_graph(
                graph_def,
                os.path.dirname(args.train_dir + '/inference_graph.pb'),
                os.path.basename(args.train_dir),
                as_text=True)
        else:
            with gfile.GFile(args.train_dir + '/inference_graph.pb', 'wb') as f:
                f.write(graph_def.SerializeToString())


def freeze_graph(args):
    checkpoint_version = saver_pb2.SaverDef.V2
    input_graph = args.train_dir + '/inference_graph.pb'
    input_checkpoint = tf.train.latest_checkpoint(args.train_dir)
    input_binary = True
    output_graph = args.train_dir + '/inference_graph_frozen.pb'
    output_node_names = 'MobilenetV1/Predictions/Reshape_1'
    input_saved_model_dir = ""
    saved_model_tags = "serve"
    input_meta_graph = ""
    variable_names_blacklist = ""
    variable_names_whitelist = ""
    initializer_nodes = ""
    clear_devices = True
    filename_tensor_name = "save/Const:0"
    restore_op_name = "save/restore_all"
    input_saver = ""
    print("freeze_graph input_checkpoint : {}".format(input_checkpoint))

    fg.freeze_graph(input_graph, input_saver, input_binary,
                    input_checkpoint, output_node_names,
                    restore_op_name, filename_tensor_name,
                    output_graph, clear_devices, initializer_nodes,
                    variable_names_whitelist, variable_names_blacklist,
                    input_meta_graph, input_saved_model_dir,
                    saved_model_tags, checkpoint_version)


def main():
    args, unknown = parse_args()
    print("********************* args : {}".format(args))
    print("********************* unknown : {} ".format(unknown))
    print("********************* args.use_grayscale : {}".format(args.use_grayscale))

    print("********************* args.model_dir : {}".format(args.model_dir))
    print("********************* args.train_dir : {}".format(args.train_dir))
    if not args.dataset_dir:
        raise ValueError(
            'You must supply the dataset directory with --dataset_dir')

    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
        #######################
        # Config model_deploy #
        #######################
        deploy_config = model_deploy.DeploymentConfig(
            num_clones=args.num_clones,
            clone_on_cpu=args.clone_on_cpu,
            replica_id=args.task,
            num_replicas=args.worker_replicas,
            num_ps_tasks=args.num_ps_tasks)

        # Create global_step
        with tf.device(deploy_config.variables_device()):
            global_step = slim.create_global_step()

        ######################
        # Select the dataset #
        ######################
        dataset = dataset_factory.get_dataset(
            args.dataset_name, 'train', args.dataset_dir, args.train_num_data)

        ######################
        # Select the network #
        ######################
        network_fn = nets_factory.get_network_fn(
            args.model_name,
            num_classes=(dataset.num_classes - args.labels_offset),
            weight_decay=args.weight_decay,
            is_training=True)

        #####################################
        # Select the preprocessing function #
        #####################################
        preprocessing_name = args.preprocessing_name or args.model_name
        
        print("train_args.use_grayscale : {}".format(args.use_grayscale))
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            preprocessing_name,
            is_training=True,
            use_grayscale=args.use_grayscale)

        ##############################################################
        # Create a dataset provider that loads data from the dataset #
        ##############################################################
        with tf.device(deploy_config.inputs_device()):
            provider = slim.dataset_data_provider.DatasetDataProvider(
                dataset,
                num_readers=args.num_readers,
                common_queue_capacity=20 * args.batch_size,
                common_queue_min=10 * args.batch_size)
            [image, label] = provider.get(['image', 'label'])
            label -= args.labels_offset

            train_image_size = args.image_size or network_fn.default_image_size

            image = image_preprocessing_fn(
                image, train_image_size, train_image_size)

            images, labels = tf.train.batch(
                [image, label],
                batch_size=args.batch_size,
                num_threads=args.num_preprocessing_threads,
                capacity=5 * args.batch_size)
            labels = slim.one_hot_encoding(
                labels, dataset.num_classes - args.labels_offset)
            batch_queue = slim.prefetch_queue.prefetch_queue(
                [images, labels], capacity=2 * deploy_config.num_clones)

        ####################
        # Define the model #
        ####################
        def clone_fn(args, batch_queue):
            """Allows data parallelism by creating multiple clones of network_fn."""
            images, labels = batch_queue.dequeue()
            logits, end_points = network_fn(images)

            #############################
            # Specify the loss function #
            #############################
            if 'AuxLogits' in end_points:
                slim.losses.softmax_cross_entropy(
                    end_points['AuxLogits'], labels,
                    label_smoothing=args.label_smoothing, weights=0.4,
                    scope='aux_loss')
            slim.losses.softmax_cross_entropy(
                logits, labels, label_smoothing=args.label_smoothing, weights=1.0)
            return end_points

        # Gather initial summaries.
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

        clones = model_deploy.create_clones(
            deploy_config, clone_fn, [args, batch_queue])
        first_clone_scope = deploy_config.clone_scope(0)
        # Gather update_ops from the first clone. These contain, for example,
        # the updates for the batch_norm variables created by network_fn.
        update_ops = tf.get_collection(
            tf.GraphKeys.UPDATE_OPS, first_clone_scope)

        # Add summaries for end_points.
        end_points = clones[0].outputs
        for end_point in end_points:
            x = end_points[end_point]
            summaries.add(tf.summary.histogram('activations/' + end_point, x))
            summaries.add(tf.summary.scalar('sparsity/' + end_point,
                                            tf.nn.zero_fraction(x)))

        # Add summaries for losses.
        for loss in tf.get_collection(tf.GraphKeys.LOSSES, first_clone_scope):
            summaries.add(tf.summary.scalar('losses/%s' % loss.op.name, loss))

        # Add summaries for variables.
        for variable in slim.get_model_variables():
            summaries.add(tf.summary.histogram(variable.op.name, variable))

        #################################
        # Configure the moving averages #
        #################################
        if args.moving_average_decay:
            moving_average_variables = slim.get_model_variables()
            variable_averages = tf.train.ExponentialMovingAverage(
                args.moving_average_decay, global_step)
        else:
            moving_average_variables, variable_averages = None, None

        if args.quantize_delay >= 0:
            contrib_quantize.create_training_graph(
                quant_delay=args.quantize_delay)

        #########################################
        # Configure the optimization procedure. #
        #########################################
        with tf.device(deploy_config.optimizer_device()):
            learning_rate = _configure_learning_rate(args,
                                                     dataset.num_samples, global_step)
            optimizer = _configure_optimizer(args, learning_rate)
            summaries.add(tf.summary.scalar('learning_rate', learning_rate))

        if args.sync_replicas:
            # If sync_replicas is enabled, the averaging will be done in the chief
            # queue runner.
            optimizer = tf.train.SyncReplicasOptimizer(
                opt=optimizer,
                replicas_to_aggregate=args.replicas_to_aggregate,
                total_num_replicas=args.worker_replicas,
                variable_averages=variable_averages,
                variables_to_average=moving_average_variables)
        elif args.moving_average_decay:
            # Update ops executed locally by trainer.
            update_ops.append(variable_averages.apply(
                moving_average_variables))

        # Variables to train.
        variables_to_train = _get_variables_to_train(args)

        #  and returns a train_tensor and summary_op
        total_loss, clones_gradients = model_deploy.optimize_clones(
            clones,
            optimizer,
            var_list=variables_to_train)
        # Add total_loss to summary.
        summaries.add(tf.summary.scalar('total_loss', total_loss))

        # Create gradient updates.
        grad_updates = optimizer.apply_gradients(clones_gradients,
                                                 global_step=global_step)
        update_ops.append(grad_updates)

        update_op = tf.group(*update_ops)
        with tf.control_dependencies([update_op]):
            train_tensor = tf.identity(total_loss, name='train_op')

        # Add the summaries from the first clone. These contain the summaries
        # created by model_fn and either optimize_clones() or _gather_clone_loss().
        summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES,
                                           first_clone_scope))

        # Merge all summaries together.
        summary_op = tf.summary.merge(list(summaries), name='summary_op')

        ###########################
        # Kicks off the training. #
        ###########################
        slim.learning.train(
            train_tensor,
            logdir=args.train_dir,
            master=args.master,
            is_chief=(args.task == 0),
            init_fn=_get_init_fn(args),
            summary_op=summary_op,
            number_of_steps=args.max_number_of_steps,
            log_every_n_steps=args.log_every_n_steps,
            save_summaries_secs=args.save_summaries_secs,
            save_interval_secs=args.save_interval_secs,
            sync_optimizer=optimizer if args.sync_replicas else None)

    evaluation(args)
    export_inference_graph(args)
    freeze_graph(args)

    converter = tf.lite.TFLiteConverter.from_frozen_graph(
        args.train_dir + '/inference_graph_frozen.pb', ['input'], ['MobilenetV1/Predictions/Reshape_1'])
    tflite_model = converter.convert()
    open(args.train_dir + "/mobilenetv1_model.tflite",
         "wb").write(tflite_model)


if __name__ == '__main__':
    main()
