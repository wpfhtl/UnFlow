import os
import copy

import tensorflow as tf
from tensorflow.python.client import device_lib

from src.e2eflow.core.train import Trainer
from src.e2eflow.experiment import Experiment
from src.e2eflow.util import convert_input_strings

from src.e2eflow.kitti.input import KITTIInput
from src.e2eflow.kitti.data import KITTIData
from src.e2eflow.chairs.data import ChairsData
from src.e2eflow.chairs.input import ChairsInput
from src.e2eflow.sintel.data import SintelData
from src.e2eflow.sintel.input import SintelInput
from src.e2eflow.synthia.data import SynthiaData
from src.e2eflow.cityscapes.data import CityscapesData
from src.e2eflow.driving.data import DrivingData
from src.e2eflow.driving.input import DrivingInput

tf.app.flags.DEFINE_string('ex', 'default',
                           'Name of the experiment.'
                           'If the experiment folder already exists in the log dir, '
                           'training will be continued from the latest checkpoint.')
tf.app.flags.DEFINE_boolean('debug', True,
                            'Enable image summaries and disable checkpoint writing for debugging.')
tf.app.flags.DEFINE_boolean('ow', False,
                            'Overwrites a previous experiment with the same name (if present)'
                            'instead of attempting to continue from its latest checkpoint.')
FLAGS = tf.app.flags.FLAGS


def main(argv=None):
    experiment = Experiment(
        name=FLAGS.ex,
        overwrite=FLAGS.ow)
    dirs = experiment.config['dirs']
    run_config = experiment.config['run']

    gpu_list_param = run_config['gpu_list']

    if isinstance(gpu_list_param, int):
        gpu_list = [gpu_list_param]
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_list_param)
    else:
        gpu_list = list(range(len(gpu_list_param.split(','))))
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list_param
    gpu_batch_size = int(run_config['batch_size'] / max(len(gpu_list), 1))
    devices = ['/gpu:' + str(gpu_num) for gpu_num in gpu_list]

    train_dataset = run_config.get('dataset', 'kitti')

    kdata = KITTIData(data_dir=dirs['data'],
                      fast_dir=dirs.get('fast'),
                      stat_log_dir=None,
                      development=run_config['development'])
    einput = KITTIInput(data=kdata,
                        batch_size=1,
                        normalize=False,
                        dims=(384, 1280))

    if train_dataset == 'chairs':
        cconfig = copy.deepcopy(experiment.config['train'])
        cconfig.update(experiment.config['train_chairs'])
        convert_input_strings(cconfig, dirs)
        citers = cconfig.get('num_iters', 0)
        cdata = ChairsData(data_dir=dirs['data'],
                           fast_dir=dirs.get('fast'),
                           stat_log_dir=None,
                           development=run_config['development'])
        cinput = ChairsInput(data=cdata,
                 batch_size=gpu_batch_size,
                 normalize=False,
                 dims=(cconfig['height'], cconfig['width']))
        tr = Trainer(
              lambda shift: cinput.input_raw(swap_images=False,
                                             shift=shift * run_config['batch_size']),
              lambda: einput.input_train_2012(),
              params=cconfig,
              normalization=cinput.get_normalization(),
              train_summaries_dir=experiment.train_dir,
              eval_summaries_dir=experiment.eval_dir,
              experiment=FLAGS.ex,
              ckpt_dir=experiment.save_dir,
              debug=FLAGS.debug,
              interactive_plot=run_config.get('interactive_plot'),
              devices=devices)
        tr.run(0, citers)

    elif train_dataset == 'kitti':
        kconfig = copy.deepcopy(experiment.config['train'])
        kconfig.update(experiment.config['train_kitti'])
        convert_input_strings(kconfig, dirs)
        kiters = kconfig.get('num_iters', 0)
        kinput = KITTIInput(data=kdata,
                            batch_size=gpu_batch_size,
                            normalize=False,
                            skipped_frames=True,
                            dims=(kconfig['height'], kconfig['width']))
        tr = Trainer(
              lambda shift: kinput.input_raw(swap_images=False,
                                             center_crop=True,
                                             shift=shift * run_config['batch_size']),
              lambda: einput.input_train_2012(),
              params=kconfig,
              normalization=kinput.get_normalization(),
              train_summaries_dir=experiment.train_dir,
              eval_summaries_dir=experiment.eval_dir,
              experiment=FLAGS.ex,
              ckpt_dir=experiment.save_dir,
              debug=FLAGS.debug,
              interactive_plot=run_config.get('interactive_plot'),
              devices=devices)
        tr.run(0, kiters)

    elif train_dataset == 'cityscapes':
        kconfig = copy.deepcopy(experiment.config['train'])
        kconfig.update(experiment.config['train_cityscapes'])
        convert_input_strings(kconfig, dirs)
        kiters = kconfig.get('num_iters', 0)
        cdata = CityscapesData(data_dir=dirs['data'],
                    fast_dir=dirs.get('fast'),
                    stat_log_dir=None,
                    development=run_config['development'])
        kinput = KITTIInput(data=cdata,
                            batch_size=gpu_batch_size,
                            normalize=False,
                            skipped_frames=False,
                            dims=(kconfig['height'], kconfig['width']))
        tr = Trainer(
              lambda shift: kinput.input_raw(swap_images=False,
                                             center_crop=True,
                                             skip=[0, 1],
                                             shift=shift * run_config['batch_size']),
              lambda: einput.input_train_2012(),
              params=kconfig,
              normalization=kinput.get_normalization(),
              train_summaries_dir=experiment.train_dir,
              eval_summaries_dir=experiment.eval_dir,
              experiment=FLAGS.ex,
              ckpt_dir=experiment.save_dir,
              debug=FLAGS.debug,
              interactive_plot=run_config.get('interactive_plot'),
              devices=devices)
        tr.run(0, kiters)

    elif train_dataset == 'synthia':
        sconfig = copy.deepcopy(experiment.config['train'])
        sconfig.update(experiment.config['train_synthia'])
        convert_input_strings(sconfig, dirs)
        siters = sconfig.get('num_iters', 0)
        sdata = SynthiaData(data_dir=dirs['data'],
                fast_dir=dirs.get('fast'),
                stat_log_dir=None,
                development=run_config['development'])
        sinput = KITTIInput(data=sdata,
                            batch_size=gpu_batch_size,
                            normalize=False,
                            dims=(sconfig['height'], sconfig['width']))
        tr = Trainer(
              lambda shift: sinput.input_raw(swap_images=False,
                                             shift=shift * run_config['batch_size']),
              lambda: einput.input_train_2012(),
              params=sconfig,
              normalization=sinput.get_normalization(),
              train_summaries_dir=experiment.train_dir,
              eval_summaries_dir=experiment.eval_dir,
              experiment=FLAGS.ex,
              ckpt_dir=experiment.save_dir,
              debug=FLAGS.debug,
              interactive_plot=run_config.get('interactive_plot'),
              devices=devices)
        tr.run(0, siters)

    elif train_dataset == 'kitti_ft':
        ftconfig = copy.deepcopy(experiment.config['train'])
        ftconfig.update(experiment.config['train_kitti_ft'])
        convert_input_strings(ftconfig, dirs)
        ftiters = ftconfig.get('num_iters', 0)
        ftinput = KITTIInput(data=kdata,
                             batch_size=gpu_batch_size,
                             normalize=False,
                             dims=(ftconfig['height'], ftconfig['width']))
        tr = Trainer(
              lambda shift: ftinput.input_train_gt(40),
              lambda: einput.input_train_2015(40),
              supervised=True,
              params=ftconfig,
              normalization=ftinput.get_normalization(),
              train_summaries_dir=experiment.train_dir,
              eval_summaries_dir=experiment.eval_dir,
              experiment=FLAGS.ex,
              ckpt_dir=experiment.save_dir,
              debug=FLAGS.debug,
              interactive_plot=run_config.get('interactive_plot'),
              devices=devices)
        tr.run(0, ftiters)

    ## Add

    elif train_dataset == 'driving':

        dconfig = copy.deepcopy(experiment.config['train'])

        dconfig.update(experiment.config['train_driving'])

        convert_input_strings(dconfig, dirs)

        diters = dconfig.get('num_iters', 0)

        ddata = DrivingData(data_dir=dirs['data'],

                            fast_dir=dirs.get('fast'),

                            stat_log_dir=None,

                            development=run_config['development'])

        dinput = DrivingInput(data=ddata,

                              batch_size=gpu_batch_size,

                              normalize=False,

                              dims=(dconfig['height'], dconfig['width']))

        tr = Trainer(

            lambda shift: dinput.input_raw(swap_images=False,

                                           shift=shift * run_config['batch_size']),

            lambda: einput.input_train_2012(),

            params=dconfig,

            normalization=dinput.get_normalization(),

            train_summaries_dir=experiment.train_dir,

            eval_summaries_dir=experiment.eval_dir,

            experiment=FLAGS.ex,

            ckpt_dir=experiment.save_dir,

            debug=FLAGS.debug,

            interactive_plot=run_config.get('interactive_plot'),

            devices=devices)

        tr.run(0, diters)

    ## Add
    elif train_dataset == 'sintel':

        stconfig = copy.deepcopy(experiment.config['train'])

        stconfig.update(experiment.config['train_sintel'])

        convert_input_strings(stconfig, dirs)

        stiters = stconfig.get('num_iters', 0)

        stdata = SintelData(data_dir=dirs['data'],

                            fast_dir=dirs.get('fast'),

                            stat_log_dir=None,

                            development=run_config['development'])

        stinput = SintelInput(data=stdata,

                              batch_size=gpu_batch_size,

                              normalize=False,

                              dims=(stconfig['height'], stconfig['width']))

        tr = Trainer(

            lambda shift: stinput.input_raw(swap_images=False,

                                           shift=shift * run_config['batch_size']),

            lambda: einput.input_train_2012(),

            params=stconfig,

            normalization=stinput.get_normalization(),

            train_summaries_dir=experiment.train_dir,

            eval_summaries_dir=experiment.eval_dir,

            experiment=FLAGS.ex,

            ckpt_dir=experiment.save_dir,

            debug=FLAGS.debug,

            interactive_plot=run_config.get('interactive_plot'),

            devices=devices)

        tr.run(0, stiters)

    ### Add

    elif train_dataset == 'driving_ft':

        dftconfig = copy.deepcopy(experiment.config['train'])

        dftconfig.update(experiment.config['train_driving_ft'])

        convert_input_strings(dftconfig, dirs)

        dftiters = dftconfig.get('num_iters', 0)

        dftdata = DrivingData(data_dir=dirs['data'],

                              fast_dir=dirs.get('fast'),

                              stat_log_dir=None,

                              development=run_config['development'])

        dftinput = DrivingInput(data=dftdata,

                                batch_size=gpu_batch_size,

                                normalize=False,

                                dims=(dftconfig['height'], dftconfig['width']))

        # Add deinput

        deinput = DrivingInput(data=dftdata,

                               batch_size=1,

                               normalize=False,

                               dims=(960, 540))  # The image size in driving dataset:960pixels*540pixels

        tr = Trainer(

            lambda shift: dftinput.input_train_ft(),

            lambda: deinput.input_test_ft(),

            supervised=True,

            params=dftconfig,

            normalization=dftinput.get_normalization(),

            train_summaries_dir=experiment.train_dir,

            eval_summaries_dir=experiment.eval_dir,

            experiment=FLAGS.ex,

            ckpt_dir=experiment.save_dir,

            debug=FLAGS.debug,

            interactive_plot=run_config.get('interactive_plot'),

            devices=devices)

        tr.run(0, dftiters)

    else:
      raise ValueError(
          "Invalid dataset. Dataset must be one of "
          "{synthia, kitti, kitti_ft, cityscapes, chairs}")

    if not FLAGS.debug:
        experiment.conclude()


if __name__ == '__main__':
    tf.app.run()