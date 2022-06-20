import time
import cv2 as cv
import skvideo.io as vio
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from pathlib import Path
from collections import OrderedDict
from trackron.utils.visdom import Visdom
from trackron.utils.plotting import draw_figure
from .build import build_tracker

_tracker_disp_colors = {
    1: (0, 255, 0),
    2: (0, 0, 255),
    3: (255, 0, 0),
    4: (255, 255, 255),
    5: (0, 0, 0),
    6: (0, 255, 128),
    7: (123, 123, 123),
    8: (255, 128, 0),
    9: (128, 0, 255)
}


def trackerlist(name,
                parameter_name,
                net,
                run_ids=None,
                display_name: str = None):
  """Generate list of trackers.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_ids: A single or list of run_ids.
        display_name: Name to be displayed in the result plots.
    """
  if run_ids is None or isinstance(run_ids, int):
    run_ids = [run_ids]
  return [
      TrackingActor(name, parameter_name, net, run_id, display_name)
      for run_id in run_ids
  ]


class TrackingActor:
  """Wraps the tracker for evaluation and running purposes.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_id: The run id.
        display_name: Name to be displayed in the result plots.
    """

  def __init__(self,
               cfg,
               net,
               run_id: int = None,
               display_name: str = None,
               output_dir: str = "./output",
               tracking_mode: str = 'sot',
               debug_level: int = -1,
               tracking_category: int = None):
    assert run_id is None or isinstance(run_id, int)

    self.cfg = cfg
    self.name = cfg.TRACKER.NAME
    self.run_id = run_id
    self.display_name = display_name
    self.output_dir = Path(output_dir)
    self.tracking_mode = tracking_mode
    self.tracker = build_tracker(cfg, net, tracking_mode)
    self.debug_level = debug_level
    self.tracking_category = tracking_category ### only effective for mot mode

    if self.run_id is None:
      self.results_dir = self.output_dir
      self.segmentation_dir = self.output_dir
    else:
      self.results_dir = self.output_dir / "{:03d}".format(self.run_id)
      self.segmentation_dir = self.output_dir / "{}_{:03d}".format(self.run_id)

    self.visdom = None
    if debug_level > 1:
      self._init_visdom({}, debug_level)
    self.pause_mode = False

  def _init_visdom(self, visdom_info, debug):
    visdom_info = {} if visdom_info is None else visdom_info
    self.pause_mode = False
    self.step = False
    if debug > 0 and visdom_info.get('use_visdom', True):
      try:
        self.visdom = Visdom(debug, {
            'handler': self._visdom_ui_handler,
            'win_id': 'Tracking'
        },
                             visdom_info=visdom_info)

        # Show help
        help_text = 'You can pause/unpause the tracker by pressing ''space'' with the ''Tracking'' window ' \
                    'selected. During paused mode, you can track for one frame by pressing the right arrow key.' \
                    'To enable/disable plotting of a data block, tick/untick the corresponding entry in ' \
                    'block list.'
        self.visdom.register(help_text, 'text', 1, 'Help')
      except:
        time.sleep(0.5)
        print(
            '!!! WARNING: Visdom could not start, so using matplotlib visualization instead !!!\n'
            '!!! Start Visdom in a separate terminal window by typing \'visdom\' !!!'
        )
    self.tracker.visdom = self.visdom
    self.tracker.debug_level = debug

  def _visdom_ui_handler(self, data):
    if data['event_type'] == 'KeyPress':
      if data['key'] == ' ':
        self.pause_mode = not self.pause_mode

      elif data['key'] == 'ArrowRight' and self.pause_mode:
        self.step = True

  def __call__(self, seq, mode='sot', *args, **kwargs):
    # Get init information
    init_info = seq.init_info()
    self.init_output()
    if self.tracking_category is not None and 'category' not in kwargs:
      kwargs['category'] = self.tracking_category
    output = self._track_sequence(seq, init_info, mode=mode, *args, **kwargs)
    return output

  def init_output(self):
    if self.tracking_mode in ['sot', 'vos']:
      self.output = {'target_bbox': [], 'time': [], 'segmentation': []}
    elif self.tracking_mode == 'mot':
      self.output = {'track_out': [], 'detect_out': [], 'frames': []}

  def run_sequence(self,
                   seq,
                   visualization=None,
                   debug=None,
                   visdom_info=None,
                   multiobj_mode=None):
    """Run tracker on sequence.
        args:
            seq: Sequence to run the tracker on.
            visualization: Set visualization flag (None means default value specified in the parameters).
            debug: Set debug level (None means default value specified in the parameters).
            visdom_info: Visdom info.
            multiobj_mode: Which mode to use for multiple objects.
        """
    visualization_ = visualization

    debug_ = debug
    if debug is None:
      debug_ = self.cfg.TRACKER.DEBUG_LEVEL
    if visualization is None:
      if debug is None:
        visualization_ = self.cfg.TRACKER.VISUALIZATION
      else:
        visualization_ = True if debug else False

    # params.visualization = visualization_
    # params.debug = debug_

    self._init_visdom(visdom_info, debug_)
    if visualization_ and self.visdom is None:
      self.init_visualization()

    # Get init information
    init_info = seq.init_info()
    self.init_output()

    output = self._track_sequence(seq,
                                  init_info,
                                  save_video=False,
                                  visualization=visualization)
    return output

  def update_tracking_outputs(self, tracker_out: dict, defaults: dict={}):
    if self.tracking_mode in ['sot', 'vos']:
      defaults = {} if defaults is None else defaults
      for key in self.output.keys():
        val = tracker_out.get(key, defaults.get(key, None))
        if key in tracker_out or val is not None:
          self.output[key].append(val)
    elif self.tracking_mode == 'mot':
      track_out, detect_out = tracker_out
      self.output['track_out'] += [track_out]
      self.output['detect_out'] += [detect_out]
      if defaults is not None:
        self.output['frames'] += [defaults.get('init_image', [])]
    else:
      raise NotImplementedError

  def _track_sequence(self,
                      seq,
                      init_info,
                      save_video=False,
                      visualization=False,
                      mode='sot',
                      **kwargs):
    # Initialize
    image = self._read_image(seq.frames[0])
    init_info['video'] = seq.name
    ## video writer
    vwriter = vio.FFmpegWriter("%s.mp4" % seq.name) if save_video else None

    start_time = time.time()
    out = self.tracker.initialize(image, init_info, **kwargs)
    init_info['time'] = time.time() - start_time
    self.update_tracking_outputs(out, init_info)
    if self.visdom is not None:
      self.tracker.visdom_draw_tracking(image, out)

    prev_output = out

    for frame_num, frame_path in enumerate(seq.frames[1:], start=1):
      while True:
        if not self.pause_mode:
          break
        elif self.step:
          self.step = False
          break
        else:
          time.sleep(0.1)

      image = self._read_image(frame_path)
      start_time = time.time()
      info = seq.frame_info(frame_num)
      info['previous_output'] = out
      info['time'] = time.time() - start_time
      #### track image
      out = self.tracker.track(image, info)
      self.update_tracking_outputs(out, info)
      # if self.tracker.tracking_mode == "sot":
      #   self.tracker.track_mot(image, info)
      # elif frame_num > 30:
      #   self.tracker.track_sot(image, info)

      segmentation = out.get('segmentation', None) if isinstance(out,
                                                                 dict) else None
      if self.visdom is not None:
        self.tracker.visdom_draw_tracking(
            #     image, out['target_bbox'], segmentation)
            image,
            out,
            segmentation,
            vwriter)
      # if frame_num % 30 == 0:
      # # if frame_num == 30:
      #   self.tracker.switch_tracking_mode(image, info)
    if save_video:
      vwriter.close()
    self.tracker.finish()
    return self.output



  def init_visualization(self):
    self.fig, self.ax = plt.subplots(1)
    self.fig.canvas.mpl_connect('key_press_event', self.press)
    plt.tight_layout()

  def visualize(self, image, state, segmentation=None):
    self.ax.cla()
    self.ax.imshow(image)
    if segmentation is not None:
      self.ax.imshow(segmentation, alpha=0.5)

    if isinstance(state, (OrderedDict, dict)):
      boxes = [v for k, v in state.items()]
    else:
      boxes = (state,)

    for i, box in enumerate(boxes, start=1):
      col = _tracker_disp_colors[i]
      col = [float(c) / 255.0 for c in col]
      rect = patches.Rectangle((box[0], box[1]),
                               box[2],
                               box[3],
                               linewidth=1,
                               edgecolor=col,
                               facecolor='none')
      self.ax.add_patch(rect)

    if getattr(self, 'gt_state', None) is not None:
      gt_state = self.gt_state
      rect = patches.Rectangle((gt_state[0], gt_state[1]),
                               gt_state[2],
                               gt_state[3],
                               linewidth=1,
                               edgecolor='g',
                               facecolor='none')
      self.ax.add_patch(rect)
    self.ax.set_axis_off()
    self.ax.axis('equal')
    draw_figure(self.fig)

    if self.pause_mode:
      keypress = False
      while not keypress:
        keypress = plt.waitforbuttonpress()

  def reset_tracker(self):
    pass

  def press(self, event):
    if event.key == 'p':
      self.pause_mode = not self.pause_mode
      print("Switching pause mode!")
    elif event.key == 'r':
      self.reset_tracker()
      print("Resetting target pos to gt!")

  def _read_image(self, image_file: Path):
    im = cv.imread(str(image_file))
    return cv.cvtColor(im, cv.COLOR_BGR2RGB)
