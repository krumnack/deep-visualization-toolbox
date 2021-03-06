import time

from threading import Lock

from image_misc import get_tiles_height_width_ratio


class VisAppState(object):

    def __init__(self, my_net, settings, bindings):
        '''
        Arguments:
        :param net:
            an instance of Caffe.Classifier.
        :param settings: the settings object to be used to configure
            this application
        :param key_bindings: the key_bindings object assigning
            key_codes to tags
        '''
        self.lock = Lock()  # State is accessed in multiple threads
        self.settings = settings
        self.bindings = bindings
        self.my_net = my_net
        self._layers = my_net.get_layer_ids(include_input = False)
        if hasattr(self.settings, 'caffevis_filter_layers'):
            for name in self._layers:
                if self.settings.caffevis_filter_layers(name):
                    print '  Layer filtered out by caffevis_filter_layers: %s' % name
            self._layers = filter(lambda name: not self.settings.caffevis_filter_layers(name), self._layers)

        self._populate_net_layer_info()
        self.layer_boost_indiv_choices = self.settings.caffevis_boost_indiv_choices   # 0-1, 0 is noop
        self.layer_boost_gamma_choices = self.settings.caffevis_boost_gamma_choices   # 0-inf, 1 is noop
        self.caffe_net_state = 'free'     # 'free', 'proc', or 'draw'
        self.extra_msg = ''
        self.back_stale = True       # back becomes stale whenever the last back diffs were not computed using the current backprop unit and method (bprop or deconv)
        self.next_frame = None
        self.jpgvis_to_load_key = None
        self.last_key_at = 0
        self.quit = False

        self._reset_user_state()


    def _reset_user_state(self):
        '''Reset the state of the application.

        Resetting the state includes the following aspects:
            * select the first layer
            * the cursor is place in the top area (layer selection)
            * boost values are reset
        '''
        self.layer_idx = 0
        self.layer = self._layers[0]

        # initialize the cursor area:
        # 'top' (layer selection) or 'bottom' (units)
        self.cursor_area = 'top'
        self.selected_unit = 0

        #
        # Set display properties to defaults
        # (some defaults can be obtained from the settings object)
        #

        # Whether or not to show desired patterns instead of
        # activations in layers pane
        self.pattern_mode = False

        # layers_pane_zoom_mode:
        #   0: off,
        #   1: zoom selected (and show pref in small pane),
        #   2: zoom backprop
        self.layers_pane_zoom_mode = 0
        # layers_show_back:
        #   False: show forward activations.
        #   True: show backward diffs
        self.layers_show_back = False

        self.show_label_predictions = self.settings.caffevis_init_show_label_predictions
        self.show_unit_jpgs = self.settings.caffevis_init_show_unit_jpgs


        self._layer_boost_indiv_idx = self.settings.caffevis_boost_indiv_default_idx
        self.layer_boost_indiv = self.layer_boost_indiv_choices[self._layer_boost_indiv_idx]
        self._layer_boost_gamma_idx = self.settings.caffevis_boost_gamma_default_idx
        self.layer_boost_gamma = self.layer_boost_gamma_choices[self._layer_boost_gamma_idx]



        #
        # Back propagation:
        #
        
        # Which layer and unit (or channel) to use for backprop
        self.backprop_layer = self.layer
        self.backprop_unit = self.selected_unit
        self.backprop_selection_frozen = False    # If false, backprop unit tracks selected unit
        self.back_enabled = False
        self.back_mode = 'grad'      # 'grad' or 'deconv'
        self.back_filt_mode = 'raw'  # 'raw', 'gray', 'norm', 'normblur'
        
        
        # additional message to be displayed in the status bar
        kh,_ = self.bindings.get_key_help('help_mode')
        self.extra_msg = '%s for help' % kh[0]

        self.drawing_stale = True


    def _populate_net_layer_info(self):
        """Prepare the layer infos.

        For each layer, save the number of filters and precompute
        tile arrangement.

        The net_layer_info values are dictionaries with the following
        entries:
            isconv (Boolean)
            data_shape (tuple of ints): shape of the layer
            n_tiles (int): the number of tiles to display
            tiles_rc (pair of ints): rows and columns
            tile_rows (int): rows
            tile_cols (int): columns

        ULF[fixme]: maybe not really part of the application state but of the core application ...
        """
        print 'debug[net]: VisAppState._populate_net_layer_info: Populating layer info ...'
        self.net_layer_info = {}
        for layer_id in self.my_net.get_layer_ids():
            self.net_layer_info[layer_id] = {}
            layer_shape = self.my_net.get_layer_shape(layer_id)
            print 'debug[net]: VisAppState._populate_net_layer_info:', layer_id, ": ", layer_shape

            assert len(layer_shape) in (1,3), 'Expected either 1 for FC or 3 for conv layer'
            self.net_layer_info[layer_id]['isconv'] = (len(layer_shape) == 3)
            self.net_layer_info[layer_id]['data_shape'] = layer_shape
            self.net_layer_info[layer_id]['n_tiles'] = layer_shape[0]
            self.net_layer_info[layer_id]['tiles_rc'] = get_tiles_height_width_ratio(layer_shape[0], self.settings.caffevis_layers_aspect_ratio)
            self.net_layer_info[layer_id]['tile_rows'] = self.net_layer_info[layer_id]['tiles_rc'][0]
            self.net_layer_info[layer_id]['tile_cols'] = self.net_layer_info[layer_id]['tiles_rc'][1]

            # Caffe layers (caffe_yos):
            # Populating layer info ...
            # data :  (1, 3, 227, 227)
            # conv1 :  (1, 96, 55, 55)
            # pool1 :  (1, 96, 27, 27)
            # norm1 :  (1, 96, 27, 27)
            # conv2 :  (1, 256, 27, 27)
            # pool2 :  (1, 256, 13, 13)
            # norm2 :  (1, 256, 13, 13)
            # conv3 :  (1, 384, 13, 13)
            # conv4 :  (1, 384, 13, 13)
            # conv5 :  (1, 256, 13, 13)
            # pool5 :  (1, 256, 6, 6)
            # fc6 :  (1, 4096)
            # fc7 :  (1, 4096)
            # fc8 :  (1, 1000)
            # prob :  (1, 1000)

    def _get_layer_tiles_rc(self, layer_id = None):
        """ULF: only called by (app.py)"""
        return self.net_layer_info[self.layer if layer_id is None else layer_id]['tiles_rc']
        
    def handle_key(self, key):
        '''Handle keyboard events.

        The VisAppState is interested in keys that change the visual
        state of the applications, i.e.:
            * select a new unit using the cursor keys (sel_*)
            * select a new layers (sel_layer_* = [u/o])
            * switch between forward/backward/deconv mode
            * activate zoom mode (zoom_mode = [z])
            * toggle the display labels (toggle_label_predictions = [8])
            * toggle display of precomputed unit jpegs (toggle_unit_jpgs = [9])
            * ...
            * reset the state (reset_state = [esc])
        '''
        #print 'Ignoring key:', key
        if key == -1:
            return key

        with self.lock:
            key_handled = True
            self.last_key_at = time.time()
            tag = self.bindings.get_tag(key)
            print 'debug[key]:', key, '(', tag, ') at', self.last_key_at
            
            if tag == 'reset_state':
                self._reset_user_state()
            elif tag == 'sel_layer_left':
                #hh,ww = self.tiles_height_width
                #self.selected_unit = self.selected_unit % ww   # equivalent to scrolling all the way to the top row
                #self.cursor_area = 'top' # Then to the control pane
                self.layer_idx = max(0, self.layer_idx - 1)
                self.layer = self._layers[self.layer_idx]
            elif tag == 'sel_layer_right':
                #hh,ww = self.tiles_height_width
                #self.selected_unit = self.selected_unit % ww   # equivalent to scrolling all the way to the top row
                #self.cursor_area = 'top' # Then to the control pane
                self.layer_idx = min(len(self._layers) - 1, self.layer_idx + 1)
                self.layer = self._layers[self.layer_idx]
            elif tag == 'sel_left':
                self.move_selection('left')
            elif tag == 'sel_right':
                self.move_selection('right')
            elif tag == 'sel_down':
                self.move_selection('down')
            elif tag == 'sel_up':
                self.move_selection('up')

            elif tag == 'sel_left_fast':
                self.move_selection('left', self.settings.caffevis_fast_move_dist)
            elif tag == 'sel_right_fast':
                self.move_selection('right', self.settings.caffevis_fast_move_dist)
            elif tag == 'sel_down_fast':
                self.move_selection('down', self.settings.caffevis_fast_move_dist)
            elif tag == 'sel_up_fast':
                self.move_selection('up', self.settings.caffevis_fast_move_dist)

            elif tag == 'boost_individual':
                self._layer_boost_indiv_idx = (self._layer_boost_indiv_idx + 1) % len(self.layer_boost_indiv_choices)
                self.layer_boost_indiv = self.layer_boost_indiv_choices[self._layer_boost_indiv_idx]
            elif tag == 'boost_gamma':
                self._layer_boost_gamma_idx = (self._layer_boost_gamma_idx + 1) % len(self.layer_boost_gamma_choices)
                self.layer_boost_gamma = self.layer_boost_gamma_choices[self._layer_boost_gamma_idx]
            elif tag == 'pattern_mode':
                self.pattern_mode = not self.pattern_mode
                if self.pattern_mode and not hasattr(self.settings, 'caffevis_unit_jpg_dir'):
                    print 'Cannot switch to pattern mode; caffevis_unit_jpg_dir not defined in settings.py.'
                    self.pattern_mode = False
            elif tag == 'show_back':
                # If in pattern mode: switch to fwd/back. Else toggle fwd/back mode
                if self.pattern_mode:
                    self.pattern_mode = False
                else:
                    self.layers_show_back = not self.layers_show_back
                if self.layers_show_back:
                    if not self.back_enabled:
                        self.back_enabled = True
                        self.back_stale = True
            elif tag == 'back_mode':
                if not self.back_enabled:
                    self.back_enabled = True
                    self.back_mode = 'grad'
                    self.back_stale = True
                else:
                    if self.back_mode == 'grad':
                        self.back_mode = 'deconv'
                        self.back_stale = True
                    else:
                        self.back_enabled = False
            elif tag == 'back_filt_mode':
                    if self.back_filt_mode == 'raw':
                        self.back_filt_mode = 'gray'
                    elif self.back_filt_mode == 'gray':
                        self.back_filt_mode = 'norm'
                    elif self.back_filt_mode == 'norm':
                        self.back_filt_mode = 'normblur'
                    else:
                        self.back_filt_mode = 'raw'
            elif tag == 'ez_back_mode_loop':
                # Cycle:
                # off -> grad (raw) -> grad(gray) -> grad(norm) -> grad(normblur) -> deconv
                if not self.back_enabled:
                    self.back_enabled = True
                    self.back_mode = 'grad'
                    self.back_filt_mode = 'raw'
                    self.back_stale = True
                elif self.back_mode == 'grad' and self.back_filt_mode == 'raw':
                    self.back_filt_mode = 'norm'
                elif self.back_mode == 'grad' and self.back_filt_mode == 'norm':
                    self.back_mode = 'deconv'
                    self.back_filt_mode = 'raw'
                    self.back_stale = True
                else:
                    self.back_enabled = False
            elif tag == 'freeze_back_unit':
                # Freeze selected layer/unit as backprop unit
                self.backprop_selection_frozen = not self.backprop_selection_frozen
                if self.backprop_selection_frozen:
                    # Grap layer/selected_unit upon transition from non-frozen -> frozen
                    self.backprop_layer = self.layer
                    self.backprop_unit = self.selected_unit                    
            elif tag == 'zoom_mode':
                self.layers_pane_zoom_mode = (self.layers_pane_zoom_mode + 1) % 3
                if self.layers_pane_zoom_mode == 2 and not self.back_enabled:
                    # Skip zoom into backprop pane when backprop is off
                    self.layers_pane_zoom_mode = 0

            elif tag == 'toggle_label_predictions':
                self.show_label_predictions = not self.show_label_predictions

            elif tag == 'toggle_unit_jpgs':
                self.show_unit_jpgs = not self.show_unit_jpgs

            else:
                key_handled = False

            self._ensure_valid_selected()

            self.drawing_stale = key_handled   # Request redraw any time we handled the key

        return (None if key_handled else key)

    def redraw_needed(self):
        with self.lock:
            return self.drawing_stale

    def move_selection(self, direction, dist = 1):
        if direction == 'left':
            if self.cursor_area == 'top':
                self.layer_idx = max(0, self.layer_idx - dist)
                self.layer = self._layers[self.layer_idx]
            else:
                self.selected_unit -= dist
        elif direction == 'right':
            if self.cursor_area == 'top':
                self.layer_idx = min(len(self._layers) - 1, self.layer_idx + dist)
                self.layer = self._layers[self.layer_idx]
            else:
                self.selected_unit += dist
        elif direction == 'down':
            if self.cursor_area == 'top':
                self.cursor_area = 'bottom'
            else:
                self.selected_unit += self.net_layer_info[self.layer]['tile_cols']
        elif direction == 'up':
            if self.cursor_area == 'top':
                pass
            else:
                self.selected_unit -= self.net_layer_info[self.layer]['tile_cols'] * dist
                if self.selected_unit < 0:
                    self.selected_unit += self.net_layer_info[self.layer]['tile_cols']
                    self.cursor_area = 'top'

    def _ensure_valid_selected(self):
        n_tiles = self.net_layer_info[self.layer]['n_tiles']

        # Forward selection
        self.selected_unit = max(0, self.selected_unit)
        self.selected_unit = min(n_tiles-1, self.selected_unit)

        # Backward selection
        if not self.backprop_selection_frozen:
            # If backprop_selection is not frozen, backprop layer/unit follows selected unit
            if not (self.backprop_layer == self.layer and self.backprop_unit == self.selected_unit):
                self.backprop_layer = self.layer
                self.backprop_unit = self.selected_unit
                self.back_stale = True    # If there is any change, back diffs are now stale
