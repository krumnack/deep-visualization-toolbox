import sys
import os
import numpy as np

from net import Net

# ULF[fixme]: does not belong to the network ...
from image_misc import get_tiles_height_width_ratio


class CaffeNet(Net):

    def __init__(self, settings):
        """Initialize the caffe network.

        Initializing the caffe network includes two steps:
        (1) importing the caffe library. We are interested in
        the modified version that provides deconvolution support.
        (2) load the caffe model data.

        Arguments:
        settings: The settings object to be used. CaffeNet will only
        used settings prefixed with "caffe". ULF[todo]: check this claim!
        """
        super(CaffeNet, self).__init__(settings)

        self._range_scale = 1.0      # not needed; image already in [0,255]

        
        #ULF[todo]: explain, make this a setting
        self._net_channel_swap = (2,1,0)
        #self._net_channel_swap = None
        if self._net_channel_swap:
            self._net_channel_swap_inv = tuple([self._net_channel_swap.index(ii) for ii in range(len(self._net_channel_swap))])
        else:
            self._net_channel_swap_inv = None


        # (1) import caffe library
        #
        sys.path.insert(0, os.path.join(settings.caffevis_caffe_root, 'python'))
        import caffe
        print 'debug[caffe]: CaffeNet.__init__: using Caffe in', caffe.__file__

        # Check if the imported caffe provides all required functions
        self._check_caffe_version(caffe)
        
        # Set the mode to CPU or GPU.
        # Note: in the latest Caffe versions, there is one Caffe object
        # *per thread*, so the mode must be set per thread!
        # Here we set the mode for the main thread; it is also separately
        # set in CaffeProcThread.
        if settings.caffevis_mode_gpu:
            caffe.set_mode_gpu()
            print 'debug[caffe]: CaffeNet.__init__: CaffeVisApp mode (in main thread):     GPU'
        else:
            caffe.set_mode_cpu()
            print 'debug[caffe]: CaffeNet.__init__: CaffeVisApp mode (in main thread):     CPU'
        print 'debug[caffe]: CaffeNet.__init__: Loading the classifier (', settings.caffevis_deploy_prototxt, settings.caffevis_network_weights, ') ...'


        # (2) load the caffe model
        #        
        # ULF[hack]: make Caffe silent - there should be a better
        # (i.e. official) way to do so. We only want to suppress
        # the info (like network topology) while still seeing warnings
        # and errors!
        suppress_output = (hasattr(self.settings, 'caffe_init_silent')
                           and self.settings.caffe_init_silent)

        if suppress_output:
            # open 2 file descriptors
            null_fds = [os.open(os.devnull, os.O_RDWR) for x in xrange(2)]
            # save the current file descriptors to a tuple
            original_fds = os.dup(1), os.dup(2)
            # put /dev/null fds on stdout (1) and stderr (2)
            os.dup2(null_fds[0], 1)
            os.dup2(null_fds[1], 2)

        self.net = caffe.Classifier(
            settings.caffevis_deploy_prototxt,
            settings.caffevis_network_weights,
            mean = None,     # Set to None for now, assign later         # self._data_mean,
            channel_swap = self._net_channel_swap,
            raw_scale = self._range_scale,
        )
        
        if suppress_output:
            # restore file original descriptors for stdout (1) and stderr (2)
            os.dup2(original_fds[0], 1)
            os.dup2(original_fds[1], 2)
            # close the temporary file descriptors
            os.close(null_fds[0])
            os.close(null_fds[1])
        print 'debug[caffe]: CaffeNet.__init__: ... loading completed.'

        self._init_data_mean()
        self._populate_net_layer_info()
        self._check_force_backward_true()


    def _check_caffe_version(self, caffe):
        """Check if the caffe version provides all required functions.

        The deep visualization toolbox requires a modified version of
        caffe, that supports deconvolution. Without this functions,
        the toolbox is able to run, but will not provide full functionality.

        This method will issue a warning, if caffe does not provide the
        required functions.
        """
        if 'deconv_from_layer' in dir(caffe.classifier.Classifier):
            print "debug[caffe]: caffe version provides all required functions. Good!"
        else:
            print "warning: Function 'deconv_from_layer' is missing in caffe. Probably you are using a wrong caffe version. Some functions will not be available!'"

        
    def _init_data_mean(self):
        """Initialize the data mean.

        The data mean values are loaded from a separate file. Caffe can
        use thes
        """
        if isinstance(self.settings.caffevis_data_mean, basestring):
            # If the mean is given as a filename, load the file
            try:
                data_mean = np.load(self.settings.caffevis_data_mean)
            except IOError:
                print '\n\nCound not load mean file:', self.settings.caffevis_data_mean
                print 'Ensure that the values in settings.py point to a valid model weights file, network'
                print 'definition prototxt, and mean. To fetch a default model and mean file, use:\n'
                print '$ cd models/caffenet-yos/'
                print '$ ./fetch.sh\n\n'
                raise
            input_shape = self.get_input_data_shape()  # e.g. 227x227
            # Crop center region (e.g. 227x227) if mean is larger (e.g. 256x256)
            excess_h = data_mean.shape[1] - input_shape[0]
            excess_w = data_mean.shape[2] - input_shape[1]
            assert excess_h >= 0 and excess_w >= 0, 'mean should be at least as large as %s' % repr(input_shape)
            data_mean = data_mean[:, (excess_h/2):(excess_h/2+input_shape[0]),
                                     (excess_w/2):(excess_w/2+input_shape[1])]
        elif self.settings.caffevis_data_mean is None:
            data_mean = None
        else:
            # The mean has been given as a value or a tuple of values
            data_mean = np.array(self.settings.caffevis_data_mean)
            # Promote to shape C,1,1
            while len(data_mean.shape) < 1:
                data_mean = np.expand_dims(data_mean, -1)
            
            #if not isinstance(data_mean, tuple):
            #    # If given as int/float: promote to tuple
            #    data_mean = tuple(data_mean)

        if data_mean is not None:
            self.net.transformer.set_mean(self.net.inputs[0], data_mean)


    def _populate_net_layer_info(self):
        """Prepare the layer infos.

        For each layer, save the number of filters and precompute
        tile arrangement (needed by CaffeVisAppState to handle
        keyboard navigation).

        The net_layer_info values are dictionaries with the following
        entries:
            isconv (Boolean)
            data_shape (tuple of ints): shape of the layer
            n_tiles (int): the number of tiles to display
            tiles_rc (pair of ints):
            tile_rows (int):
            tile_cols (int):

        FIXME[caffe]: accesses net.blobs
        ULF: this should probably done by CaffeVisAppState itself ...
        """
        print 'debug[caffe,net]: CaffeNet._populate_net_layer_info: Populating layer info ...'
        super(CaffeNet, self)._populate_net_layer_info()
        for key in self.net.blobs.keys():
            self.net_layer_info[key] = {}
            # Conv example: (1, 96, 55, 55)
            # FC example: (1, 1000)
            blob_shape = self.net.blobs[key].data.shape
            print 'debug[caffe,net]: CaffeNet._populate_net_layer_info:', key, ": ", blob_shape

            assert len(blob_shape) in (2,4), 'Expected either 2 for FC or 4 for conv layer'
            self.net_layer_info[key]['isconv'] = (len(blob_shape) == 4)
            self.net_layer_info[key]['data_shape'] = blob_shape[1:]  # Chop off batch size
            self.net_layer_info[key]['n_tiles'] = blob_shape[1]
            self.net_layer_info[key]['tiles_rc'] = get_tiles_height_width_ratio(blob_shape[1], self.settings.caffevis_layers_aspect_ratio)
            self.net_layer_info[key]['tile_rows'] = self.net_layer_info[key]['tiles_rc'][0]
            self.net_layer_info[key]['tile_cols'] = self.net_layer_info[key]['tiles_rc'][1]

            # Caffe layers:
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

    def _check_force_backward_true(self):
        """Check the force_backward flag is set in the caffe model definition.

        Checks whether the given file contains a line with the
        following text, ignoring whitespace:

            force_backward: true

        If this is not the case, a warning text will be issued.
        
        ULF: This method should not be called from outside, but it is still
        called from "optimize_image.py"
        """
        prototxt_file = self.settings.caffevis_deploy_prototxt

        found = False
        with open(prototxt_file, 'r') as ff:
            for line in ff:
                fields = line.strip().split()
                if len(fields) == 2 and fields[0] == 'force_backward:' and fields[1] == 'true':
                    found = True
                    break

        if not found:
            print '\n\nWARNING: the specified prototxt'
            print '"%s"' % prototxt_file
            print 'does not contain the line "force_backward: true". This may result in backprop'
            print 'and deconv producing all zeros at the input layer. You may want to add this line'
            print 'to your prototxt file before continuing to force backprop to compute derivatives'
            print 'at the data layer as well.\n\n'


    def get_input_id(self):
        """Get the identifier for the input layer.

        Result: The type of this dentifier depends on the underlying
            network library. However, the identifier returned by this
            method is suitable as argument for other methods in this
            class that expect a layer_id.

        """
        return self.net.inputs[0]

    def get_layer_ids(self, include_input = True):
        """Get the layer identifiers of the network layers.

        
        Arguments:
        include_input:
            a flag indicating if the input layer should be
            included in the result.
        
        Result: A list of identifiers (strings in the case of Caffe).
            Notice that the type of these identifiers may depend on
            the underlying network library. However, the identifiers
            returned by this method should be suitable as arguments
            for other methods in this class that expect a layer_id.
        """
        layers = self.net.blobs.keys()
        if not include_input:
            layers = layers[1:]
        return layers

    def get_layer_n_tiles(self, layer_id):
        """
        ULF: only called by deepvis/app_state.py (1 time)
        """
        return self.net_layer_info[layer_id]['n_tiles']


    def get_layer_tile_cols(self, layer_id):
        """
        ULF: only called by deepvis/app_state.py (3 times)
        """
        return self.net_layer_info[layer_id]['tile_cols']


    def get_layer_tiles_rc(self, layer_id):
        """
        ULF: only called by deepvis/app_state.py (1 time)
        """
        return self.net_layer_info[layer_id]['tiles_rc']


    def get_input_data_shape(self):
        """
        ULF: only called by deepvis/proc_thread.py (1 time),
        and by self._init_data_mean()
        """
        # self.net.inputs[0] is 'data'
        # (as net.inputs is a list ['data'])
        return self.net.blobs[self.net.inputs[0]].data.shape[-2:]   # e.g. 227x227

    def get_layer_data(self, layer_id, unit = None, flatten = False):
        """Provide activation data for a given layer.
        """
        data = self.net.blobs[layer_id].data
        return data.flatten() if flatten else (data[0] if unit is None else data[0,unit])


    def get_layer_diff(self, layer_id, flatten = False):
        """Provide diff data for a given layer.
        """
        diff = self.net.blobs[layer_id].diff
        return diff.flatten() if flatten else diff[0]


    def get_layer_zeros(self, layer_id):
        """
        ULF: only called by deepvis/proc_thread.py (1 time)
        ULF: there should be a better solution
        """
        #hack:
        #return self.net.blobs[backprop_layer].diff * 0
        return self.net.blobs[layer_id].diff * 0


    def preproc_forward(self, img, data_hw):
        """Prepare image data for processing.

        Uses caffe.transformer.preprocess

        Arguments:
        img:
        data_hw
        
        
        ULF: called by deepvis/proc_thread.py
        ULF: app_helper.py: provides a function with similar name
        """
        appropriate_shape = data_hw + (3,)
        assert img.shape == appropriate_shape, 'img is wrong size (got %s but expected %s)' % (img.shape, appropriate_shape)
        #resized = caffe.io.resize_image(img, self.net.image_dims)   # e.g. (227, 227, 3)
        data_blob = self.net.transformer.preprocess('data', img)                # e.g. (3, 227, 227), mean subtracted and scaled to [0,255]
        data_blob = data_blob[np.newaxis,:,:,:]                   # e.g. (1, 3, 227, 227)
        output = self.net.forward(data=data_blob)
        return output


    def backward_from_layer(self, layer_id, diffs):
        '''Compute backward gradients from layer.

        Notice: this method relies on the methond deconv_from_layer(),
        which is not part of the standard Caffe. You need the
        deconv-deep-vis-toolbox branch of caffe to run this function.

        ULF[fixme]: currently the interface freezes, when the function
        is not available - look for a better way to deal with this problem.
        '''

        #print '**** Doing backprop with %s diffs in [%s,%s]' % (backprop_layer, diffs.min(), diffs.max())
        try:
            #ULF[old]:
            self.net.backward_from_layer(layer_id, diffs, zero_higher = True)
        except AttributeError:
            print 'ERROR: required bindings (backward_from_layer) not found! Try using the deconv-deep-vis-toolbox branch as described here: https://github.com/yosinski/deep-visualization-toolbox'
            raise


    def deconv_from_layer(self, layer_id, diffs):
        '''Compute backward gradients from layer.


        Notice: this method relies on the methond deconv_from_layer(),
        which is not part of the standard Caffe. You need the
        deconv-deep-vis-toolbox branch of caffe to run this function.

        ULF[fixme]: currently the interface freezes, when the function
        is not available - look for a better way to deal with this problem.
        '''
        #print '**** Doing deconv with %s diffs in [%s,%s]' % (backprop_layer, diffs.min(), diffs.max())
        try:
            self.net.deconv_from_layer(layer_id, diffs, zero_higher = True)
        except AttributeError:
            print 'ERROR: required bindings (deconv_from_layer) not found! Try using the deconv-deep-vis-toolbox branch as described here: https://github.com/yosinski/deep-visualization-toolbox'
            raise


    def get_input_gradient_as_image(self):
        """
        
        ULF: only called by deepvis/proc_thread.py
        """
        #ULF[old]:
        #grad_blob = self.net.blobs['data'].diff
        #grad_blob = self.net.blobs[self.net.inputs[0]].diff
        grad_blob = self.get_layer_diff(self.get_input_id())
        print "debug[caffe]: grad_blob.shape =", grad_blob.shape
        
        # Manually deprocess (skip mean subtraction and rescaling)
        #grad_img = self.net.deprocess('data', diff_blob)
        #grad_blob = grad_blob[0]                    # bc01 -> c01
        grad_blob = grad_blob.transpose((1,2,0))    # c01 -> 01c
        grad_img = grad_blob[:, :, self._net_channel_swap_inv]  # e.g. BGR -> RGB
        return grad_img
