




class Net(object):

    def __init__(self, settings):
        self.settings = settings



    def get_layers_ids(self,include_input = True):
        """Get the layer identifiers of the network layers

        FIXME[todo]: may include code to filter out specific layers
        (specified by 'caffevis_filter_layers', see methond __init__
        in CaffeVisAppState)

        Arguments:
            include_input (boolean): Indicates if the identifier of the
            input layer (i.e., the data layer) should be included in
            the result.

        Result:
            A list of identifiers (suitable to access individual layers
            using the interface of this class. Actual data type May depend
            on the underlying implementation).
        """
        pass

    def get_input_id(self):
        """Get the identifier for the input layer.

        Result: The type of this dentifier depends on the underlying
            network library. However, the identifier returned by this
            method is suitable as argument for other methods in this
            class that expect a layer_id.
        """
        return self.net.inputs[0]


    def get_layer_shape(self,layer_id):
        """Get the shape of the given layer.

        Returns a tuples describing the shape of the layer:
            Fully connected layer: 1-tuple, the number of neurons,
            example: (1000, )

            Convolutional layer: n_filter x n_rows x n_columns,
            example: (96, 55, 55)
        """
        pass


    def get_input_data_shape(self):
        """Get the size (rows and columns, but not channels)
        of the input images.

        Result: a pair specifying the number of rows and the number of columns,
            e.g. (227,227).
        
        ULF: only called by deepvis/proc_thread.py (1 time),
        and by self._init_data_mean()
        """
        
        return self.get_layer_shape(self.get_input_id())[-2:]


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
