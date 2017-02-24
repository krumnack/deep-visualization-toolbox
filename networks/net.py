




class Net(object):

    def __init__(self, settings):
        self.settings = settings

    def _populate_net_layer_info(self):
        self.net_layer_info = {}


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


    def get_layer_infos(self):
        """Provide the dictionary of layer infos.

        FIXME: used in:
            CaffeVisAppState.move_selection: property 'tile_cols'
            CaffeVisAppState._ensure_valid_selected: property 'n_tiles'
            CaffeVisApp._draw_layer_pane: property 'tiles_rc'
            
        Result:
            A dictionary of layer information.
        """
        return self.net_layer_info



