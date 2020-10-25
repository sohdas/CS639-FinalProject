import numpy as np

class StateObject():
    # Images: (Batch_size x 3 x 512 x 512) giving the current batch of images.
    def __init__(self, images):
        self.images = images
        self.batch_size = images.shape[0]
        # Create a list to map from action index to views.
        action_to_view = []
        for i in range(16):
            for j in range(16):
                start_idx_x = i*32
                end_idx_x = (i+1)*32
                start_idx_y = j*32
                end_idx_y = (j+1)*32
                action_to_view.append(( (start_idx_x, end_idx_x), (start_idx_y, end_idx_y) ))
        self.action_to_view = action_to_view
        self.index = None
    
    # Given a set of action probabilities, get the images corresponding to the maximal action.
    # actions: (batch_size x 256) tensor given the probabilities of selecting a given action.
    def get_view(self, actions=None):
        # If actions==None (ie. first time selecting an action), just return the upper left view.
        if type(actions) == type(None):
            self.index = np.array([[0, 0] for _ in range(self.batch_size)])
            return self.images[:, :, 0:32, 0:32], self.index
        
        actions = actions.detach().numpy()
        action_inds = np.argmax(actions, axis=1) # indexes of max prob action.
        # List of the indices used to retrieve the subarray corresponding to the views chosen by the max action.
        view_subarray_inds = [self.action_to_view[action_max] for action_max in action_inds]
        view_locs = np.array([[action_max//16/15.0, action_max%16/15.0] for action_max in action_inds]) # THIS MIGHT BE WRONG. ORDER MIGHT BE WRONG OR THE 2D REPRESENTATION OF LOCATION MIGHT BE INADEQUATE.
        # Iterate over subarray inds to generate binary masks.
        mask = np.zeros([self.batch_size, 3, 512, 512], dtype=bool)
        for i, inds in enumerate(view_subarray_inds):
            mask[i, :, inds[0][0]:inds[0][1], inds[1][0]:inds[1][1]] = 1
        views = self.images[mask].reshape([self.batch_size, 3, 32, 32])
        self.index = view_locs
        return views, self.index