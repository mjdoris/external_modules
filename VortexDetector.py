import torch
import numpy as np
from tqdm import tqdm
from skimage.filters import threshold_mean
from skimage.measure import regionprops
from skimage.measure import label as meas_label
from scipy.ndimage import rotate
from soldet.SolitonDetector import SolitonDetector

class VortexDetector(SolitonDetector):
    def __init__(self, augment = False):
        super().__init__(process_fn = vortex_process_fn,
                         od_model = ObjectDetector2D,
                         od_dataset_fn = VortexODDataset,
                         od_loss_fn = MetzLoss2D,
                         augment = augment)

class ObjectDetector2D(torch.nn.Module):
    """ 2D "Vortex" Object Detector
        This pytorch object detector model identifies the position of excitations in two dimensions.
        Based on the work done in https://arxiv.org/abs/2012.13097.

        Parameters
        ----------
        layers : int
            The number of layers to use in the model. Each layer creates an object_cell with corresponding parameters. 
            (default = 4)
        in_channels : list
            A list of input channels to the 2D convolutions for each layer. 
            (default = [1, 16, 32, 64])
        out_channels : list
            A list of output channels to the 2D convolutions for each layer.
            (default = [16, 32, 64, 128])
        pool : list
            A list of pooling kernel sizes for each layer.
            (default = [(2,2), None, (2,2), None])
        kernel : tuple
            The 2D kernel size to use of shape (kH, kW) for each layer.
            (default = (4, 4))
        label_shape : tuple
            The size of the position labels after converting from real positions to the compressed cell representation.
            For 2D this shape is typically (height // 4, width // 4), where 4 is the number of pixels for each cell in 
            the array.
            (default = (33, 33))
        dropout : float
            How often neurons should drop out. 
            (default = 0.1)
    """
    def __init__(self, dropout = 0.1, layers = 4, in_channels = [1, 16, 32, 64], out_channels = [16, 32, 64, 128],
                 pool = [(2,2), None, (2,2), None], kernel = (4, 4), label_shape = (33, 33)):
        super().__init__()

        self.layers = torch.nn.ModuleList()
        for idx in range(layers):
            self.layers.append(object_cell(in_channels = in_channels[idx], out_channels = out_channels[idx], kernel = kernel, pool = pool[idx], dropout = dropout))

        ##Output Layers
        self.pool = torch.nn.AdaptiveMaxPool2d(label_shape)
        self.output = torch.nn.Conv2d(in_channels = out_channels[-1], out_channels = 3, kernel_size = (1,1), padding='same')
        self.output_act = torch.nn.Sigmoid()
        torch.nn.init.xavier_uniform_(self.output.weight)

    def forward(self, x):
        """ 
        Take a tensor and identify any vortices and their positions.

        Parameters
        ----------
        x : tensor of shape (B, 1, H, W)
            The input tensor to make a prediction on. The expected shape is of shape (B, 1, H, W), where B is the batch
            size, H is the image height, and W is the image width.

        Returns
        ----------
        x : tensor of shape (B, 3, H // 4, W // 4)
            The output tensor containing the probabilities for a vortex to be present in one of the cells and its 
            fractional position within a cell. Here B is the batch size and dimension 1 contains the probability (0),
            the vertical position in the cell (1), and the horizontal position within the cell (2). For the position 
            values 0 to 1 indicate which side of the cell and by extension which pixel after conversion. 
            Dimension 2 indicates the number of cells in the vertical direction and dimension 3 indicates the number of
            cells in the horizontal direction. 
        """
        for layer in self.layers:
            x = layer(x)
        x = self.pool(x)
        x = self.output_act(self.output(x))

        return x

class object_cell(torch.nn.Module):
    """ Object cell for use in 2D Object Detector.
        This cell represents the base layer of the 2D Object Detector used in identifying the probability of a vortex
        being present and its position.

        Parameters
        ----------
        in_channels: int
            The number of input channels in the image.
        out_channels: int
            The number of output channels in the image.
        kernel: tuple
            The 2D kernel size to use of shape (kH, kW)
        pool: list
            The size of the kernel to use during Max Pooling of the data.
            (default = None)
        dropout: float
            How often neurons should drop out.
            (default = 0.1)
    """
    def __init__(self, in_channels: int, out_channels: int, kernel: tuple, pool: list = None, dropout: float = 0.1):
        super().__init__()
        
        ##Network Layers
        self.conv_lay1 = torch.nn.Conv2d(in_channels, out_channels, kernel, padding='same', groups=in_channels)
        self.conv_lay2 = torch.nn.Conv2d(out_channels, out_channels, kernel, padding='same', groups=in_channels)
        self.conv_lay_act1 = torch.nn.PReLU()
        self.conv_lay_act2 = torch.nn.PReLU()
        self.bypass_conv = torch.nn.Conv2d(in_channels, out_channels, (1, 1), padding='same', groups=1)

        ##Functional Layers
        self.dropout = torch.nn.Dropout(dropout)
        if pool is not None:
            self.pool = torch.nn.MaxPool2d(pool)
        else:
            self.pool = None
        self.norm = torch.nn.BatchNorm2d(out_channels)
        
        ##Weight Init
        torch.nn.init.kaiming_normal_(self.conv_lay1.weight)
        torch.nn.init.kaiming_normal_(self.conv_lay2.weight)
    
    def forward(self, x):
        bypass = x
        
        x = self.conv_lay_act1(self.conv_lay1(x))
        x = self.conv_lay_act2(self.conv_lay2(x))
        x = self.norm(x)
        x = self.dropout(x)

        if self.pool == None:
            x = x + self.bypass_conv(bypass)
        else:
            x = self.pool(x) + self.pool(self.bypass_conv(bypass))
        
        return x

class MetzLoss2D(torch.nn.Module):
    '''
    Implementation of the loss function defined in: https://arxiv.org/abs/2012.13097
    The first term is essentially the weighted cross entropy probability for the cell belonging to the 'vortex present' 
    class.
    The second term is a mean-squared error for the fractional position within the cell.
    
    Parameters
    ----------
    CE_weight : float
        The weight value used when calculating the cross entropy probability.
    MSE_weight : float
        The weight value used when calculating the mean-squared error.
    
    '''
    def __init__(self, CE_weight: float = 10, MSE_weight: float = 10):
        super().__init__()
        self.CE_weight = CE_weight
        self.MSE_weight = MSE_weight
    
    def forward(self, prediction: torch.Tensor, target: torch.Tensor):
        eps = 1e-10
        ce_loss = (-self.CE_weight*target[:, 0, :, :]*torch.log(prediction[:, 0, :, :] + eps) 
                   - (1 - target[:, 0, :, :])*torch.log(1 - prediction[:, 0, :, :] + eps))
        mse_loss = (self.MSE_weight*target[:, 0, :, :]*((target[:, 1, :, :] - prediction[:, 1, :, :])**2 
                                                        + (target[:, 2, :, :] - prediction[:, 2, :, :])**2))
        loss = torch.sum(ce_loss + mse_loss) / prediction.shape[0]
        return loss

class VortexODDataset(torch.utils.data.Dataset):
    '''
    A dataset class for the ML based 2D object detector.
    This will work through a list, or dictionary, of dictionaries and grab the image data. This image data is
    expected to be at key 'data'.
    It will also grab the positions at key 'positions'.
    
    Parameters
    ----------
    data : list or dict
        The data to build a dataset from.
    transform_expand : bool (Unused)
        If set to True the data is augmented with rotations.
        (default = False)
    threshold : list
        A list of values that influence the conversion between real positions and cell positions.
        Threshold[0] is the minimum value to consider an excitation is present.
        Threshold[1] is the minimum distance two excitations can be considered seperate. Any distances under this
        value is considered the same excitation.
        (default = [0.5, np.sqrt(4**2 + 4**2)])
    dims : tuple
        The shape of the image data.
    '''
    def __init__(self, data: dict, augment: bool = False, threshold: list = [0.5, np.sqrt(4**2 + 4**2)], 
                 dims: tuple = (132, 132)):
        self.threshold = threshold
        self.dims = dims
        
        img_data = []
        label_data = []
        for entry in data:
            if entry['data'].shape[0] != entry['data'].shape[1]:
                raise ValueError("Loaded image data is rectangular. 2D SolDet enforces square data.")
            
            img_data.append(entry['data'])
            if 'positions' in entry.keys():
                label_data.append(entry['positions'])
            else:
                label_data.append([])

            if augment:
                #Rotation
                angle = np.random.rand()*2*np.pi
                R = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
                aug_img = rotate(entry['data'], angle*(180/np.pi), reshape = False)
                img_data.append(aug_img)
                if 'positions' in entry.keys():
                    aug_pos = []
                    for coord in entry['positions']:
                        xy_rot = R @ (np.array(coord) - np.array(entry['data'].shape) // 2)
                        xy_rot += np.array(entry['data'].shape) // 2
                        aug_pos.append(xy_rot.tolist())
                    label_data.append(aug_pos)
                else:
                    label_data.append([])

                #Noise
                noise = np.random.normal(0, 0.05, entry['data'].shape)
                noise[entry['data'] == 0] = 0
                aug_img = entry['data'] + noise
                img_data.append(aug_img)
                if 'positions' in entry.keys():
                    label_data.append(entry['positions'])
                else:
                    label_data.append([])

        img_data = np.array(img_data)
        img_data = np.reshape(img_data,(img_data.shape[0], 1, img_data.shape[1], img_data.shape[2]))
            
        self.imgs = torch.from_numpy(img_data).float()
        self.pos = torch.from_numpy(self.data_to_labels(label_data, self.threshold)).float()
        self.og_labels = label_data
            
    def __len__(self):
        '''
        Returns the length of the dataset.

        Returns
        ----------
        length : int
        '''
        return len(self.imgs)
    
    def __getitem__(self, idx: int):
        '''
        Retrieves a sample at the specified index.

        Parameters
        ----------
        idx : int
            The sample index

        Returns
        ----------
        image : ndarray
            The image data at the specified index
        pos : list of floats
            A list of positions at the specified index
        '''
        image = self.imgs[idx]
        pos = self.pos[idx]
        
        return image, pos 
    
    def labels_to_data(self, label_out: np.ndarray, threshold: list, xdim: int = 132, ydim: int = 132):
        '''
        Converts the labels in cell space to positions in pixel space.

        Parameters
        ----------
        label_out : ndarray
            An array of probability and fractional position values in cell space.
        threshold : list
            A list of values that influence the conversion between real positions and cell positions.
            Threshold[0] is the minimum value to consider an excitation is present.
            Threshold[1] is the minimum distance two excitations can be considered seperate. Any distances under this
            value is ocnsidered the same excitation.
            (default = [0.5, np.sqrt(4**2 + 4**2)])
        xdim : int
            The horizontal dimension of the image data.
            (default = 132)
        ydim : int
            The vertical dimension of the image data.
            (default = 132)

        Returns
        ----------
        labels : list
            positions in pixel space
        '''
        if type(label_out) is not np.ndarray:
            raise ValueError('Invalid type. Label data should be provided as numpy array.')
        
        return vortex_labels_func(label_out, threshold, xdim, ydim)
    
    def data_to_labels(self, label_in: list , threshold: list, xdim: int = 132, ydim: int = 132):
        '''
        Converts the positions in pixel space to positions and probability in cell space.

        Parameters
        ----------
        label_in : list
            A list of position lists.
        threshold : list
            A list of values that influence the conversion between real positions and cell positions.
            Threshold[0] is the minimum value to consider an excitation is present.
            Threshold[1] is the minimum distance two excitations can be considered seperate. Any distances under this
            value is ocnsidered the same excitation.
            (default = [0.5, np.sqrt(4**2 + 4**2)])
        xdim : int
            The horizontal dimension of the image data.
            (default = 132)
        ydim : int
            The vertical dimension of the image data.
            (default = 132)

        Returns
        ----------
        labels: ndarray
            The positions in the cell space. This is an array of (3,)
        '''
        if type(label_in) is not list:
            raise ValueError('Invalid type. Label data should be provided as a list.')
        
        return vortex_labels_func(label_in, threshold, xdim, ydim)
    
def vortex_labels_func(label_in: list | np.ndarray, threshold: list = [0.5, np.sqrt(4**2 + 4**2)], xdim: int = 132, 
                       ydim: int = 132):
    '''
    Convert between vortex positions in pixel space and cell space. This new space is a compressed representation of
    the positions in pixel space and the probability of them being present in a cell. The new space is a (3, 33, 33)
    array of values with the first (33, 33) entries representing the probability of an excitation being located in a 
    cell, and the second (33, 33) entries representing the fractional position of the excitation in that cell.
    Each cell represents a window of 4 x 4 pixels (H x W).
    
    The behavior of this function depends on the data type of label_in.41
    
    Parameters
    ----------
    label_in : list or ndarray
        If the data type is a list then it is assumed that this is a list of positions in pixel space. Valid input can
        be a list of a single value for single image input, or a list of sub lists of positions for multiple images.
        The output will be an array of (3, 33, 33).

        If the data type is an array then it is assumed this input is an array of values in cell space. For each cell
        whose probability is above the threshold will have a position calculated. This position will be based on the
        fractional position in the cell. If multiple excitations exists next to each other and fall below the threshold
        the average positions will be calculated between the two.
    threshold : list
        A list of values that influence the conversion between real positions and cell positions.
        Threshold[0] is the minimum value to consider that an excitation is present.
        Threshold[1] is the minimum distance two excitations can be considered seperate. Any distances under this
        value is considered the same excitation.
        (default = [0.5, np.sqrt(4**2 + 4**2)])
    
    Returns
    ----------
    label_out : list or ndarray
        If label_in was a list then the output is an array of (3, 33, 33) values in cell space.
        If label_in was an array then the output is a list of positions in pixel space.

    '''
    dims = [xdim, ydim]

    ### Positions to Label
    if type(label_in) == list:
        label_out = np.zeros((3, 33, 33))
        if len(label_in[0]) == 0:
            pass
        else:
            if type(label_in[0][0]) in [float, np.float32, np.float64, int]:
                for coord in label_in:
                    if coord[0] < dims[0] and coord[0] > 0:
                        if coord[1] < dims[1] and coord[1] > 0:
                            #Probability a soliton is present
                            label_out[0, int(coord[1] // 4), int(coord[0] // 4)] = 1 
                            #Fractional position along x direction of cell
                            label_out[1, int(coord[1] // 4), int(coord[0] // 4)] = (coord[0] % 4)/4 
                            #Fractional position along y direction of cell
                            label_out[2, int(coord[1] // 4), int(coord[0] // 4)] = (coord[1] % 4)/4 
                        else:
                            print('soliton positon beyond image dimensions.')
                    else:
                        print('soliton positon beyond image dimensions.')
        
            elif type(label_in[0][0]) in [list, tuple]: # A list of postions on many images
                label_out = np.zeros((len(label_in), 3, 33, 33))
                for i, pos in enumerate(label_in):
                    if len(pos[0]) == 0:
                        pass
                    else:
                        for coord in pos:
                            if coord[0] < dims[0] and coord[0] > 0:
                                if coord[1] < dims[1] and coord[1] > 0:
                                    #Probability a soliton is present
                                    label_out[i, 0, int(coord[1] // 4), int(coord[0] // 4)] = 1 
                                    #Fractional position along x direction of cell
                                    label_out[i, 1, int(coord[1] // 4), int(coord[0] // 4)] = (coord[0] % 4)/4 
                                    #Fractional position along y direction of cell
                                    label_out[i, 2, int(coord[1] // 4), int(coord[0] // 4)] = (coord[1] % 4)/4 
                                else:
                                    print('soliton positon beyond image dimensions.')
                            else:
                                print('soliton positon beyond image dimensions.')
    
    ### Label to Positions
    elif type(label_in) == np.ndarray:
        label_out = []
        if label_in.shape == (3, 33, 33):
            for i in range(33):
                for j in range(33):
                    if label_in[0, j, i] > threshold[0]:
                        label_out.append([4 * i + 4 * label_in[1, j, i], 4 * j + 4 * label_in[2, j, i]])
            
            if len(label_out) > 1:
                i = 0
                while (i+1) < len(label_out):
                    dist = np.sqrt( (label_out[i+1][0] - label_out[i][0])**2 + (label_out[i+1][1] - label_out[i][1])**2)
                    if dist < threshold[1]:
                        label_out[i][0] = (label_out[i+1][0] + label_out[i][0])/2
                        label_out[i][1] = (label_out[i+1][1] + label_out[i][1])/2
                        del label_out[i+1]
                    else:
                        i +=1

        elif label_in.shape[1:] == (3, 33, 33):
            for label in label_in:
                l_out = []
                for i in range(33):
                    for j in range(33):
                        if label[0, j, i] > threshold[0]:
                            l_out.append([4 * i + 4 * label[1, j, i], 4 * j + 4 * label[2, j, i]])
                
                if len(l_out) > 1:
                    i = 0
                    while (i+1) < len(l_out):
                        dist = np.sqrt( (l_out[i+1][0] - l_out[i][0])**2 + (l_out[i+1][1] - l_out[i][1])**2)
                        if dist < threshold[1]:
                            l_out[i][0] = (l_out[i+1][0] + l_out[i][0])/2
                            l_out[i][1] = (l_out[i+1][1] + l_out[i][1])/2
                            del l_out[i+1]
                        else:
                            i +=1
                label_out.append(l_out)
        else:
            raise ValueError('Input has incorrect dimensions. Expected (N, 3, 33, 33) or \
                             (3, 33, 33) but got {}.'.format(label_in.shape))
    else:
        raise ValueError('Input is an incorrect type. Expected list or \
                         numpy.ndarray but got {}.'.format(type(label_in)))

    return label_out
 
def vortex_process_fn(data_path: str, pos_path: str = None, label: str| int = 'unlabeled'):
    '''
    An preliminary processing function for vortex data. Similar to the SolDet default processing function, this will
    load in a target data file and prepare it for use it the 2D version of SolDet.
    
    Parameters
    ----------
    data_path : string
        The target file containing the image data. This should be a numpy file of an array of image data of shape
        (N, H, W).
    pos_path : string
        The target file containing the position data for vortex locations. This should be a numpy object file of a
        list or array of N lists or tuples containing the X, Y positions for each vortex.
        (default = None)
    label : string or int
        The class label for the image.
        (default = 'unlabeled')

    Returns
    ----------
    data_samples : list
        A list of dictionaries containing the collected pre-processed data.
        Each dictionary contains, at minimum:
            The masked and unmasked image data of shape (132, 132).
            The class label.
            The class directory.
            The original image size.
            The original data path.
        If positions were provided these are also saved in the dictionary entries.
    '''
    data = np.load(data_path)
    
    if pos_path is not None:
        pos = np.load(pos_path, allow_pickle=True).tolist()
    else:
        pos = None

    data_samples = []

    for idx, ODImage in enumerate(tqdm(data, desc='Processing vortex data..')):
        sample = {}

        fullimgsize = ODImage.shape
        yhigh = fullimgsize[0]
        xhigh = fullimgsize[1]

        if yhigh != xhigh:
            if xhigh > yhigh:
                crop = (xhigh - yhigh) // 2
                ODImage = ODImage[:, crop:-crop]
                fullimgsize = ODImage.shape
                yhigh = fullimgsize[0]
                xhigh = fullimgsize[1]
                if pos is not None:
                    for i in range(len(pos[idx])):
                        pos[idx][i] = [pos[idx][i][0] - crop, pos[idx][i][1]]
            else:
                crop = (yhigh - xhigh) // 2
                ODImage = ODImage[crop:-crop, :]
                fullimgsize = ODImage.shape
                yhigh = fullimgsize[0]
                xhigh = fullimgsize[1]
                if pos is not None:
                    for i in range(len(pos[idx])):
                        pos[idx][i] = [pos[idx][i][0], pos[idx][i][1] - crop]

        Yc, Xc = np.ogrid[:yhigh, :xhigh]
        thresh_min = threshold_mean(ODImage)
        binary_min = ODImage > thresh_min
        regions = regionprops(meas_label(binary_min))

        area = []
        for region in regions:
            area.append(region.area) 
        area = np.asarray(area)
        region = regions[(area == np.max(area)).nonzero()[0][0]]

        ceny, cenx = region.centroid
        circ_r = region.major_axis_length / 2

        mask = np.sqrt((Xc - cenx)**2 + (Yc - ceny)**2) <= circ_r
        cropped_ODImage = ODImage*mask

        sample['Original Data Size'] = fullimgsize
        sample['cloud_data'] = ODImage
        sample['masked_data'] = cropped_ODImage
        sample['label'] = label
        sample['filename'] = data_path
        sample['class_dir'] = 'Vortex_' + str(label)
        if pos is not None:
            #re-sort to make the position ordering match that of the labeling function
            labels = vortex_labels_func(pos[idx], xdim=fullimgsize[1], ydim=fullimgsize[0])
            sample['positions'] = np.array(vortex_labels_func(labels, xdim=fullimgsize[1], ydim=fullimgsize[0]))
        
        data_samples += [sample]
    
    return data_samples