import cv2
import numpy as np
import math
import torch 
from scipy.ndimage.morphology import distance_transform_edt

class Line(object):
    def __init__(self, coordinates=[0, 0, 1, 1]):
        """
        coordinates: [y0, x0, y1, x1]
        """
        assert isinstance(coordinates, list)
        assert len(coordinates) == 4
        assert coordinates[0]!=coordinates[2] or coordinates[1]!=coordinates[3]
        self.__coordinates = coordinates

    @property
    def coord(self):
        return self.__coordinates

    @property
    def length(self):
        start = np.array(self.coord[:2])
        end = np.array(self.coord[2::])
        return np.sqrt(((start - end) ** 2).sum())

    def angle(self):
        y0, x0, y1, x1 = self.coord
        if x0 == x1:
            return -np.pi / 2
        return np.arctan((y0-y1) / (x0-x1))

    def rescale(self, rh, rw):
        coor = np.array(self.__coordinates)
        r = np.array([rh, rw, rh, rw])
        self.__coordinates = np.round(coor * r).astype(np.int).tolist()

    def __repr__(self):
        return str(self.coord)


class LineAnnotation(object):
    def __init__(self, size, lines, divisions=12):
        # assert isinstance(lines, Line)
        # assert isinstance(size, Line)
        assert divisions > 1
        assert size[0] > 1 and size[1] > 1
        self.size = size
        self.divisions = divisions
        self.lines = lines
        # binary mask with shape [H, W]
        self.__mask = None
        # oriental mask with shape [ndivision, H, W]
        self.__oriental_mask = None
        # oriental mask only with angle [H, W]
        self.__angle_mask = None
        # regression label [distance_regression, oriental_regrression] with shape [H, W, 2] and [H, W, ndivision]
        self.__regression_label = None
        # the offset of non-line pixels to line pixels
        self.__offset = None

    def mask(self):
        if self.__mask is None:
            self.__mask = line2mask(self.size, self.lines)
        return self.__mask
    
    def oriental_mask(self):
        if self.__oriental_mask is None:
            mask = self.mask()
            oriental_mask_ = np.zeros([self.divisions] + self.size, np.uint8)
            angle_mask_ = np.zeros(self.size, np.uint8)
            for idx, l in enumerate(self.lines):
                mask1 = mask == (idx+1)
                orient = round(( l.angle() + np.pi/2 ) / (np.pi / self.divisions)) % self.divisions # 0, 1, ..., 11
                assert orient >= 0 and orient < self.divisions
                oriental_mask_[int(orient), mask1] = 1
                angle_mask_[mask1] = orient
            self.__oriental_mask = oriental_mask_
            self.__angle_mask = angle_mask_
        return self.__oriental_mask

    def angle_mask(self):
        if self.__angle_mask is None:
            angle_mask = self.oriental_mask
        return self.__angle_mask

    def regression_label(self):
        if self.__regression_label is None:
            # reg_oriental_label = np.zeros(self.size+[self.divisions], dtype=np.float)
            angle = np.zeros(self.size+[self.divisions]).reshape(-1, self.divisions)
            reg_distance_label = np.zeros(self.size+[2], dtype=np.float)
            orient = np.zeros(len(self.lines))
            dist_pre_line = np.zeros([len(self.lines)]+self.size)
            mask = self.mask()
            for idx, l in enumerate(self.lines):
                dist_pre_line[idx] = distance_transform_edt(mask != (idx+1))
                orient[idx] = l.angle()
            _, [indicesY, indicesX] = distance_transform_edt(mask==0, return_indices=True)
            dx = indicesX - np.tile(range(self.size[1]), (self.size[0], 1))
            dy = indicesY - np.tile(range(self.size[0]), (self.size[1], 1)).transpose()
            theta = orient[np.argmin(dist_pre_line, 0).reshape(-1)] # [H*W]
            angle[:] = [-np.pi/2 + k*np.pi / self.divisions for k in range(self.divisions)]
            d_theta = theta - angle.transpose()
            reg_oriental_label = d_theta.reshape([-1]+self.size).transpose()
            reg_distance_label[:,:,1] = dx
            reg_distance_label[:,:,0] = dy
            self.__regression_label = [reg_distance_label, reg_oriental_label]
        return self.__regression_label
    
    def offset(self):
        if self.__offset is None:
            mask = self.__mask.astype(bool)
            H, W = mask.shape
            bw_dist, bw_idx = distance_transform_edt(np.logical_not(mask), return_indices=True)

            tmp0 = np.arange(H).reshape(H, 1).repeat(W, 1).reshape(1, H, W)
            tmp1 = np.arange(W).reshape(1, W).repeat(H, 0).reshape(1, H, W)
            xys = np.concatenate((tmp0, tmp1), axis=0)
            offset = bw_idx - xys
            # check corectness
            x, y = np.random.choice(W), np.random.choice(H)
            assert np.sqrt((offset[:, y, x]**2).sum()) == bw_dist[y, x]

        return self.__offset

    def normed_offset(self):
        mask = self.__mask.astype(bool)
        bw_dist, _ = distance_transform_edt(np.logical_not(mask), return_indices=True)
        bw_dist[bw_dist == 0] = 1

        return self.offset() / bw_dist

    def rescale(self, r):
        """
        Downsample annotations
        """
        assert r > 0 and (isinstance(r, int) or isinstance(r, float))

        for l in self.lines:
            l.rescale(rh=1/r, rw=1/r)

        self.size = (np.array(self.size) / r).astype(np.int).tolist()

        self.__mask = None
        self.__oriental_mask = None
        self.__angle_mask = None
        self.__regression_label = None

    def resize(self, size):
        H, W = size
        rH = H / self.size[0]
        rW = W / self.size[1]

        self.size = [H, W]

        for l in self.lines:
            l.rescale(rh=rH, rw=rW)

        self.__mask = None
        self.__oriental_mask = None
        self.__angle_mask = None
        self.__regression_label = None



def line2mask(size, lines):
    H, W = size
    mask = np.zeros((H, W), np.uint8)
    for idx, l in enumerate(lines):
        y0, x0, y1, x1 = l.coord
        cv2.line(mask, (x0, y0), (x1, y1), (idx+1), 2)
    return mask

def get_boundary_point(y, x, angle, H, W):
    '''
    Given point y,x with angle, return a two point in image boundary with shape [H, W]
    return point:[x, y]
    '''
    point1 = None
    point2 = None
    
    if angle == -np.pi / 2:
        point1 = (x, 0)
        point2 = (x, H-1)
    elif angle == 0.0:
        point1 = (0, y)
        point2 = (W-1, y)
    else:
        k = np.tan(angle)
        if y-k*x >=0 and y-k*x < H:  #left
            if point1 == None:
                point1 = (0, int(y-k*x))
            elif point2 == None:
                point2 = (0, int(y-k*x))
                if point2 == point1: point2 = None
        # print(point1, point2)
        if k*(W-1)+y-k*x >= 0 and k*(W-1)+y-k*x < H: #right
            if point1 == None:
                point1 = (W-1, int(k*(W-1)+y-k*x))
            elif point2 == None:
                point2 = (W-1, int(k*(W-1)+y-k*x)) 
                if point2 == point1: point2 = None
        # print(point1, point2)
        if x-y/k >= 0 and x-y/k < W: #top
            if point1 == None:
                point1 = (int(x-y/k), 0)
            elif point2 == None:
                point2 = (int(x-y/k), 0)
                if point2 == point1: point2 = None
        # print(point1, point2)
        if x-y/k+(H-1)/k >= 0 and x-y/k+(H-1)/k < W: #bottom
            if point1 == None:
                point1 = (int(x-y/k+(H-1)/k), H-1)
            elif point2 == None:
                point2 = (int(x-y/k+(H-1)/k), H-1)
                if point2 == point1: point2 = None
        # print(int(x-y/k+(H-1)/k), H-1)
        if point2 == None : point2 = point1
    return point1, point2

# def proposal2line(y, x, angle, size, num_directions=12):
#     '''
#     y, x, angle are the proposal point and angle. 
#     '''
#     assert angle >= 0 and angle < num_directions
#     H, W = size
#     angle = int2arc(angle, num_directions)
#     point1, point2 = get_boundary_point(y, x, angle, H, W)
#     if point1 == None or point2 == None:
#         print(y, x, angle, H, W)
#     return Line(coordinates=[point1[1], point1[0], point2[1], point2[0]])

# def proposal2coords(proposal):
#     N, C, H, W = proposal.size()
    
#     proposal = proposal.detach().cpu().numpy()
#     batch_coords = []
#     for b in range(N):
#         indexs = np.argwhere(proposal[b, ...])
#         select_num = indexs.shape[0]
        
#         if select_num == 0:
#             batch_coords.append(None)
#             continue

#         coords = torch.zeros((select_num, 5))
#         for idx, (c, y, x) in enumerate(indexs):
#             (x1, y1), (x2, y2) = get_boundary_point(y, x, int2arc(c, 12), H, W)
#             coords[idx, 0] = float(x1)
#             coords[idx, 1] = float(y1)
#             coords[idx, 2] = float(x2)
#             coords[idx, 3] = float(y2)
#         coords = coords.cuda()
#         batch_coords.append(coords)
#     return batch_coords

# def proposal2label_mapping(proposal, label):
#     N, C, H, W = proposal.size()
    
#     proposal = proposal.detach().cpu().numpy()
#     indexs = np.argwhere(proposal)
#     select_num = indexs.shape[0]
    
#     label_mapping = torch.zeros((select_num, 1))
#     for idx, (n, c, y, x) in enumerate(indexs):
#         label_mapping[idx, 0] = label[n, c, y, x]
#     label_mapping = label_mapping.to(label)
#     return label_mapping

def int2arc(k, num_directions):
    '''
    convert int to arc system with num_directions division.
    '''
    return -np.pi / 2 + np.pi / num_directions * k

def arc2int(theta, num_directions):
    '''
    convert arc system to int with num_directions division.
    '''
    return round(( theta + np.pi/2 ) / (np.pi / num_directions)) % num_directions
