import cv2
import numpy as np

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

