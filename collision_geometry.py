import numpy as np

X_MAX = 8.08
Y_MAX = 4.48

def map_to_world(obj_pos, obj_orn, rel_p):
    rot_p = np.array([np.cos(obj_orn)*rel_p[0]-np.sin(obj_orn)*rel_p[1],
                      np.sin(obj_orn)*rel_p[0]+np.cos(obj_orn)*rel_p[1]])
    return rot_p + obj_pos

class Line:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2

    @staticmethod
    def onSegment(p, q, r):
        if ( (q[0] <= max(p[0], r[0])) and (q[0] >= min(p[0], r[0])) and
            (q[1] <= max(p[1], r[1])) and (q[1] >= min(p[1], r[1]))):
            return True
        return False

    @staticmethod
    def orientation(p, q, r):
        # to find the orientation of an ordered triplet (p,q,r)
        # function returns the following values:
        # 0 : Collinear points
        # 1 : Clockwise points
        # 2 : Counterclockwise
        
        # See https://www.geeksforgeeks.org/orientation-3-ordered-points/amp/
        # for details of below formula.
        
        val = (float(q[1] - p[1]) * (r[0] - q[0])) - (float(q[0] - p[0]) * (r[1] - q[1]))
        if (val > 0):
            
            # Clockwise orientation
            return 1
        elif (val < 0):
            
            # Counterclockwise orientation
            return 2
        else:
            
            # Collinear orientation
            return 0
 
    # The main function that returns true if
    # the line segment 'p1q1' and 'p2q2' intersect.
    @staticmethod
    def doIntersect(p1,q1,p2,q2):
        
        # Find the 4 orientations required for
        # the general and special cases
        o1 = Line.orientation(p1, q1, p2)
        o2 = Line.orientation(p1, q1, q2)
        o3 = Line.orientation(p2, q2, p1)
        o4 = Line.orientation(p2, q2, q1)
    
        # General case
        if ((o1 != o2) and (o3 != o4)):
            return True
    
        # Special Cases
    
        # p1 , q1 and p2 are collinear and p2 lies on segment p1q1
        if ((o1 == 0) and Line.onSegment(p1, p2, q1)):
            return True
    
        # p1 , q1 and q2 are collinear and q2 lies on segment p1q1
        if ((o2 == 0) and Line.onSegment(p1, q2, q1)):
            return True
    
        # p2 , q2 and p1 are collinear and p1 lies on segment p2q2
        if ((o3 == 0) and Line.onSegment(p2, p1, q2)):
            return True
    
        # p2 , q2 and q1 are collinear and q1 lies on segment p2q2
        if ((o4 == 0) and Line.onSegment(p2, q1, q2)):
            return True
    
        # If none of the cases
        return False

    def intersect(self, line2):
        return Line.doIntersect(self.p1, self.p2, line2.p1, line2.p2)

class Box:
    def __init__(self, center, lx, ly, theta):
        self.p1 = map_to_world(center, theta, np.array([-lx/2, -ly/2]))
        self.p2 = map_to_world(center, theta, np.array([ lx/2, -ly/2]))
        self.p3 = map_to_world(center, theta, np.array([lx/2, ly/2]))
        self.p4 = map_to_world(center, theta, np.array([-lx/2, ly/2]))
        self.lines = []
        self.lines.append(Line(self.p1, self.p2))
        self.lines.append(Line(self.p2, self.p3))
        self.lines.append(Line(self.p3, self.p4))
        self.lines.append(Line(self.p4, self.p1))

    def line_intersect(self, line2):
        for line in self.lines:
            if line.intersect(line2):
                return True
        return False

# Need to know where random blocks are
# random_obstacles: [np.array([Ox, Oy]),...]
class Map:
    def __init__(self):
        self.obstacles = []
        self.num_static = 9
        self.obstacles.append(Box(np.array([1.9, 2.24]), 0.8, 0.2, 0)) # B8
        self.obstacles.append(Box(np.array([4.34, 2.24]), 0.25, 0.25, np.pi/4)) # B5
        self.obstacles.append(Box(np.array([4.04, 1.035]), 1.0, 0.2, 0)) # B4
        self.obstacles.append(Box(np.array([4.04, Y_MAX-1.035]), 1.0, 0.2, 0)) # B6
        self.obstacles.append(Box(np.array([X_MAX-1.9, 2.24]), 0.8, 0.2, 0)) # B2
        self.obstacles.append(Box(np.array([1.6, 0.5]), 0.2, 1.0, 0)) # B7
        self.obstacles.append(Box(np.array([0.5, Y_MAX-1.1]), 1.0, 0.2, 0)) # B9
        self.obstacles.append(Box(np.array([X_MAX-1.6, Y_MAX-0.5]), 0.2, 1.0, 0)) # B3
        self.obstacles.append(Box(np.array([X_MAX-0.5, 1.1]), 1.0, 0.2, 0)) #B1


    def add_obstacles(self, random_obstacles):
        # Add random obstacles
        for ro in random_obstacles:
            self.obstacles.append(Box(np.array([ro[0], ro[1]]), 0.3, 0.3, 0))

    def remove_rand_obstacles(self):
        if len(self.obstacles) == self.num_static:
            return 
        else:
            self.obstacles = self.obstacles[:-5]
        
    def line_intersect(self, line2):
        for o in self.obstacles:
            if o.line_intersect(line2):
                return True
        return False

if __name__ == "__main__":
    import time
    map = Map([])
    line2 = Line(np.array([1.2, 1.2]), np.array([X_MAX-1.2, Y_MAX-1.2]))
    t = time.time()
    print(map.line_intersect(line2))
    print(time.time()-t)

        