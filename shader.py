import numpy as np
from PIL import Image

class RBGImage():
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.data =  np.zeros((self.height, self.width, 3), np.uint8)
    def set(self, x: int, y: int, color: np.array):
        assert(x < self.width and y < self.height)
        self.data[self.height - 1 - y,x,:] = color
    def get(self, x: int, y: int) -> np.array:
        assert(x < self.width)
        assert(y < self.height)
        return self.data[self.height - 1 - y,x,:]
    def save_as_png(self, filename: str):
        img = Image.fromarray(self.data)
        img.save(filename)

def line_v1(ax : int, ay : int, bx : int, by : int, framebuffer : np.ndarray, color : np.array):
    t = 0.0
    while t <= 1.0:
        x = int(ax + (bx - ax) * t)
        y = int(ay + (by - ay) * t)
        print(x, y)
        framebuffer[x,y] = color
        t += 0.02
def line(ax : int, ay : int, bx : int, by : int, framebuffer : np.ndarray, color : np.array):
    steep = abs(ax - bx) < abs(ay - by)
    if steep:
        temp = ax
        ax = ay
        ay = temp
        temp = bx
        bx = by
        by = temp
    if ax > bx:
        temp = ax
        ax = bx
        bx = temp
        temp = ay
        ay = by
        by = temp

    if ax == bx:
        return
    for x in range(ax, bx + 1):
        t = float(x - ax) / float(bx - ax)
        y = int(ay + (by - ay) * t)
        #print(t, x, y)
        if steep:
            framebuffer[y,x] = color
        else:
            framebuffer[x,y] = color
def rasterize(ax, ay, bx, by):
    steep = abs(ax - bx) < abs(ay - by)
    if steep:
        temp = ax
        ax = ay
        ay = temp
        temp = bx
        bx = by
        by = temp
    if ax > bx:
        temp = ax
        ax = bx
        bx = temp
        temp = ay
        ay = by
        by = temp
    line = []
    for x in range(ax, bx + 1):
        t = float(x - ax) / float(bx - ax)
        y = int(ay + (by - ay) * t)
        if steep:
            line.append((y, x))
        else:
            line.append((x, y))
    return line
def triangle(ax, ay, bx, by, cx, cy, framebuffer : np.ndarray, color : np.array):
    line(ax, ay, bx, by, framebuffer, color)
    line(bx, by, cx, cy, framebuffer, color)
    line(cx, cy, ax, ay, framebuffer, color)
def fill_triangle(ax, ay, bx, by, cx, cy, framebuffer : np.ndarray, color : np.array):
    def sort_key(item):
        return item[1]
    verts = sorted([(ax,ay), (bx,by), (cx,cy)], key=sort_key)
    print(verts)
    # connect lowest point with 2nd lowest -> array of pixels
    # connect lowest point to highest point -> array of pixels
    ax, ay = verts[0]
    bx, by = verts[1]
    line1 = rasterize(ax, ay, bx, by)
    ax, ay = verts[1]
    bx, by = verts[2]
    line1.extend(rasterize(ax, ay, bx, by))
    ax, ay = verts[0]
    bx, by = verts[2]
    line2 = rasterize(ax, ay, bx, by)
    points_dict1 = {}
    for x,y in line1:
        existing = points_dict1.get(y)
        if existing is not None:
            if x > existing:
                points_dict1[y] = x
        else:
            points_dict1[y] = x

    for p in line2:
        line(p[0], p[1], points_dict1[p[1]], p[1], framebuffer, color)

def fill_triangle_2(ax, ay, bx, by, cx, cy, framebuffer : RBGImage, color : np.array):
    # sort by y values
    if ay > by:
        ay, by = by, ay
        ax, bx = bx, ax
    if by > cy:
        by, cy = cy, by
        bx, cx = cx, bx
    if ay > by:
        ay, by = by, ay
        ax, bx = bx, ax
    # lower half: from lowest vertex to second lowest
    total_height = cy - ay
    segment_height = by - ay
    for y in range(ay, by + 1):
        x1 = ax + (cx - ax) * (y - ay) / total_height
        x2 = ax + (bx - ax) * (y - ay) / segment_height
        if x1 > x2:
            x1, x2 = x2, x1
        for x in range(int(x1), int(x2) + 1):
            framebuffer.set(x, y, color)
    segment_height = cy - by
    #ax = ax + (cx - ax) * (by - ay) / total_height
    for y in range(by, cy + 1):
        x1 = bx + (cx - bx) * (y - by) / segment_height
        x2 = ax + (cx - ax) * (y - ay) / total_height
        if x1 > x2:
            x1, x2 = x2, x1
        for x in range(int(x1), int(x2) + 1):
            framebuffer.set(x, y, color)
def signed_triangle_area(ax,ay,bx,by,cx,cy):
    return .5*((by-ay)*(bx+ax) + (cy-by)*(cx+bx) + (ay-cy)*(ax+cx))
def fill_triangle_3(ax,ay,az, bx, by,bz, cx, cy,cz, framebuffer : RBGImage, zbuffer : RBGImage,colorA, colorB, colorC : np.array):
    bbox_x1 = min(ax, bx, cx)
    bbox_y1 = min(ay, by, cy)
    bbox_x2 = max(ax, bx, cx)
    bbox_y2 = max(ay, by, cy)
    total_area = signed_triangle_area(ax,ay,bx,by,cx,cy)
    for x in range(int(bbox_x1), int(bbox_x2) + 1):
        for y in range(int(bbox_y1), int(bbox_y2) + 1):
            alpha = signed_triangle_area(x,y,bx,by,cx,cy) / total_area
            beta = signed_triangle_area(ax,ay,x,y,cx,cy) / total_area
            gamma = signed_triangle_area(ax,ay,bx,by,x,y) / total_area
            if alpha < 0 or beta < 0 or gamma < 0:
                continue
            z = alpha * az + beta * bz + gamma * cz
            z_color = (z + 1)*255/2
            if zbuffer.get(x,y)[0] >= z_color:
                continue
            color = alpha*np.array(colorA) + beta*np.array(colorB) + gamma*np.array(colorC)
            zbuffer.set(x,y, [z_color,z_color,z_color])
            framebuffer.set(x,y, color)


def draw_from_obj_file():
    H,W = 1024,1024
    blue = [0,0,255]
    red = [255,0,0]
    green = [0,255,0]
    yellow = [255,200,0]
    pixel_buffer = RBGImage(W,H)
    z_buffer = RBGImage(W,H)
    vertices = []
    faces = []
    with open("diablo3_pose.obj", "r") as f:
        for line in f:
            line = line.strip()
            parts = line.split()
            if len(parts) == 0:
                continue
            if parts[0] == "v":
                vertices.append(np.array([float(parts[1]), float(parts[2]), float(parts[3])]))
                #print(vertices[-1])
            elif parts[0] == "f":
                v_indices = [ int(part.split('/')[0]) - 1 for part in parts[1:] ]
                faces.append(v_indices)
                #print(v_indices)

    #faces = sorted(faces, key = lambda x: max(vertices[x[0]][2], vertices[x[1]][2], vertices[x[2]][2]), reverse = False)
    for i,j,k in faces:
        c = 5
        ax,ay,az = (vertices[i][0]/(1-vertices[i][2]/c) + 1)*W/2, (vertices[i][1]/(1-vertices[i][2]/c) + 1)*H/2, vertices[i][2]
        bx,by,bz = (vertices[j][0]/(1-vertices[j][2]/c)  + 1)*W/2, (vertices[j][1]/(1-vertices[j][2]/c)  + 1)*H/2, vertices[j][2]
        cx,cy,cz = (vertices[k][0]/(1-vertices[k][2]/c)  + 1)*W/2, (vertices[k][1]/(1-vertices[k][2]/c)  + 1)*H/2, vertices[k][2]
        fill_triangle_3(ax,ay,az,bx,by,bz,cx,cy,cz,pixel_buffer,z_buffer, red, green, blue)


    pixel_buffer.save_as_png("shader_test.png")
    z_buffer.save_as_png("shader_depth.png")

H,W = 128,128
blue = [0,0,255]
red = [255,0,0]
green = [0,255,0]
yellow = [255,200,0]
pixel_buffer = np.zeros((H,W,3), dtype=np.uint8)
ax, ay = 20, 15
bx, by = 48, 80
cx, cy = 75, 48
framebuffer = RBGImage(W,H)
#fill_triangle_3(ax, ay, bx, by, cx, cy, framebuffer, red, green, blue)
framebuffer.save_as_png("shader_test.png")
draw_from_obj_file()
