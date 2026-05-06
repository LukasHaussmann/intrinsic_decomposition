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

# transformation matrix from world to camera space
def lookat(eye, center, up):
    n = (eye - center)
    n = n / np.linalg.norm(n)
    l = np.cross(up,n)
    l = l / np.linalg.norm(l)
    m = np.cross(n,l)
    m = m / np.linalg.norm(m)
    return np.array([np.append(l, 0), np.append(m, 0), np.append(n, 0), [0,0,0,1]]) @ np.array([[1,0,0,-center[0]],[0,1,0,-center[1]],[0,0,1,-center[2]],[0,0,0,1]])

# transform from [-1,1] to screen dimensions
def viewport(x, y, w, h):
    return np.array([[w/2., 0, 0, x + w/2.],
                     [0, h/2., 0, y + h/2.],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]
                     ])
def perspective(focal):
    return np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, -1/focal, 1]
                     ])
class Model:
    def __init__(self, filename):
        self.vertices = []
        self.vertex_normals = []
        self.faces = []
        with open(filename, "r") as f:
            for line in f:
                line = line.strip()
                parts = line.split()
                if len(parts) == 0:
                    continue
                if parts[0] == "v":
                    self.vertices.append(np.array([float(parts[1]), float(parts[2]), float(parts[3])]))
                elif parts[0] == "f":
                    v_indices = [ int(part.split('/')[0]) - 1 for part in parts[1:] ]
                    normal_indices = [ int(part.split('/')[2]) - 1 for part in parts[1:] ]
                    self.faces.append((v_indices,normal_indices))
                elif parts[0] == "vn":
                    self.vertex_normals.append(np.array([float(parts[1]), float(parts[2]), float(parts[3])]))

    def vert(self, face : int, vertex : int):
        return self.vertices[self.faces[face][0][vertex]]
    def normal(self, face : int, vertex : int):
        return self.vertex_normals[self.faces[face][0][vertex]]
class Camera:
    def __init__(self, eye, center, up):
        self.eye = eye
        self.center = center
        self.up = up
class Shader:
    def __init__(self, model : Model, camera : Camera):
        self.model = model
        self.ModelView = lookat(camera.eye, camera.center, camera.up)
        self.Perspective = perspective(np.linalg.norm(camera.eye - camera.center))
        self.triangle = [np.zeros(3), np.zeros(3), np.zeros(3)]
        self.tri_normals = [np.zeros(3), np.zeros(3), np.zeros(3)]
        self.color = None
    def vertex(self, face : int , vertex : int):
        vert = self.model.vert(face, vertex) # vertex in world coordinates
        normal = self.model.normal(face, vertex) # vertex in world coordinates
        eye_coord = self.ModelView @ np.append(vert, 1)
        self.triangle[vertex] = eye_coord[:3] # vert in eye coordinates
        self.tri_normals[vertex] = normal
        return  self.Perspective @ eye_coord
    def fragment(self, bc):
        return self.color
class PhongShader(Shader):
    def __init__(self, model : Model, camera : Camera, lightsource, shininess):
        super().__init__(model, camera)
        self.lightsource = lightsource / np.linalg.norm(lightsource)
        self.shininess = shininess
        self.camera_eye = camera.eye
        self.ambient = np.array([20,20,20])
        self.diffuse = np.array([20,20,20])
        self.specular = np.array([20,20,20])

    def fragment(self, bc):
        #normal = np.cross(self.triangle[0] - self.triangle[1], self.triangle[0] - self.triangle[2])
        normal = self.tri_normals[0] * bc[0] + self.tri_normals[1] * bc[1] + self.tri_normals[2] * bc[2]
        normal = normal / np.linalg.norm(normal)

        diffuse_intensity = max(0,normal.T @ self.lightsource)
        frag_point = self.triangle[0] * bc[0] + self.triangle[1] * bc[1] + self.triangle[2] * bc[2]
        viewdir = self.camera_eye - frag_point
        viewdir = viewdir / np.linalg.norm(viewdir)
        reflect_angle = 2 * normal * (normal.T @ self.lightsource) - self.lightsource
        reflect_angle = reflect_angle / np.linalg.norm(reflect_angle)
        specular_intensity = max(0,reflect_angle.T @ viewdir) ** self.shininess
        return np.array([255,255,255]) * min(1,.3 + .4*diffuse_intensity + .9*specular_intensity)
        return self.ambient + self.diffuse * diffuse_intensity + self.specular * specular_intensity


def rasterize(clip, shader : Shader, framebuffer : RBGImage, zbuffer : RBGImage, viewport_transform):
    a,b,c = clip[0], clip[1], clip[2]
    a,b,c = a/a[3], b/b[3], c/c[3] # normalize
    a,b,c = viewport_transform @ a, viewport_transform @ b, viewport_transform @ c # transform to screen coordinates
    ax, ay, az = a[0], a[1], a[2]
    bx, by, bz = b[0], b[1], b[2]
    cx, cy, cz = c[0], c[1], c[2]
    bbox_x1 = min(ax, bx, cx)
    bbox_y1 = min(ay, by, cy)
    bbox_x2 = max(ax, bx, cx)
    bbox_y2 = max(ay, by, cy)
    total_area = signed_triangle_area(ax,ay,bx,by,cx,cy)
    for x in range(max(0,int(bbox_x1)), min(int(bbox_x2), framebuffer.width - 1) + 1):
        for y in range(max(0,int(bbox_y1)), min(int(bbox_y2), framebuffer.height - 1) + 1):
            alpha = signed_triangle_area(x,y,bx,by,cx,cy) / total_area
            beta = signed_triangle_area(ax,ay,x,y,cx,cy) / total_area
            gamma = signed_triangle_area(ax,ay,bx,by,x,y) / total_area
            if alpha < 0 or beta < 0 or gamma < 0:
                continue
            z = alpha * az + beta * bz + gamma * cz
            z_color = (z + 1)*255/2
            if zbuffer.get(x,y)[0] >= z_color:
                continue
            color = shader.fragment([alpha,beta,gamma])
            if color is None:
                color = [0,0,0]
            zbuffer.set(x,y, [z_color,z_color,z_color])
            framebuffer.set(x,y, color)

H,W = 800,800
blue = [0,0,255]
red = [255,0,0]
green = [0,255,0]
yellow = [255,200,0]
model = Model("head.obj")
#model = Model("diablo3_pose.obj")
camera = Camera(
    eye = np.array([-1,0,4]),
    center = np.array([0,0,0]),
    up = np.array([0,1,0])
)
shader = PhongShader(model, camera, [1,1,1], 100)
framebuffer = RBGImage(W,H)
zbuffer = RBGImage(W,H)
viewport_transform = viewport(0,0,W,H)
shader.color = red
for f in range(len(model.faces)):
    clip = [shader.vertex(f, 0),shader.vertex(f, 1),shader.vertex(f, 2)]
    shader.color = np.random.randint(0, 255, 3, dtype=np.uint8)
    rasterize(clip, shader, framebuffer, zbuffer, viewport_transform)

framebuffer.save_as_png("shader_test.png")
