import numpy as np
import cv2
import math

def load_cube_faces(face_paths):
    faces = {}
    names = ['right', 'left', 'top', 'bottom', 'front', 'back']
    for name, path in zip(names, face_paths):
        faces[name] = cv2.imread(path)
    return faces

def cube_to_equirect(faces, output_width=1024, output_height=512):
    face_size = faces['front'].shape[0]  # assume square faces
    output = np.zeros((output_height, output_width, 3), dtype=np.uint8)

    for y in range(output_height):
        for x in range(output_width):
            theta = (x / output_width) * 2 * math.pi - math.pi  # -π to π
            phi = (y / output_height) * math.pi - (math.pi / 2)  # -π/2 to π/2

            # Convert spherical to 3D Cartesian coords
            vx = math.cos(phi) * math.sin(theta)
            vy = math.sin(phi)
            vz = math.cos(phi) * math.cos(theta)

            abs_v = np.abs([vx, vy, vz])
            max_axis = np.argmax(abs_v)

            if max_axis == 0:  # X major axis
                face = 'right' if vx > 0 else 'left'
                sc = -vz / abs_v[0]
                tc = -vy / abs_v[0]
            elif max_axis == 1:  # Y major axis
                face = 'top' if vy > 0 else 'bottom'
                sc = vx / abs_v[1]
                tc = vz / abs_v[1]
            else:  # Z major axis
                face = 'front' if vz > 0 else 'back'
                sc = vx / abs_v[2]
                tc = -vy / abs_v[2]

            # Convert [-1,1] to [0,face_size)
            u = int(((sc + 1) / 2) * face_size)
            v = int(((tc + 1) / 2) * face_size)
            u = np.clip(u, 0, face_size - 1)
            v = np.clip(v, 0, face_size - 1)

            output[y, x] = faces[face][v, u]

    return output

# Example usage
face_paths = [
    'results/equi_RIGHT.png', 'results/equi_LEFT.png',
    'results/equi_TOP.png', 'results/equi_BOTTOM.png',
    'results/equi_FRONT.png', 'results/equi_BACK.png'
]
faces = load_cube_faces(face_paths)
print("AAAAAA: ", faces)
equirect_image = cube_to_equirect(faces, output_width=2048, output_height=1024)
cv2.imwrite('equirect_output.jpg', equirect_image)
