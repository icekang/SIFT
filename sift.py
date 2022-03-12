from dataclasses import dataclass
from urllib import response
import numpy as np
import cv2 as cv

from numpy import histogram


@dataclass
class Point2f:
    x: float
    y: float


@dataclass
class KeyPoint:
    angle: float
    class_id: int
    octave: int
    pt: Point2f
    response: float
    size: float


class Utility:
    def derivative(self, matrix: np.array, point: tuple | list) -> np.array:
        """
        matrix: 2D or 3D array
        point: coordinates

        return vector (2, ) or (3, ) of derivative of the matrix at the given coordinates
        """
        if len(point) == 3:
            x, y, z = point
            dx = matrix[x + 1, y, z] - matrix[x - 1, y, z]
            dy = matrix[x, y + 1, z] - matrix[x, y - 1, z]
            dz = matrix[x, y, z + 1] - matrix[x, y, z - 1]
            return np.array([dx, dy, dz])

        elif len(point) == 2:
            x, y = point
            dx = matrix[x + 1, y] - matrix[x - 1, y]
            dy = matrix[x, y + 1] - matrix[x, y - 1]
            return np.array([dx, dy])
        else:
            raise ValueError("point must be a tuple or list of length 2 or 3")

    def hessian(self, matrix: np.array, point: tuple) -> np.array:
        gradientx = np.zeros((len(point), len(point)))
        gradienty = np.zeros((len(point), len(point)))
        if len(point) == 3:
            a, b, c = matrix.shape
            assert a == b == c == 5
            gradientz = np.zeros((len(point), len(point)))
            for i in range(1, 4):
                for j in range(1, 4):
                    for k in range(1, 4):
                        dx, dy, dz = self.derivative(matrix, (i, j, k))
                        gradientx[i, j] = dx
                        gradienty[i, j] = dy
                        gradientz[i, j] = dz
        elif len(point) == 2:
            a, b = matrix.shape
            assert a == b == 5
            for i in range(1, 4):
                for j in range(1, 4):
                    dx, dy = self.derivative(matrix, (i, j))
                    gradientx[i, j] = dx
                    gradienty[i, j] = dy

        if len(point) == 3:
            dxx, dxy, dxz = self.derivative(gradientx, (1, 1))
            dyx, dyy, dyz = self.derivative(gradienty, (1, 1))
            dzx, dzy, dzz = self.derivative(gradientz, (1, 1))
            hessian = np.array([
                [dxx, dxy, dxz],
                [dyx, dyy, dyz],
                [dzx, dzy, dzz],
            ])
        elif len(point) == 2:
            dxx, dxy = self.derivative(gradientx, (1, 1))
            dyx, dyy = self.derivative(gradienty, (1, 1))
            hessian = np.array([
                [dxx, dxy],
                [dyx, dyy],
            ])

        return hessian


class SIFT:
    def detect(self, img):
        """Detect keypoints of an image
        Args:
            img (cv.image): grayscale image

        Returns:
            kp: list of keypoints
        """
        HEIGH, WIDTH, _channels = map(int, src.shape)

        K = np.sqrt(2)  # as suggested [Lowe 2004]
        SIGMA = 1.6  # as suggested [Lowe 2004]
        N_OCTAVES = 4  # as suggested [Lowe 2004]
        DOWN_SAMPLING_FACTOR = 2  # as suggested [Lowe 2004]
        SCALE_LEVEL = 5  # as suggested [Lowe 2004]
        REJECTING_CRITERION = 0.03  # as suggested [Lowe 2004]
        PRINCIPAL_CURVATURE_THRESHOLD = 10  # as suggested [Lowe 2004]

        kps = []
        des = []

        # Create Gaussian Scale-Space / Gaussin Pyramid Images: pyramids[octaves][scale_level]
        src = img.copy()
        pyramids = []
        for octave in range(N_OCTAVES):
            if octave > 0:
                src = cv.pyrDown(
                    src,
                    dstsize=(
                        WIDTH // DOWN_SAMPLING_FACTOR,
                        HEIGH // DOWN_SAMPLING_FACTOR,
                    ),
                )
            pyramid = []
            for scale in (SIGMA * (K**i) for i in range(SCALE_LEVEL)):
                dst = cv.GaussianBlur(src, ksize=(0, 0), sigmaX=scale)
                pyramid.append(dst)
            pyramids.append(pyramid)

        # Create DoG Pyramid Images: difference_of_gaussians[octave][z,h,w]
        # z = 0, 1, 2, ..., SCALE_LEVEL - 2
        # h = 0, 1, ..., HEIGHT // DOWN_SAMPLING_FACTOR**z - 1
        # w = 0, 1, ..., WIDTH // DOWN_SAMPLING_FACTOR**z - 1
        difference_of_gaussians = []
        for octave in range(N_OCTAVES):
            difference_of_gaussian = []
            for scale_level in range(1, SCALE_LEVEL):
                dog = K - 1
                dog *= pyramids[octave][scale_level] - pyramids[octave][scale_level - 1]
                difference_of_gaussian.append(dog)
            difference_of_gaussians.append(np.array(difference_of_gaussian))

        # 3. Detection of scale-space extrema: Finding extrema of DoG images
        window = 3
        step = 1
        for octave in range(N_OCTAVES):
            for z in range((SCALE_LEVEL - 1) - window + step):
                scale = SIGMA * (K**z)
                for h in range(HEIGH // DOWN_SAMPLING_FACTOR**z - window):
                    for w in range(WIDTH // DOWN_SAMPLING_FACTOR**z - window):
                        # TODO scale image to the original size
                        neighborhood = difference_of_gaussians[octave][
                            z : z + window, h : h + window, w : w + window
                        ]
                        neighborhoodH = difference_of_gaussians[octave][
                            z : z + 5, h : h + 5, w : w + 5
                        ]
                        if neighborhood[1, 1, 1] == np.max(neighborhood, axis=None):
                            x = np.array([h + 1, w + 1, z])

                            # 4. Accurate keypoint localization
                            middle_coor = (1, 1, 1)
                            middle_coorH = (2, 2, 2)
                            gradient = Utility.derivative(neighborhood, middle_coor)
                            hessian = Utility.hessian(
                                neighborhoodH,
                                middle_coorH
                            )
                            offset = -np.linalg.inv(hessian) @ gradient
                            response = neighborhood[1, 1, 1] + 0.5 * np.inner(
                                gradient, offset
                            )
                            if np.abs(response) < REJECTING_CRITERION:
                                continue

                            # 4.1 Eliminating edge responses
                            hessian2D = Utility.hessian(
                                neighborhoodH[:, :, 2], middle_coorH[:2]
                            )
                            tr_hessian2D = np.trace(hessian2D)
                            det_hessian2D = np.linalg.det(hessian2D)
                            if (
                                tr_hessian2D**2 / det_hessian2D
                                >= (PRINCIPAL_CURVATURE_THRESHOLD + 1) ** 2
                                / PRINCIPAL_CURVATURE_THRESHOLD
                            ):
                                continue

                            # 5. Orientation assignment
                            L = (
                                pyramids[octave][z] / scale
                            )  # should now be scale invariant
                            gradients = np.zeros(L.shape)
                            orientations = np.zeros(L.shape)
                            for h in range(HEIGH // DOWN_SAMPLING_FACTOR**z - window):
                                for w in range(
                                    WIDTH // DOWN_SAMPLING_FACTOR**z - window
                                ):
                                    m = np.sqrt(
                                        (L[h, w + 1] - L[h, w - 1]) ** 2
                                        + (L[h + 1, w] - L[h - 1, w]) ** 2
                                    )
                                    theta = np.arctan2(
                                        L[h + 1, w] - L[h - 1, w],
                                        L[h, w + 1] - L[h, w - 1],
                                    )
                                    gradients[h, w] = m
                                    orientations[h, w] = (
                                        theta + 2 * np.pi if theta < 0 else theta
                                    )  # [0, 2pi]
                            histogram = np.histogram(orientations, bins=36, range=(0, 2 * np.pi))[0]
                            principle_orientation = np.argmax(histogram)
                            principle_orientation = 10 * principle_orientation
                            #  6. Descriptor representation
                            # Get 16x16 window matrix
                            ksize = 16
                            ksize = np.around(ksize, decimals=0)
                            # apply Gaussian window to gradients and orientations
                            g = cv.GaussianBlur(
                                gradients[
                                    h - ksize // 2 : h + ksize // 2,
                                    w - ksize // 2 : w + ksize // 2,
                                ],
                                ksize=(ksize + 1, ksize + 1),
                            )
                            g = g[1:, 1:]
                            descriptor = []
                            for i in range(4):
                                for j in range(4):
                                    r = i * ksize // 4
                                    c = j * ksize // 4
                                    o = np.histogram(
                                        g[
                                            r : r + ksize // 4,
                                            c : c + ksize // 4,
                                        ],
                                        bins=8,
                                        range=(0, 2 * np.pi),
                                    )[0]
                                    descriptor += o
                            descriptor = np.array(descriptor) # 4x4x8
                            descriptor = descriptor / np.linalg.norm(descriptor) # normalize to unit length
                            descriptor = np.clip(descriptor, a_min=0, a_max=0.2) # clip to [0, 0.2] as suggested in by Lowe 2004

                            keypoint = KeyPoint(angle=principle_orientation, class_id=-1, octave=octave, pt=np.array([h, w]) * scale, response=response, size=scale)

                            kps.append(keypoint)
                            des.append(descriptor)


        return kps, des
