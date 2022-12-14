'''
MIT License

Copyright (c) 2022 Francesco Ioli

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import os
import cv2
import pickle
import numpy as np
import pandas as pd

# import exifread
from typing import List, Union
from scipy import linalg
from pathlib import Path

#--- Camera ---#


class Camera:
    ''' Class to help manage Cameras. '''
    # @TODO =: revise all Camera methods

    def __init__(self, width=None, height=None,
                 K=None, dist=None,
                 R=None, t=None,
                 calib_path=None
                 ):
        ''' Initialize pinhole camera model '''
        # TODO: add checks on inputs
        # If not None, convert inputs to np array
        if K is not None:
            K = np.array(K)
        if R is not None:
            R = np.array(R)
        if t is not None:
            t = np.array(t)
        if dist is not None:
            dist = np.array(dist)
        # TODO: add assertion to check that only K and dist OR calib_path is provided.

        self.width = width  # Image width [px]
        self.height = height  # Image height [px]
        self.K = K          # Calibration matrix (Intrisics)
        self.dist = dist    # Distortion vector in OpenCV format
        self.R = R      # rotation matrix (from world to cam)
        self.t = t      # translation vector (from world to cam)
        self.P = None   # Projection matrix (from world to cam)
        self.C = None   # camera center (in world coordinates)
        self.pose = None    # Pose matrix
        # (describes change of basis from camera to world)
        self.extrinsics = None  # Extriniscs matrix
        # (describes change of basis from world to cam)

        # If calib_path is provided, read camera calibration from file
        if calib_path is not None:
            self.read_calibration_from_file(calib_path)

        if R is None and t is None:
            self.reset_EO()
            # self.compose_P()
            # self.C_from_P()

    def reset_EO(self):
        ''' Reset camera EO as to make camera reference system parallel to world reference system '''
        self.extrinsics = np.eye(4)
        self.update_camera_from_extrinsics()
        self.extrinsics_to_pose()
        self.C_from_P()
        # self.R = np.identity(3)
        # self.t = np.zeros((3,)).reshape(3,1)
        # self.P = P_from_KRT(self.K, self.R, self.t)
        # self.C_from_P()

    def build_camera_EO(self,
                        extrinsics: np.ndarray = None,
                        pose: np.ndarray = None
                        ) -> None:

        if extrinsics is not None:
            self.extrinsics = extrinsics
            self.extrinsics_to_pose()
            self.update_camera_from_extrinsics()

        elif pose is not None:
            self.pose = pose
            self.pose_to_extrinsics()
            self.update_camera_from_extrinsics()

        else:
            raise ValueError(
                'Not enough data to build Camera External Orientation matrixes.')

    def build_pose_matrix(self,
                          R: np.ndarray,
                          C: np.ndarray) -> None:
        # Check for input dimensions
        if R.shape != (3, 3):
            raise ValueError(
                'Wrong dimension of the R matrix. It must be a 3x3 numpy array')
        if C.shape == (3,) or C.shape == (1, 3):
            C = C.T
        elif C.shape != (3, 1):
            raise ValueError(
                'Wrong dimension of the C vector. It must be a 3x1 or a 1x3 numpy array')

        self.pose = np.eye(4)
        self.pose[0:3, 0:3] = R
        self.pose[0:3, 3:4] = C

    def Rt_to_extrinsics(self):
        '''
        [ R | t ]    [ I | t ]   [ R | 0 ]
        | --|-- |  = | --|-- | * | --|-- |
        [ 0 | 1 ]    [ 0 | 1 ]   [ 0 | 1 ]
        '''
        R_block = self.build_block_matrix(self.R)
        t_block = self.build_block_matrix(self.t)
        self.extrinsics = np.dot(t_block, R_block)
        return self.extrinsics

    def extrinsics_to_pose(self):
        '''
        '''
        if self.extrinsics is None:
            self.Rt_to_extrinsics()

        R = self.extrinsics[0:3, 0:3]
        t = self.extrinsics[0:3, 3:4]

        Rc = R.T
        C = -np.dot(Rc, t)

        Rc_block = self.build_block_matrix(Rc)
        C_block = self.build_block_matrix(C)

        self.pose = np.dot(C_block, Rc_block)

        return self.pose

    def pose_to_extrinsics(self):
        ''' 
        '''
        if self.pose is None:
            print('Camera pose not available. Compute it first.')
            return None
        else:
            Rc = self.pose[0:3, 0:3]
            C = self.pose[0:3, 3:4]

            R = Rc.T
            t = -np.dot(R, C)

            t_block = self.build_block_matrix(t)
            R_block = self.build_block_matrix(R)
            self.extrinsics = np.dot(t_block, R_block)
            self.update_camera_from_extrinsics()

            return self.extrinsics

    def update_camera_from_extrinsics(self):
        '''

        '''
        if self.extrinsics is None:
            print('Camera extrinsics not available. Compute it first.')
            return None
        else:
            self.R = self.extrinsics[0:3, 0:3]
            self.t = self.extrinsics[0:3, 3:4]
            self.P = np.dot(self.K, self.extrinsics[0:3, :])
            self.C_from_P()

    def get_C_from_pose(self):
        '''

        '''
        return self.pose[0:3, 3:4]

    def C_from_P(self):
        '''
        Compute and return the camera center from projection matrix P, as
        C = [- inv(KR) * Kt] = [-inv(P[1:3]) * P[4]]
        '''
        # if self.C is not None:
        #     return self.C
        # else:
        self.C = - \
            np.dot(np.linalg.inv(self.P[:, 0:3]), self.P[:, 3].reshape(3, 1))
        return self.C

    def t_from_RC(self):
        ''' Deprecrated function. Use extrinsics_to_pose instead.
        Compute and return the camera translation vector t, given the camera
        centre and the roation matrix X, as
        t = [-R * C]
        The relation is derived from the formula of the camera centre
        C = [- inv(KR) * Kt]
        '''
        self.t = -np.dot(self.R, self.C)
        self.compose_P()
        return self.t

    def compose_P(self):
        '''
        Compose and return the 4x3 P matrix from 3x3 K matrix, 3x3 R matrix and 3x1 t vector, as:
            K[R | t]
        '''
        if (self.K is None):
            print("Invalid calibration matrix. Unable to compute P.")
            self.P = None
            return None
        elif (self.R is None):
            print("Invalid Rotation matrix. Unable to compute P.")
            self.P = None
            return None
        elif (self.t is None):
            print("Invalid translation vector. Unable to compute P.")
            self.P = None
            return None

        RT = np.zeros((3, 4))
        RT[:, 0:3] = self.R
        RT[:, 3:4] = self.t
        self.P = np.dot(self.K, RT)
        return self.P

    def factor_P(self):
        ''' Factorize the camera matrix into K, R, t as P = K[R | t]. '''

        # factor first 3*3 part
        K, R = linalg.rq(self.P[:, :3])

        # make diagonal of K positive
        T = np.diag(np.sign(np.diag(K)))
        if linalg.det(T) < 0:
            T[1, 1] *= -1

        self.K = np.dot(K, T)
        self.R = np.dot(T, R)  # T is its own inverse
        self.t = np.dot(linalg.inv(self.K), self.P[:, 3]).reshape(3, 1)

        return self.K, self.R, self.t

    def project_points(self, points3d):
        '''
        Overhelmed method(see lib.geometry) for projecting 3D to image coordinates.

        Project 3D points(Nx3 array) to image coordinates, given the projection matrix P(4x3 matrix)
        If K matric and dist vector are given, the function computes undistorted image projections(otherwise, zero distortions are assumed)
        Returns: 2D projected points(Nx2 array) in image coordinates
        '''
        points3d = cv2.convertPointsToHomogeneous(points3d)[:, 0, :]
        m = np.dot(self.P, points3d.T)
        m = m[0:2, :] / m[2, :]
        m = m.astype(float).T

        if self.dist is not None and self.K is not None:
            m = cv2.undistortPoints(
                m, self.K, self.dist, None, self.K)[:, 0, :]

        return m.astype(float)

    def read_calibration_from_file(self, path):
        '''
        Read camera internal orientation from file, save in camera class
        and return them.
        The file must contain the full K matrix and distortion vector,
        according to OpenCV standards, and organized in one line, as follow:
        fx 0. cx 0. fy cy 0. 0. 1. k1, k2, p1, p2, [k3, [k4, k5, k6
        Values must be float(include the . after integers) and divided by a
        white space.
        -------
        Returns:  K, dist
        '''
        path = Path(path)
        if not path.exists():
            print('Error: calibration filed does not exist.')
            return None, None
        with open(path, 'r') as f:
            data = np.loadtxt(f)
            K = data[0:9].astype(float).reshape(3, 3, order='C')
            if len(data) == 13:
                print('Using OPENCV camera model.')
                dist = data[9:13].astype(float)
            elif len(data) == 14:
                print('Using OPENCV camera model + k3')
                dist = data[9:14].astype(float)
            elif len(data) == 17:
                print('Using FULL OPENCV camera model')
                dist = data[9:17].astype(float)
            else:
                print('invalid intrinsics data.')
                return None, None
            # TODO: implement other camera models and estimate K from exif.
        self.K = K
        self.dist = dist
        return K, dist

    def get_P_homogeneous(self):
        """
        Return the 4x4 P matrix from 3x4 P matrix, as:
            [      P     ]
            [------------]
            [ 0  0  0  1 ]
        """
        P_hom = np.eye(4)
        P_hom[0:3, 0:4] = self.P

        return P_hom

    def euler_from_R(self):
        '''
        Compute Euler angles from rotation matrix
        - ------
        Returns:  [omega, phi, kappa]
        '''
        omega = np.arctan2(self.R[2, 1], self.R[2, 2])
        phi = np.arctan2(-self.R[2, 0],
                         np.sqrt(self.R[2, 1]**2+self.R[2, 2]**2))
        kappa = np.arctan2(self.R[1, 0], self.R[0, 0])

        return [omega, phi, kappa]

    def build_block_matrix(self, mat):
        # TODO: add description
        '''

        '''
        if mat.shape[1] == 3:
            block = np.block([[mat, np.zeros((3, 1))],
                              [np.zeros((1, 3)), 1]]
                             )
        elif mat.shape[1] == 1:
            block = np.block([[np.eye(3), mat],
                              [np.zeros((1, 3)), 1]]
                             )
        else:
            print('Error: unknown input matrix dimensions.')
            return None

        return block
