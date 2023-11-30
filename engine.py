import numpy as np
import math as mt

class Vector2:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    @classmethod
    def zeros(cls):
        return cls(0, 0)

    @classmethod
    def ones(cls):
        return cls(1, 1)

    @classmethod
    def from_list(cls, lst):
        if len(lst) == 2:
            return cls(lst[0], lst[1])
        else:
            raise ValueError("Lista precisa receber 2 valores.")

    def to_list(self):
        return [self.x, self.y]

    def construct_matrix(self):
        return np.array([[[self.x, self.y], [4, 5]],
                         [[6, 7], [8, 9]],
                         [[10, 11], [12, 13]]])

    def linear_interpolation(self, other, t):
        if not isinstance(other, Vector2):
            raise ValueError("Interpolação necessita de 2 argumentos.")
        t = np.clip(t, 0, 1)  
        result_x = self.x + t * (other.x - self.x)
        result_y = self.y + t * (other.y - self.y)
        return Vector2(result_x, result_y)

    def magnitude(self):
        return np.sqrt(self.x**2 + self.y**2)

    def normalize(self):
        mag = self.magnitude()
        if mag == 0:
            return Vector2.zeros()
        return Vector2(self.x / mag, self.y / mag)

    def dot_product(self, other):
        if not isinstance(other, Vector2):
            raise ValueError("o produto apenas não tem 2 vetores.")
        return self.x * other.x + self.y * other.y

    def __str__(self):
        return f"Vector2({self.x}, {self.y})"


class Vector3:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    @classmethod
    def zeros(cls):
        return cls(0, 0, 0)

    @classmethod
    def ones(cls):
        return cls(1, 1, 1)

    @classmethod
    def from_list(cls, lst):
        if len(lst) == 3:
            return cls(lst[0], lst[1], lst[2])
        else:
            raise ValueError("Lista recebe mais de 3 valores.")

    def to_list(self):
        return [self.x, self.y, self.z]

    def construct_matrix(self):
        return np.array([[[self.x, self.y, self.z], [4, 5, 6]],
                         [[7, 8, 9], [10, 11, 12]],
                         [[13, 14, 15], [16, 17, 18]]])

    def linear_interpolation(self, other, t):
        if not isinstance(other, Vector3):
            raise ValueError("Interpolação só pode ser realiza entre um vector3.")
        t = np.clip(t, 0, 1)  
        result_x = self.x + t * (other.x - self.x)
        result_y = self.y + t * (other.y - self.y)
        result_z = self.z + t * (other.z - self.z)
        return Vector3(result_x, result_y, result_z)

    def magnitude(self):
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)

    def normalize(self):
        mag = self.magnitude()
        if mag == 0:
            return Vector3.zeros()
        return Vector3(self.x / mag, self.y / mag, self.z / mag)

    def dot_product(self, other):
        if not isinstance(other, Vector3):
            raise ValueError("o produto precisa ser um vector3.")
        return self.x * other.x + self.y * other.y + self.z * other.z

    def __str__(self):
        return f"Vector3({self.x}, {self.y}, {self.z})"


class Lerp:
    @staticmethod
    def linear_interpolation_vector2(vector1, vector2, t):
        return vector1.linear_interpolation(vector2, t)

    @staticmethod
    def linear_interpolation_vector3(vector1, vector2, t):
        return vector1.linear_interpolation(vector2, t)


import numpy as np

class Quaternion:
    def __init__(self, vector3: Vector3, w, x, y, z):
        self.vector3 = vector3
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    @classmethod
    def identity(cls, vector3: Vector3):
        return cls(vector3, 1, 0, 0, 0)

    @classmethod
    def from_axis_angle(cls, vector3: Vector3, axis, angle):
        norm = np.linalg.norm(axis)
        if norm == 0:
            raise ValueError("Axis must have a non-zero length.")
        
        axis = axis / norm
        half_angle = angle / 2
        w = np.cos(half_angle)
        x, y, z = axis * np.sin(half_angle)
        return cls(vector3, w, x, y, z)

    def conjugate(self):
        return Quaternion(self.vector3, self.w, -self.x, -self.y, -self.z)

    def normalize(self):
        mag = np.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)
        if mag == 0:
            return Quaternion.identity(self.vector3)
        return Quaternion(self.vector3, self.w / mag, self.x / mag, self.y / mag, self.z / mag)

    def __mul__(self, other):
        if not isinstance(other, Quaternion):
            raise ValueError("Multiplicação válida apenas com outro Quaternion.")
        
        w = self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z
        x = self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y
        y = self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x
        z = self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w

        return Quaternion(self.vector3, w, x, y, z)

    def __str__(self):
        return f"Quaternion({self.w}, {self.x}, {self.y}, {self.z})"

#math

class Math:
    Pi = mt.pi

    @staticmethod
    def cos(angle):
        return mt.cos(angle)

    @staticmethod
    def sin(angle):
        return mt.sin(angle)

    @staticmethod
    def floor(integer):
        return int(integer)

    @staticmethod
    def tan(angle):
        return mt.tan(angle)

    @staticmethod
    def cot(angle):
        return 1 / mt.tan(angle)

    @staticmethod
    def sec(angle):
        return 1 / mt.cos(angle)

    @staticmethod
    def quad(integer):
        return integer**2

    @staticmethod
    def sqrt(integer):
        return mt.sqrt(integer)

    @staticmethod
    def ceil(integer):
        return mt.ceil(integer)
