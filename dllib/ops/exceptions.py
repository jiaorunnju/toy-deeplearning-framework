"""
This file contains useful user-defined exceptions
"""


class InvalidShapeError(Exception):

    def __init__(self, messgae: str):
        self.message = messgae
