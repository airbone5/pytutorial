import builtins
from collections.abc import Callable
from typing import Any, overload, Literal

class Person:
    def __init__(self):
      self.name="me"
    def hello():
       print("hello")
# class Person1:
#     def __init__(self, name, age):
#         self.name = name
#         self.age = age

_x : Person
hello=_x.hello