import tensorflow as tf 
import numpy as np 
import os


class student(object):
    """docstring for student"""
    def __init__(self,name,salary=2000,status=3000):
       
       
        self.salary = salary
        self.status = status

    def method(self):
        return self.name, self.salary,self.status

    def calculation(self):

        self.mysalary = 3000
        return self.mysalary


print(student('').calculation())


