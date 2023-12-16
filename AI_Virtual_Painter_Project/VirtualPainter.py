import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm


folder_path = 'Header'
my_list = os.listdir(folder_path)
print(my_list)

# Сортуємо список за номерами файлів (враховуючи лише файли з розширенням .png)
sorted_list = sorted([file for file in my_list if file.endswith(".jpg")])
print(sorted_list)

