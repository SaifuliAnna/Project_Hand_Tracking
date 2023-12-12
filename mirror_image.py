import os
import cv2


folder_path = "finger_images"
my_list = os.listdir(folder_path)
output_path = "finger_images_right"

for img_path in my_list:
    # Завантажте ваше зображення
    img = cv2.imread(f"{folder_path}/{img_path}")

    # Відобразіть зображення дзеркально
    mirrored_img = cv2.flip(img, 1)

    # Збережіть дзеркальне зображення
    new_output_path = f"{output_path}//{img_path}"
    cv2.imwrite(new_output_path, mirrored_img)
    print(f'Mirrored image saved at: {new_output_path}')



"""
import cv2

# Завантажте ваше зображення
image_path = 'finger_images/1.jpeg'
img = cv2.imread(image_path)

# Відобразіть зображення дзеркально
mirrored_img = cv2.flip(img, 1)

# Відобразіть оригінальне та дзеркальне зображення поруч
cv2.imshow('Original Image', img)
cv2.imshow('Mirrored Image', mirrored_img)

# Збережіть дзеркальне зображення
output_path = 'finger_images_right/1.jpeg'
cv2.imwrite(output_path, mirrored_img)
print(f'Mirrored image saved at: {output_path}')

# # Зачекайте, доки користувач не натисне клавішу для закриття вікна
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
