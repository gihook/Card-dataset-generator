import global_variables.py

bord_size = 2
alphamask = np.ones((card_height, card_width), dtype = np.uint8) * 255
cv2.rectangle(alphamask, (0, 0), (card_width - 1, card_height - 1), 0, bord_size)
cv2.line(alphamask, (bord_size * 3, 0), (0, bord_size * 3), 0, bord_size)
cv2.line(alphamask, (card_width - bord_size * 3, 0), (bord_size * 3, card_height), 0, bord_size)
cv2.line()
