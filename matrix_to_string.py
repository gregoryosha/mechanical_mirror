import numpy as np

img_arr = [[1, 2, 3], [4, 5, 6], [7, 8 ,9]]
in_del, out_del = " ", ";"
flat_matrix = str(out_del.join([in_del.join([str(ele) for ele in sub]) for sub in img_arr]))
print(img_arr)

img_mod = np.matrix(flat_matrix)
print(img_mod)

