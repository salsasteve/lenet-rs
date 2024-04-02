import numpy as np
from scipy.signal import convolve2d, correlate2d

test_images = [
    [ 
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1]
    ],
    [  
        [0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0]
    ],
    [  
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1]
    ],
    [  
        [-1, -1, -1, -1, -1],
        [-1, 2, 2, 2, -1],
        [-1, 2, 8, 2, -1],
        [-1, 2, 2, 2, -1],
        [-1, -1, -1, -1, -1]
    ],
    [  
        [0, -1, 0, -1, 0],
        [-1, 0, -1, 0, -1],
        [0, -1, 4, -1, 0],
        [-1, 0, -1, 0, -1],
        [0, -1, 0, -1, 0]
    ],
    [  
        [-2, -1, 0, -1, -2],
        [-1, 0, 1, 0, -1],
        [0, 1, 2, 1, 0],
        [-1, 0, 1, 0, -1],
        [-2, -1, 0, -1, -2]
    ]
]

# Kernel
kernel = np.ones((5, 5), dtype=int)

# Convolution
for i in range(len(test_images)):
    output_matrix_scipy = convolve2d(test_images[i], kernel, mode='same')
    print(output_matrix_scipy+1)
