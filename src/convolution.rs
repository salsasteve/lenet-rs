
struct ConvParams {
    stride: (usize, usize), // stride for height and width
    padding: (usize, usize), // padding for height and width
}

struct Kernel {
    weights: Vec<Vec<Vec<Vec<f32>>>>, // [out_channels][in_channels][kernel_height][kernel_width]
    bias: Vec<f32>, // [out_channels]
}

fn pretty_print_matrix(matrix: &Vec<Vec<Vec<f32>>>) {
    for i in 0..matrix.len() {
        for j in 0..matrix[0].len() {
            for k in 0..matrix[0][0].len() {
                print!("{} ", matrix[i][j][k]);
            }
            println!();
        }
        println!();
    }
}

// Utility function for applying padding to the input feature map
fn apply_padding(input: &Vec<Vec<Vec<f32>>>, padding: (usize, usize)) -> Vec<Vec<Vec<f32>>> {
    let mut padded_input = vec![vec![vec![0.0; input[0][0].len() + 2 * padding.1]; input[0].len() + 2 * padding.0]; input.len()];

    for i in 0..input.len() {
        for j in 0..input[0].len() {
            for k in 0..input[0][0].len() {
                padded_input[i][j + padding.0][k + padding.1] = input[i][j][k];
            }
        }
    }

    padded_input
}

fn convolve_single_position(
    input: &Vec<Vec<Vec<f32>>>, // input feature maps: [in_channels][height][width]
    kernel: &Kernel, // convolution kernels and biases
    params: &ConvParams, // convolution parameters  
    output_x: usize, // x position in the output feature map
    output_y: usize, // y position in the output feature map
) -> Vec<f32> {
    let in_channels = input.len();
    let kernel_height = kernel.weights[0][0].len();
    let kernel_width = kernel.weights[0][0][0].len();
    let padding = params.padding;
    let stride = params.stride;
    let out_channels = kernel.weights.len();
    let kernel_in_channels = kernel.weights[0].len();
    let input_height = input[0].len();
    let input_width = input[0][0].len();
    // println!("num of input channels: {}, kernel_in_channels: {}", in_channels, kernel_in_channels);

    if in_channels != kernel_in_channels {
        println!("out_channels: {}, in_channels: {}", in_channels, kernel_in_channels);
        panic!("Number of input channels must match the number of kernel in channels");
    }

    if kernel_width % 2 == 0 || kernel_height % 2 == 0 {
        panic!("Only odd kernel sizes are supported");
    }
    if kernel_width > input_width || kernel_height > input_height {
        panic!("Kernel size must be smaller than input size");
    }

    let mut output_values = vec![0.0; out_channels];
    println!("output_x: {}, output_y: {}", output_x, output_y);
    for out_channel in 0..out_channels {
        let mut acc = 0.0; // Accumulator for the sum
        for in_channel in 0..in_channels {
            for ky in 0..kernel_height {
                for kx in 0..kernel_width {
                    let in_y = output_y * stride.0 + ky;
                    let in_x = output_x * stride.1 + kx;
                    
                    let input_val = input[in_channel][in_y][in_x];
                    let kernel_val = kernel.weights[out_channel][in_channel][ky][kx];
                    acc += input_val * kernel_val;
                    println!("input_val: {}, kernel_val: {}, ky: {}, kx: {}, in_y: {}, in_x: {}", input_val, kernel_val, ky, kx, in_y, in_x);
                
                }
            }
            
        }
        acc += kernel.bias[out_channel];
        output_values[out_channel] = acc;
        println!("acc: {}", acc);
    }

    output_values
}


fn convolution_2d(
    input: &Vec<Vec<Vec<f32>>>, // input feature maps: [in_channels][height][width]
    kernel: &Kernel, // convolution kernels and biases
    params: &ConvParams, // convolution parameters
) -> Vec<Vec<Vec<f32>>> {
    let input_height = input[0].len();
    let input_width = input[0][0].len();
    let out_channels = kernel.weights.len();
    let kernel_height = kernel.weights[0][0].len();
    let kernel_width = kernel.weights[0][0][0].len();
    let output_height = (input_height - kernel_height + 2 * params.padding.0 ) / params.stride.0 + 1;
    let output_width = (input_width - kernel_width + 2 * params.padding.1) / params.stride.1 + 1;
            
    let padded_input = apply_padding(&input, params.padding);

    let mut output = vec![vec![vec![0.0; output_width]; output_height]; out_channels];
    println!("output_height: {}, output_width: {}", output_height, output_width);

    pretty_print_matrix(&padded_input);
    pretty_print_matrix(&kernel.weights[0]);
    pretty_print_matrix(&kernel.weights[1]);

    for y in 0..output_height {
        for x in 0..output_width {
            let output_values = convolve_single_position(&padded_input, kernel, params, x, y);
            for out_channel in 0..out_channels {
                output[out_channel][y][x] = output_values[out_channel];
            }
        }
    }

    output
}
    

#[cfg(test)]
mod tests {
    use super::*;

    fn get_test_data0() -> (Vec<Vec<Vec<f32>>>, Kernel) {
        // Define a simple 3x3x1 image (1 channel, 3x3 pixels)
        let images = vec![vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0]
        ]];

        // Define a 3x3 kernel that sums the values it covers
        // [out_channels][in_channels][kernel_height][kernel_width
        let weights = vec![vec![vec![vec![1.0, 1.0, 1.0], vec![1.0, 1.0, 1.0], vec![1.0, 1.0, 1.0]]]]; // [1][1][2][2] kernel
        let bias = vec![0.0]; // No bias for simplicity

        // Create the Kernel struct
        let kernel = Kernel {
            weights: weights,
            bias: bias,
        };

        (images, kernel)
    }

    fn get_test_data1() -> (Vec<Vec<Vec<f32>>>, Kernel) {
        // Define a simple 3x3x1 image (1 channel, 3x3 pixels)
        let images = vec![vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0]
        ]];

        // Define a 2x2 kernel that sums the values it covers
        // [out_channels][in_channels][kernel_height][kernel_width
        let weights = vec![vec![vec![vec![1.0, 1.0, 1.0], vec![1.0, 1.0, 1.0], vec![1.0, 1.0, 1.0]]]]; // [1][1][2][2] kernel
        let bias = vec![0.0]; // No bias for simplicity

        // Create the Kernel struct
        let kernel = Kernel {
            weights: weights,
            bias: bias,
        };

        (images, kernel)
    }

    fn get_test_data2() -> (Vec<Vec<Vec<f32>>>, Kernel) {
        // Define 2 simple 5x5 images
        let images = vec![
            vec![vec![1.0, 0.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0, 0.0], vec![0.0, 0.0, 1.0, 0.0, 0.0], vec![0.0, 0.0, 0.0, 1.0, 0.0], vec![0.0, 0.0, 0.0, 0.0, 1.0]],
            vec![vec![-2.0, -1.0, 0.0, -1.0, -2.0], vec![-1.0, 0.0, 1.0, 0.0, -1.0], vec![0.0, 1.0, 2.0, 1.0, 0.0], vec![-1.0, 0.0, 1.0, 0.0, -1.0], vec![-2.0, -1.0, 0.0, -1.0, -2.0]]
        ];

        // Define 2 channel 3x3 kernel that sums the values it covers
        // [out_channels][in_channels][kernel_height][kernel_width]
        let weights = vec![
            vec![
                vec![vec![1.0; 3]; 3], // First feature map with all ones
                vec![vec![1.0; 3]; 3]
            ], 
            vec![
                vec![vec![2.0; 3]; 3], // Second feature map with all twos
                vec![vec![2.0; 3]; 3]
            ]
        ]; // [2][2][3][3] kernel

        let bias = vec![0.0; 2];

        // Create Kernel struct
        let kernel = Kernel {
            weights: weights, // wrap to match the [out_channels][in_channels][kernel_height][kernel_width] structure
            bias: bias,
        };

        (images, kernel)
    }

    fn get_test_data3() -> (Vec<Vec<Vec<f32>>>, Kernel) {
        let images = vec![
            vec![vec![1.0, 0.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0, 0.0], vec![0.0, 0.0, 1.0, 0.0, 0.0], vec![0.0, 0.0, 0.0, 1.0, 0.0], vec![0.0, 0.0, 0.0, 0.0, 1.0]],
            vec![vec![-2.0, -1.0, 0.0, -1.0, -2.0], vec![-1.0, 0.0, 1.0, 0.0, -1.0], vec![0.0, 1.0, 2.0, 1.0, 0.0], vec![-1.0, 0.0, 1.0, 0.0, -1.0], vec![-2.0, -1.0, 0.0, -1.0, -2.0]]
        ];

        // Define 2 channel 3x3 kernel that sums the values it covers
        // [out_channels][in_channels][kernel_height][kernel_width]
        let weights = vec![
            vec![
                vec![vec![1.0; 3]; 3], // First feature map with all ones
                vec![vec![1.0; 3]; 3]
            ], 
            vec![
                vec![vec![2.0; 3]; 3], // Second feature map with all twos
                vec![vec![2.0; 3]; 3]
            ]
        ]; // [2][2][3][3] kernel

        let bias = vec![0.0, 0.0];

        // Create Kernel struct
        let kernel = Kernel {
            weights: weights, // wrap to match the [out_channels][in_channels][kernel_height][kernel_width] structure
            bias: bias,
        };

        (images, kernel)

    }

    #[test]
    fn test_apply_padding() {
        let input = vec![vec![vec![1.0; 5]; 5]; 1];
        let padding = (2, 2);

        let padded_input = apply_padding(&input, padding);
 
        let expected_output = vec![
            vec![
                vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                vec![0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                vec![0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                vec![0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                vec![0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                vec![0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            ]
        ];


        assert_eq!(padded_input, expected_output);
    }

    #[test]
    fn test_convolve_single_position() {
        let (images, kernel) = get_test_data0();
        let convolution_params = ConvParams {
            stride: (1, 1),
            padding: (0, 0),
        };


        let output = convolve_single_position(&images, &kernel, &convolution_params, 0, 0);
        // for our 3x3 kernel and selected input, it should sum up 
        let expected_output = vec![45.0];

        assert_eq!(output, expected_output, "Convolution at position ({}, {}) did not match expected output.", 0, 0);
    }

    #[test]
    fn test_convolve_single_position_with_padding() {
        let (images, kernel) = get_test_data1();
        let convolution_params = ConvParams {
            stride: (1, 1),
            padding: (1, 1),  // Padding added
        };
        let output_x = 1;  // Considering padding, kernel applies at the corner
        let output_y = 1;

        // Compute convolution for a single position with padding
        let padded_input = apply_padding(&images, convolution_params.padding);

        // pretty print the padded input
        pretty_print_matrix(&padded_input);
        // pretty print the kernel
        pretty_print_matrix(&kernel.weights[0]);

        let output = convolve_single_position(&padded_input, &kernel, &convolution_params, output_x, output_y);

        // Calculate the expected output
        // Considering the padding, the convolution at the top-left corner will cover the padded area
        // 1+2+3+4+5+6+7+8+9 = 45
        let expected_output = vec![45.0]; 

        assert_eq!(output, expected_output, "Convolution with padding at position ({}, {}) did not match expected output.", output_x, output_y);
    }

    #[test]
    fn test_convolution_2d(){
        let (images, kernel) = get_test_data2();
        let convolution_params = ConvParams {
            stride: (1, 1),
            padding: (0, 0),
        };

        let output = convolution_2d(&images, &kernel, &convolution_params);

        let expected_output = vec![
            vec![
                vec![3.0, 5.0, 1.0],
                vec![5.0, 9.0, 5.0],
                vec![1.0, 5.0, 3.0]
            ],
            vec![
                vec![6.0, 10.0, 2.0],
                vec![10.0, 18.0, 10.0],
                vec![2.0, 10.0, 6.0]
            ]
        ];

        for (i, output) in output.iter().enumerate() {
            println!("Output: {:?}", output);   
            assert_eq!(output, &expected_output[i]);
        }

    }

    #[test]
    fn test_convolution_2d_with_padding(){

        let (images, kernel) = get_test_data3();
        let convolution_params = ConvParams {
            stride: (1, 1),
            padding: (1, 1),
        };

        let output = convolution_2d(&images, &kernel, &convolution_params);

        let expected_output = vec![
            vec![
                vec![-2.0, -1.0, 0.0, -3.0, -4.0],
                vec![-1.0, 3.0, 5.0, 1.0, -3.0],
                vec![0.0, 5.0, 9.0, 5.0, 0.0],
                vec![-3.0, 1.0, 5.0, 3.0, -1.0],
                vec![-4.0, -3.0, 0.0, -1.0, -2.0]
            ],
            vec![
                vec![-4.0, -2.0, 0.0, -6.0, -8.0],
                vec![-2.0, 6.0, 10.0, 2.0, -6.0],
                vec![0.0, 10.0, 18.0, 10.0, 0.0],
                vec![-6.0, 2.0, 10.0, 6.0, -2.0],
                vec![-8.0, -6.0, 0.0, -2.0, -4.0
                
                
                ]
            ]
        ];

        for (i, output) in output.iter().enumerate() {
            println!("Output: {:?}", output);   
            assert_eq!(output, &expected_output[i]);
        }

    }
}
