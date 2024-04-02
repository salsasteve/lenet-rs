fn average_pooling(matrix: Vec<Vec<f32>>, stride: usize, pool_size: usize) -> Vec<Vec<f32>> {
    let rows = matrix.len();
    let cols = matrix[0].len();
    let mut pooled_matrix = Vec::new();

    // Iterate over the matrix with the given stride and pool size
    for i in (0..rows).step_by(stride) {
        let mut row_vec = Vec::new();
        for j in (0..cols).step_by(stride) {
            let mut sum = 0.0;
            let mut count = 0;

            // Calculate the sum of elements in the current pool
            for x in i..usize::min(i + pool_size, rows) {
                for y in j..usize::min(j + pool_size, cols) {
                    sum += matrix[x][y];
                    count += 1;
                }
            }

            // Calculate the average and add it to the row vector
            if count > 0 {
                row_vec.push(sum / count as f32);
            }
        }
        if !row_vec.is_empty() {
            pooled_matrix.push(row_vec);
        }
    }

    pooled_matrix
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_average_pooling() {
        let matrix = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![5.0, 6.0, 7.0, 8.0],
            vec![9.0, 10.0, 11.0, 12.0],
            vec![13.0, 14.0, 15.0, 16.0],
        ];
        let stride = 2;
        let pool_size = 2;

        let result = average_pooling(matrix, stride, pool_size);
        let expected = vec![
            vec![3.5, 5.5], // Average of [1.0, 2.0, 5.0, 6.0] and [3.0, 4.0, 7.0, 8.0]
            vec![11.5, 13.5], // Average of [9.0, 10.0, 13.0, 14.0] and [11.0, 12.0, 15.0, 16.0]
        ];

        assert_eq!(result, expected);
    }
}
