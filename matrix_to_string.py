def string_to_matrix(input_string):
    """
    Converts a string into a 2D matrix.

    Parameters:
    - input_string (str): The input string containing matrix elements.

    Returns:
    - list: The 2D matrix.
    """
    # Split the string into rows using semicolons as delimiters
    rows = input_string.strip().split(';')

    # Split each row into elements using spaces as delimiters
    matrix = [list(map(int, row.split())) for row in rows]

    return matrix


# Example usage:
input_str = "1 2 3; 4 5 6; 7 8 9; 10 11 12"

result_matrix = string_to_matrix(input_str)

# Print the result
print("Resulting 2D matrix:")
for row in result_matrix:
    print(row)
