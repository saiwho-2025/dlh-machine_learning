#!/usr/bin/env python3
matrix = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
the_middle = []
for row in matrix:
    the_middle.append(row[2:4])
print("The middle columns of the matrix are: {}".format(the_middle))
