#!/usr/bin/env python3.4
# -*- coding: utf-8 -*-
import tkinter, tkinter.constants, tkinter.filedialog, numpy, tkinter.messagebox, sys, matplotlib, tkinter.simpledialog, math, tkinter.font
import matplotlib.mlab as ml
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


##########################################################################
#    Copyright (C) 2015 Iñigo Marcos Alcalde                             #
##########################################################################
#                               LICENSE                                  #
##########################################################################
#    MEPSA is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program. If not, see <http://www.gnu.org/licenses/>.
##########################################################################
##########################################################################
#Contact: pagomez@cbm.csic.es and imarcos@cbm.csic.es
##########################################################################
#Cite: Iñigo Marcos-Alcalde, Javier Setoain, Jesús I. Mendieta-Moreno, Jesús Mendieta, and Paulino Gómez-Puertas. (2015) MEPSA: minimum energy pathway analysis for energy landscapes. Bioinformatics 31: 3853-3855
##########################################################################

version = "1.2 Python 3.x"

def print_progress(current, total):
    step = total / 10
    step_counter = 1
    threshold = 0
    while (step_counter <= 10):
        threshold = step * step_counter
        if (current <= threshold and current + 1 > threshold):
            print(str (step_counter * 10) + "%")
        step_counter += 1

def get_index_array (input_array):
    for counter in range (numpy.shape(input_array)[0]):
        input_array[counter] = counter
    return input_array


def select_around_mode (is_4_axes_bool):
    around_x_mod = ([0,1,0,-1,1,1,-1,-1])
    around_y_mod = ([1,0,-1,0,1,-1,-1,1])
    if is_4_axes_bool == True:
        around_size = 8
    else:
        around_size = 4
    return around_x_mod, around_y_mod, around_size

def create_neighbors_array(index_matrix, x_array, y_array, z_array, is_4_axes_bool):
    around_x_mod, around_y_mod, around_size = select_around_mode (is_4_axes_bool)
    neighbors_array = numpy.empty((around_size+1,numpy.shape(z_array)[0]), dtype=int); neighbors_array.fill(-1)
    neighbor_pos = 1   
    for counter in range (numpy.shape(z_array)[0]):
        neighbor_pos = 1
        neighbors_array[0][counter] = 0
        for around_counter in range (around_size):
            new_x = x_array[counter] + around_x_mod[around_counter]
            if new_x >= 0 and new_x < numpy.shape(index_matrix)[0]:
                new_y = y_array[counter] + around_y_mod[around_counter]
                if new_y >= 0 and new_y < numpy.shape(index_matrix)[1]:
                    neighbors_array[int(neighbor_pos), int(counter)] = index_matrix[int(new_x),int(new_y)]
                    neighbor_pos += 1
                    neighbors_array[0, counter] += 1
    return neighbors_array

########################################
########################################

def propagate_point_connectivity(working_array, point_index, z_array, neighbors_array, node_array, mask_array, first_working_point, last_pos_in_working_array, node_matrix, occupied_points_array, connectivity_propagation_array, connected_minima, total_minima, well_sampling):
    stop = False
    for neighbor_counter in range (1,(neighbors_array[0][point_index]+1).astype(int)):
        if (mask_array[neighbors_array[neighbor_counter][point_index]] > 0):
            if (occupied_points_array[neighbors_array[neighbor_counter][point_index]] == 0):
                first_working_point, last_pos_in_working_array = insert_point_in_working_array(working_array, neighbors_array[neighbor_counter][point_index], z_array, first_working_point, last_pos_in_working_array, occupied_points_array)
                connectivity_propagation_array[neighbors_array[neighbor_counter][point_index]] = connectivity_propagation_array[point_index]
            if (well_sampling == True and connectivity_propagation_array[neighbors_array[neighbor_counter][point_index]] != connectivity_propagation_array[point_index]):
                connected_minima[connectivity_propagation_array[neighbors_array[neighbor_counter][point_index]]] = 1
                connected_minima[connectivity_propagation_array[point_index]] = 1
                if (numpy.sum(connected_minima) == total_minima):
                    stop = True
                if (node_matrix[connectivity_propagation_array[neighbors_array[neighbor_counter][point_index]]][connectivity_propagation_array[point_index]] == 0):
                    node_matrix[connectivity_propagation_array[neighbors_array[neighbor_counter][point_index]]][connectivity_propagation_array[point_index]] = 1
                    node_matrix[connectivity_propagation_array[point_index]][connectivity_propagation_array[neighbors_array[neighbor_counter][point_index]]] = 1
    return working_array, first_working_point, last_pos_in_working_array, stop

def define_connectivity(node_array, mask_array, z_array, neighbors_array, in_node_matrix, points_in_minima_groups, well_sampling = False):
    working_array = numpy.empty((numpy.shape(z_array)[0]+1,3), dtype = int); working_array.fill(-1)#[0] index [1] previous [2] next
    node_matrix = numpy.empty(numpy.shape(in_node_matrix)); node_matrix.fill(0)
    first_working_point = -1
    last_pos_in_working_array = -1
    iteration_counter = 0
    occupied_points_array = numpy.zeros(numpy.shape(z_array)[0]); #0 empty, 1 occupied, 2 in working array
    connectivity_propagation_array = numpy.empty(numpy.shape(z_array)[0], dtype = int); connectivity_propagation_array.fill(-1)#-1 empty, n node_number
    points_to_propagate_array = numpy.empty(numpy.shape(z_array)[0], dtype = int); points_to_propagate_array.fill(-1)
    points_to_propagate_current_pos = 0
    total_points_place_holder = 0
    total_minima = numpy.shape(points_in_minima_groups)[0]
    connected_minima = numpy.zeros(total_minima)
    for counter_x in range (numpy.shape(points_in_minima_groups)[0]):
        for counter_y in range (numpy.shape(points_in_minima_groups)[1]):
            if (points_in_minima_groups[counter_x][counter_y] != -1):
                first_working_point, last_pos_in_working_array = insert_point_in_working_array (working_array, points_in_minima_groups[counter_x][counter_y], z_array, first_working_point, last_pos_in_working_array, occupied_points_array)
                connectivity_propagation_array[points_in_minima_groups[counter_x][counter_y]] = counter_x
    stop = False
    while (last_pos_in_working_array != -2 and (stop == False or well_sampling == False)):
        iteration_counter += 1
        #print "Cycle " + str(iteration_counter)
        points_to_propagate_array[0] = first_working_point
        points_to_propagate_current_pos = 0
        while (last_pos_in_working_array != -2):
            if (working_array[points_to_propagate_array[points_to_propagate_current_pos]][2] != -1 and z_array[working_array[first_working_point][0]] == z_array[working_array[working_array[points_to_propagate_array[points_to_propagate_current_pos]][2]][0]]):
                points_to_propagate_array[points_to_propagate_current_pos+1] = working_array[points_to_propagate_array[points_to_propagate_current_pos]][2]
                points_to_propagate_current_pos += 1
            else:
                points_to_propagate_current_pos += 1
                break
        for propagate_counter in range (points_to_propagate_current_pos):
            if(stop == False):
                working_array, first_working_point, last_pos_in_working_array, stop = propagate_point_connectivity(working_array, working_array[points_to_propagate_array[propagate_counter]][0], z_array, neighbors_array, node_array, mask_array, first_working_point, last_pos_in_working_array, node_matrix, occupied_points_array, connectivity_propagation_array, connected_minima, total_minima, well_sampling)
                total_points_place_holder, first_working_point, last_pos_in_working_array = remove_point_from_working_array(working_array, points_to_propagate_array[propagate_counter], first_working_point, last_pos_in_working_array, occupied_points_array, total_points_place_holder, iteration_counter)
        print_progress(total_points_place_holder, numpy.shape(z_array)[0])
    return connectivity_propagation_array, node_matrix
    
def well_sampling_analysis (origin, target, input_matrix, node_array, node_matrix_template, is_4_axes_bool, x_values, y_values):
    points_in_minima_groups = numpy.array([origin,target]).reshape(2,1)
    print("Analysing well sampling")
    input_matrix_shape = numpy.shape(input_matrix)
    x_array, y_array, z_array = generate_three_column_format(get_index_array(numpy.empty(input_matrix_shape[0])), get_index_array(numpy.empty(input_matrix_shape[1])), input_matrix)
    index_matrix = create_index_matrix(input_matrix, x_array, y_array)
    neighbors_array = create_neighbors_array(index_matrix, x_array, y_array, z_array, is_4_axes_bool)
    mask_array = numpy.empty(numpy.shape(x_array)[0]); mask_array.fill(1)
    print("Propagating connectivity")
    connectivity_propagation_array, connectivity_node_matrix = define_connectivity (node_array, mask_array, z_array, neighbors_array, node_matrix_template, points_in_minima_groups, well_sampling = True)
    print("Obtaining barriers")
    minimum_barrier_matrix, total_points_array, total_barriers = connectivity_propagation_array_analysis(connectivity_propagation_array, points_in_minima_groups, x_values, y_values, x_array, y_array, z_array, neighbors_array)
    counter = 0
    out_pos = 0
    temp_points = numpy.empty((total_points_array[counter],3))
    for map_counter in range(numpy.shape(connectivity_propagation_array)[0]):
        if connectivity_propagation_array[map_counter] == counter:
            temp_points[out_pos][0] = x_values[int(x_array[map_counter])]
            temp_points[out_pos][1] = y_values[int(y_array[map_counter])]
#            temp_points[out_pos][2] = connectivity_propagation_array[map_counter]
            temp_points[out_pos][2] = input_matrix[int(x_array[map_counter])][int(y_array[map_counter])]
            out_pos += 1
    points_list = temp_points.tolist()
    points_stack = [points_list,]
    for counter in range(1,numpy.shape(points_in_minima_groups)[0]):
        temp_points = numpy.empty((total_points_array[counter],3))
        out_pos = 0
        for map_counter in range(numpy.shape(connectivity_propagation_array)[0]):
            if connectivity_propagation_array[map_counter] == counter:
                temp_points[out_pos][0] = x_values[int(x_array[map_counter])]
                temp_points[out_pos][1] = y_values[int(y_array[map_counter])]
#                temp_points[out_pos][2] = connectivity_propagation_array[map_counter]
                temp_points[out_pos][2] = input_matrix[int(x_array[map_counter])][int(y_array[map_counter])]
                out_pos += 1
        points_list = temp_points.tolist()
        points_stack.append(points_list)
    return numpy.asarray(points_stack[0]), numpy.asarray(points_stack[1])
    

########################################
########################################
    
def node_list_generation (origin_point, target_point, mask_array, neighbors_array, connectivity_propagation_array):
    mask_position = target_point
    total_out_points = 0
    temp_pos = 0
    temp_array = numpy.empty(numpy.shape(mask_array)[0], dtype = int)
    temp_array[total_out_points] = target_point
    total_out_points +=1
    while mask_position != origin_point:
        temp_score = numpy.shape(mask_array)[0]
        for neighbor_counter in range (1,(neighbors_array[0][mask_position]+1).astype(int)):
            if mask_array[neighbors_array[neighbor_counter][mask_position]] > 0 and mask_array[neighbors_array[neighbor_counter][mask_position]] < temp_score:
                temp_pos = neighbors_array[neighbor_counter][mask_position]
                temp_score = mask_array[neighbors_array[neighbor_counter][mask_position]]
        mask_position = temp_pos
        temp_array[total_out_points] = mask_position
        total_out_points +=1
    out_nodes = ((connectivity_propagation_array[temp_array[total_out_points-1]],))
    last_pos = 0
    for counter in range(total_out_points-2,-1,-1):
        if connectivity_propagation_array[temp_array[counter]] != out_nodes[last_pos]:
            last_pos += 1
            out_nodes = numpy.append(out_nodes,connectivity_propagation_array[temp_array[counter]])
    return out_nodes
    
def node_finder (input_matrix, is_4_axes_bool, x_values, y_values, flat_nodes_bool):
    print("Searching for nodes")
    input_matrix_shape = numpy.shape(input_matrix)
    x_array, y_array, z_array = generate_three_column_format(get_index_array(numpy.empty(input_matrix_shape[0])), get_index_array(numpy.empty(input_matrix_shape[1])), input_matrix)
    index_matrix = create_index_matrix(input_matrix, x_array, y_array)
    neighbors_array = create_neighbors_array(index_matrix, x_array, y_array, z_array, is_4_axes_bool)
    max_check = False
    min_check = False
    node_array = numpy.empty(numpy.shape(z_array)[0]); node_array.fill(-2)
    minima_counter = -1
    for global_counter in range (numpy.shape(z_array)[0]):
        if (max_check == True):
            max_check = False
        if (min_check == True):
            min_check = False
        for neighbor_counter in range (1,(neighbors_array[0][global_counter]+1).astype(int)):
            if (z_array[global_counter] > z_array[int(neighbors_array[neighbor_counter][global_counter])]):
                max_check = True
                break
            elif (z_array[global_counter] < z_array[neighbors_array[neighbor_counter][global_counter]] and flat_nodes_bool == False and min_check == False):
                min_check = True
                
        if (max_check == False and (flat_nodes_bool == True or min_check == True)):
        #if (max_check == 0):
            minima_counter += 1
            node_array[global_counter] = minima_counter
        print_progress(global_counter, numpy.shape(z_array)[0])
    minima_groups_counter = -1
    if (minima_counter<1):
        return 1, node_array, node_array, node_array, node_array ##This means there is 1 or 0 minima groups and, therefore, analysis cannot be made
    minima_groups_counter = minima_counter
    print(str(minima_groups_counter + 1) + " nodes found")
    if minima_groups_counter < 1:
        return 1, minima_groups_counter, minima_groups_counter, minima_groups_counter, minima_groups_counter
    max_points_in_a_group = 1
    points_in_minima_groups = numpy.empty((minima_groups_counter+1,max_points_in_a_group), dtype = int); points_in_minima_groups.fill(-1)
    points_to_plot = numpy.empty((minima_groups_counter+1,3))
    points_to_plot_counter = 0
    for global_counter in range (numpy.shape(z_array)[0]):
        if (node_array[global_counter] > -1):
            points_to_plot[points_to_plot_counter][0] = x_values[int(x_array[global_counter])]
            points_to_plot[points_to_plot_counter][1] = y_values[int(y_array[global_counter])]
            points_to_plot[points_to_plot_counter][2] = node_array[global_counter]
            points_to_plot_counter += 1
            points_in_minima_groups[int(node_array[global_counter])][0] = global_counter
    node_matrix = numpy.zeros((minima_groups_counter+1,minima_groups_counter+1))
    return 0, points_to_plot, node_array, points_in_minima_groups, node_matrix
        
def insert_point_in_working_array(working_array, point_index, z_array, first_working_point, last_pos_in_working_array, occupied_points_array):
    working_array[last_pos_in_working_array+1][0] = point_index
    first_working_point = int(first_working_point)
    point_index = int(point_index)
    if (first_working_point == -1):
        first_working_point = 0
        last_pos_in_working_array = 0
    else:
        if (z_array[int(working_array[first_working_point][0])] >=  z_array[point_index]):
            working_array[first_working_point][1] = last_pos_in_working_array+1
            working_array[last_pos_in_working_array+1][2] = first_working_point
            first_working_point = last_pos_in_working_array+1
        else:
            current_pos = first_working_point
            while (current_pos != -1):
                if (z_array[int(working_array[current_pos][0])] >=  z_array[point_index]):
                    working_array[last_pos_in_working_array+1][1] = working_array[current_pos][1]
                    working_array[last_pos_in_working_array+1][2] = current_pos
                    working_array[int(working_array[current_pos][1])][2] = last_pos_in_working_array+1   
                    working_array[current_pos][1] = last_pos_in_working_array+1
                    break
                if (working_array[current_pos][2] == -1):
                    working_array[current_pos][2] = last_pos_in_working_array+1
                    working_array[last_pos_in_working_array+1][1] = current_pos
                    working_array[last_pos_in_working_array+1][2] = -1
                    break
                current_pos = int(working_array[current_pos][2])
    last_pos_in_working_array += 1
    occupied_points_array[int(point_index)] = -2
    return int(first_working_point), int(last_pos_in_working_array)

def remove_point_from_working_array(working_array, working_array_point_index, first_working_point, last_pos_in_working_array, occupied_points_array, total_points, iteration_counter):
    if (working_array[working_array_point_index][1] == -1):
        if (working_array[working_array_point_index][2] != -1):
            working_array[working_array[working_array_point_index][2]][1] = -1
            first_working_point = working_array[working_array_point_index][2]
        else:
            last_pos_in_working_array = -2 #there are no more points available
    else:
        if (working_array[working_array_point_index][2] != -1):
            working_array[working_array[working_array_point_index][2]][1] = working_array[working_array_point_index][1]
            working_array[working_array[working_array_point_index][1]][2] = working_array[working_array_point_index][2]
        else:
            working_array[working_array[working_array_point_index][1]][2] = -1
    total_points += 1
    occupied_points_array[working_array[working_array_point_index][0]] = iteration_counter
    return total_points, first_working_point, last_pos_in_working_array
    
def propagate_point(working_array, point_index, z_array, neighbors_array, mask_array, first_working_point, last_pos_in_working_array, occupied_points_array, total_points):
    for neighbor_counter in range (1,(neighbors_array[0][point_index]+1)):
        if (mask_array[neighbors_array[neighbor_counter][point_index]] > 0 and occupied_points_array[neighbors_array[neighbor_counter][point_index]] == 0):
            first_working_point, last_pos_in_working_array = insert_point_in_working_array(working_array, neighbors_array[neighbor_counter][point_index], z_array, first_working_point, last_pos_in_working_array, occupied_points_array)
    return total_points, working_array, first_working_point, last_pos_in_working_array

def mask_gen (origin_point, target_point, mask_array, z_array, neighbors_array):
    working_array = numpy.empty((numpy.shape(z_array)[0]+1,3), dtype = int); working_array.fill(-1)#[0] index [1] previous [2] next
    first_working_point = -1
    last_pos_in_working_array = -1
    iteration_counter = 1
    occupied_points_array = numpy.zeros(numpy.shape(z_array)[0]);
    points_to_propagate_array = numpy.empty(numpy.shape(z_array)[0], dtype=int); points_to_propagate_array.fill(-1)
    points_to_propagate_current_pos = 0
    total_points = 0
    first_working_point, last_pos_in_working_array = insert_point_in_working_array (working_array, origin_point, z_array, first_working_point, last_pos_in_working_array, occupied_points_array)
    while (occupied_points_array[target_point] == 0  and last_pos_in_working_array != -2):
        iteration_counter += 1
        #print "Cycle " + str(iteration_counter)
        points_to_propagate_array[0] = first_working_point
        points_to_propagate_current_pos = 0
        while (last_pos_in_working_array != -2):
            if (working_array[points_to_propagate_array[points_to_propagate_current_pos]][2] != -1 and z_array[working_array[first_working_point][0]] == z_array[working_array[working_array[points_to_propagate_array[points_to_propagate_current_pos]][2]][0]]):
                points_to_propagate_array[points_to_propagate_current_pos+1] = working_array[points_to_propagate_array[points_to_propagate_current_pos]][2]
                points_to_propagate_current_pos += 1
            else:
                points_to_propagate_current_pos += 1
                break
        for propagate_counter in range (points_to_propagate_current_pos):
            total_points, working_array, first_working_point, last_pos_in_working_array = propagate_point(working_array, int(working_array[points_to_propagate_array[propagate_counter]][0]), z_array, neighbors_array, mask_array, first_working_point, last_pos_in_working_array, occupied_points_array, total_points)
            total_points, first_working_point, last_pos_in_working_array = remove_point_from_working_array(working_array, points_to_propagate_array[propagate_counter], first_working_point, last_pos_in_working_array, occupied_points_array, total_points, iteration_counter)
    if (last_pos_in_working_array == -2):
        return 1, occupied_points_array, total_points
    else:
        return 0, occupied_points_array, total_points
        
def get_points (origin_point, target_point, mask_array, neighbors_array, x_values, y_values, x_array, y_array, z_array):
    mask_position = target_point
    total_out_points = 0
    temp_pos = 0
    temp_array = numpy.empty(numpy.shape(mask_array)[0], dtype = int)
    temp_array[total_out_points] = target_point
    total_out_points +=1
    while mask_position != origin_point:
        temp_score = numpy.shape(z_array)[0]
        for neighbor_counter in range (1,(neighbors_array[0][mask_position]+1).astype(int)):
            if mask_array[neighbors_array[neighbor_counter][mask_position]] > 0 and mask_array[neighbors_array[neighbor_counter][mask_position]] < temp_score:
                temp_pos = neighbors_array[neighbor_counter][mask_position]
                temp_score = mask_array[neighbors_array[neighbor_counter][mask_position]]
        mask_position = temp_pos
        temp_array[total_out_points] = mask_position
        total_out_points +=1
    out_pos = total_out_points -1
    out_points = numpy.empty((total_out_points,3))
    for counter in range(total_out_points):
        out_points[out_pos][0] = x_values[int(x_array[temp_array[counter]])]
        out_points[out_pos][1] = y_values[int(y_array[temp_array[counter]])]
        out_points[out_pos][2] = z_array[temp_array[counter]]
        out_pos -=1
    return out_points
    
def connectivity_propagation_array_analysis(connectivity_propagation_array, points_in_minima_groups, x_values, y_values, x_array, y_array, z_array, neighbors_array):
    total_points_array = numpy.zeros(numpy.shape(points_in_minima_groups)[0], dtype = int)
    minimum_barrier_matrix = numpy.empty((numpy.shape(points_in_minima_groups)[0],numpy.shape(points_in_minima_groups)[0]), dtype = int); minimum_barrier_matrix.fill(-1)
    for counter in range(numpy.shape(connectivity_propagation_array)[0]):
        if (connectivity_propagation_array[counter] != -1):
            total_points_array[connectivity_propagation_array[counter]] += 1
        for neighbor_counter in range (1,(neighbors_array[0][counter]+1).astype(int)):
            if connectivity_propagation_array[counter] != connectivity_propagation_array[neighbors_array[neighbor_counter][counter]] and (connectivity_propagation_array[counter] != -1 and connectivity_propagation_array[neighbors_array[neighbor_counter][counter]] != -1):
                x = connectivity_propagation_array[counter]
                y = connectivity_propagation_array[neighbors_array[neighbor_counter][counter]]
                if minimum_barrier_matrix[x][y] == -1 or (minimum_barrier_matrix[x][y] > -1 and z_array[minimum_barrier_matrix[x][y]] > z_array[counter]):
                    minimum_barrier_matrix[x][y] = counter
                    minimum_barrier_matrix[y][x] = counter
    total_barriers = 0    
    for x in range(numpy.shape(points_in_minima_groups)[0]):
        for y in range(x,numpy.shape(points_in_minima_groups)[0]):
            if minimum_barrier_matrix[x][y] != -1:
                temp_pos = minimum_barrier_matrix[x][y]
                minimum_barrier_matrix[x][y] = -1
                for neighbor_counter in range (1,(neighbors_array[0][temp_pos]+1).astype(int)):
                    if connectivity_propagation_array[temp_pos] != connectivity_propagation_array[neighbors_array[neighbor_counter][temp_pos]]:
                        if minimum_barrier_matrix[x][y] == -1 or (minimum_barrier_matrix[x][y] > -1 and z_array[minimum_barrier_matrix[x][y]] > z_array[neighbors_array[neighbor_counter][temp_pos]]) and (connectivity_propagation_array[neighbors_array[neighbor_counter][temp_pos]] == x or connectivity_propagation_array[neighbors_array[neighbor_counter][temp_pos]] == y):
                            minimum_barrier_matrix[x][y] = neighbors_array[neighbor_counter][temp_pos]
                total_barriers += 1
                
    return minimum_barrier_matrix, total_points_array, total_barriers

def global_connectivity_analyis (input_matrix, node_array, points_in_minima_groups, node_matrix_template, is_4_axes_bool, x_values, y_values,well_sampling):
    print("Analysing global connectivity")
    input_matrix_shape = numpy.shape(input_matrix)
    x_array, y_array, z_array = generate_three_column_format(get_index_array(numpy.empty(input_matrix_shape[0])), get_index_array(numpy.empty(input_matrix_shape[1])), input_matrix)
    index_matrix = create_index_matrix(input_matrix, x_array, y_array)
    neighbors_array = create_neighbors_array(index_matrix, x_array, y_array, z_array, is_4_axes_bool)
    mask_array = numpy.empty(numpy.shape(x_array)[0]); mask_array.fill(1)
    print("Propagating connectivity")
    connectivity_propagation_array, connectivity_node_matrix = define_connectivity (node_array, mask_array, z_array, neighbors_array, node_matrix_template, points_in_minima_groups, well_sampling)
    print("Obtaining barriers")
    minimum_barrier_matrix, total_points_array, total_barriers = connectivity_propagation_array_analysis(connectivity_propagation_array, points_in_minima_groups, x_values, y_values, x_array, y_array, z_array, neighbors_array)
    counter = 0
    out_pos = 0
    temp_points = numpy.empty((total_points_array[counter],3))
    for map_counter in range(numpy.shape(connectivity_propagation_array)[0]):
        if connectivity_propagation_array[map_counter] == counter:
            temp_points[out_pos][0] = x_values[int(x_array[map_counter])]
            temp_points[out_pos][1] = y_values[int(y_array[map_counter])]
            temp_points[out_pos][2] = connectivity_propagation_array[map_counter]
            out_pos += 1
    points_list = temp_points.tolist()
    points_stack = [points_list,]
    for counter in range(1,numpy.shape(points_in_minima_groups)[0]):
        temp_points = numpy.empty((total_points_array[counter],3))
        out_pos = 0
        for map_counter in range(numpy.shape(connectivity_propagation_array)[0]):
            if connectivity_propagation_array[map_counter] == counter:
                temp_points[out_pos][0] = x_values[int(x_array[map_counter])]
                temp_points[out_pos][1] = y_values[int(y_array[map_counter])]
                temp_points[out_pos][2] = connectivity_propagation_array[map_counter]
                out_pos += 1
        points_list = temp_points.tolist()
        points_stack.append(points_list)
    barrier_points = numpy.empty((total_barriers,3))
    minima_points = numpy.empty((numpy.shape(points_in_minima_groups)[0],3))
    out_pos = 0
    for x in range(numpy.shape(points_in_minima_groups)[0]):
        minima_points[x][0] = x_values[int(x_array[points_in_minima_groups[x][0]])]
        minima_points[x][1] = y_values[int(y_array[points_in_minima_groups[x][0]])]
        minima_points[x][2] = z_array[points_in_minima_groups[x][0]]
        for y in range(x,numpy.shape(points_in_minima_groups)[0]):
            if minimum_barrier_matrix[x][y] != -1:
                barrier_points[out_pos][0] = x_values[int(x_array[int(minimum_barrier_matrix[x][y])])]
                barrier_points[out_pos][1] = y_values[int(y_array[int(minimum_barrier_matrix[x][y])])]
                barrier_points[out_pos][2] = z_array[minimum_barrier_matrix[x][y]]
                out_pos += 1
    return points_stack, barrier_points, minima_points, connectivity_propagation_array

def find_path (input_matrix, origin_point, target_point, node_array, points_in_minima_groups, node_matrix_template, is_4_axes_bool, x_values, y_values, if_node_by_node):
    print("Preparing")
    input_matrix_shape = numpy.shape(input_matrix)
    x_array, y_array, z_array = generate_three_column_format(get_index_array(numpy.empty(input_matrix_shape[0])), get_index_array(numpy.empty(input_matrix_shape[1])), input_matrix)
    index_matrix = create_index_matrix(input_matrix, x_array, y_array)
    neighbors_array = create_neighbors_array(index_matrix, x_array, y_array, z_array, is_4_axes_bool)
    OT_mask_array = numpy.empty(numpy.shape(x_array)[0]); OT_mask_array.fill(1)
    kkk_mask_array = numpy.empty(numpy.shape(x_array)[0]); kkk_mask_array.fill(1)
    print("Global sampling run")
    mask_gen_bool, OT_mask_array, OT_total_points = mask_gen (origin_point, target_point, OT_mask_array, z_array, neighbors_array)
    print("Done")
    if if_node_by_node == 0:
        out_points = get_points (origin_point, target_point, OT_mask_array, neighbors_array, x_values, y_values, x_array, y_array, z_array)
    else:
        print("Definining node connectivity")
        connectivity_propagation_array, connectivity_node_matrix = define_connectivity (node_array, OT_mask_array, z_array, neighbors_array, node_matrix_template, points_in_minima_groups)
        final_node_list = node_list_generation(origin_point, target_point, OT_mask_array,  neighbors_array, connectivity_propagation_array)
        print("Node by node runs")
        counter = 1
        print(str(counter) + " of " + str(numpy.shape(final_node_list)[0]-1))
        mask_gen_bool, temp_mask_array, total_points = mask_gen(points_in_minima_groups[final_node_list[counter-1]][0], points_in_minima_groups[final_node_list[counter]][0], OT_mask_array, z_array, neighbors_array)
        out_points = get_points (points_in_minima_groups[final_node_list[counter-1]][0], points_in_minima_groups[final_node_list[counter]][0], temp_mask_array, neighbors_array, x_values, y_values, x_array, y_array, z_array)
        for counter in range (2, numpy.shape(final_node_list)[0]):
            print(str(counter) + " of " + str(numpy.shape(final_node_list)[0]-1))
            mask_gen_bool, temp_mask_array, total_points = mask_gen (points_in_minima_groups[final_node_list[counter-1]][0], points_in_minima_groups[final_node_list[counter]][0], OT_mask_array, z_array, neighbors_array)
            temp_out_points = get_points (points_in_minima_groups[final_node_list[counter-1]][0], points_in_minima_groups[final_node_list[counter]][0], temp_mask_array, neighbors_array, x_values, y_values, x_array, y_array, z_array)
            temp_out_points = numpy.delete(temp_out_points, 0, axis = 0)
            out_points = numpy.append(out_points,temp_out_points, axis=0)
    visited_points = numpy.empty((OT_total_points,3))
    visited_points_position = 0
    for counter in range (numpy.shape(OT_mask_array)[0]):
        if (OT_mask_array[counter] > 0):
            visited_points[visited_points_position][0] = x_values[int(x_array[counter])]
            visited_points[visited_points_position][1] = y_values[int(y_array[counter])]
            visited_points[visited_points_position][2] = z_array[counter]
            visited_points_position += 1
    return 0, out_points, visited_points

########################################
########################################

def smooth_path(path_points, iterations):
    print("Smoothing")
    path_smooth_temp = numpy.empty((2,numpy.shape(path_points)[0],2))
    for counter in range(numpy.shape(path_points)[0]):
        path_smooth_temp[0][counter][0] = path_points[counter][0]
        path_smooth_temp[1][counter][0] = path_points[counter][0]
        path_smooth_temp[0][counter][1] = path_points[counter][1]
        path_smooth_temp[1][counter][1] = path_points[counter][1]
    writing_layer = 0
    reference_layer = 1
    for iteration_counter in range(iterations):
        writing_layer, reference_layer = reference_layer, writing_layer
        for counter in range(1,numpy.shape(path_points)[0]-1):
            path_smooth_temp[writing_layer][counter][0] = (path_smooth_temp[reference_layer][counter-1][0] + path_smooth_temp[reference_layer][counter][0] + path_smooth_temp[reference_layer][counter+1][0])/3
            path_smooth_temp[writing_layer][counter][1] = (path_smooth_temp[reference_layer][counter-1][1] + path_smooth_temp[reference_layer][counter][1] + path_smooth_temp[reference_layer][counter+1][1])/3
        print_progress (iteration_counter,iterations)
    return 0, path_smooth_temp[writing_layer]
    

def smooth_map(input_matrix, is_4_axes_bool, iterations):
    print("Smoothing")
    last_pos = numpy.shape(input_matrix)[0]-1
    input_matrix_shape = numpy.shape(input_matrix[last_pos])
    x_array, y_array, z_array = generate_three_column_format(get_index_array(numpy.empty(input_matrix_shape[0])), get_index_array(numpy.empty(input_matrix_shape[1])), input_matrix[last_pos])
    index_matrix = create_index_matrix(input_matrix[last_pos], x_array, y_array)
    neighbors_array = create_neighbors_array(index_matrix, x_array, y_array, z_array, is_4_axes_bool)
    out_matrix = numpy.empty((1,numpy.shape(input_matrix)[1],numpy.shape(input_matrix)[2]))
    map_smooth_temp = numpy.empty((2,numpy.shape(x_array)[0]))
    for counter in range(numpy.shape(x_array)[0]):
        map_smooth_temp[0][counter] = z_array[counter]
        map_smooth_temp[1][counter] = z_array[counter]
    writing_layer = 0
    reference_layer = 1
    for iteration_counter in range(iterations):
        writing_layer, reference_layer = reference_layer, writing_layer
        for counter in range(numpy.shape(x_array)[0]):
            map_smooth_temp[writing_layer][counter] = map_smooth_temp[reference_layer][counter]
            for neighbor_counter in range (1,(neighbors_array[0][counter]+1).astype(int)):
                map_smooth_temp[int(writing_layer)][int(counter)] += map_smooth_temp[int(reference_layer)][int(neighbors_array[neighbor_counter][counter])]
            map_smooth_temp[writing_layer][counter] = map_smooth_temp[writing_layer][counter] / (neighbors_array[0][counter]+1).astype(int)
        print_progress (iteration_counter,iterations)
    for counter in range(numpy.shape(x_array)[0]):
        out_matrix[0][int(x_array[counter])][int(y_array[counter])] = map_smooth_temp[int(writing_layer)][counter]
    return numpy.append(input_matrix, out_matrix, axis = 0)

########################################
########################################

def create_index_matrix(reference_matrix, x_array, y_array):
    index_matrix = numpy.empty(numpy.shape(reference_matrix))
    for counter in range (numpy.shape(x_array)[0]):
        index_matrix[int(x_array[counter])][int(y_array[counter])] = counter
    return index_matrix

def find_limits(sorted_array,center,increment):
    abs_increment = abs(increment);
    up_limit = center+abs_increment;
    down_limit = center-abs_increment;
    sorted_array_last_pos = numpy.shape(sorted_array)[0]-1
    if sorted_array[0] > down_limit:
        down_limit = sorted_array[0]
    if sorted_array[sorted_array_last_pos] < up_limit:
        up_limit = sorted_array[sorted_array_last_pos]
    return up_limit, down_limit

def node_search(x_array, y_array, z_array, node_array, x_values, y_values, x_center, x_increment, y_center, y_increment):
    x_range = numpy.empty(2); x_range[0] = x_center - abs(x_increment); x_range[1] = x_center + abs(x_increment)
    y_range = numpy.empty(2); y_range[0] = y_center - abs(y_increment); y_range[1] = y_center + abs(y_increment)
    temp_counter = -1
    for counter in range(numpy.shape(z_array)[0]):
        if node_array[counter] > -1:
            if x_values[int(x_array[counter])] >= x_range[0] and x_values[int(x_array[counter])] <= x_range[1]:
                if y_values[int(y_array[counter])] >= y_range[0] and y_values[int(y_array[counter])] <= y_range[1]:
                    if temp_counter == -1:
                        temp_counter = counter
                    elif z_array[counter] < z_array[temp_counter]:
                        temp_counter = counter
    return temp_counter
    
def node_arbitrary(x_array, y_array, x_values, y_values, x_center, y_center):
    shortest_distance = distance(x_center, y_center, x_values[int(x_array[0])], y_values[int(y_array[0])])
    temp_counter = 0
    for counter in range(1, numpy.shape(x_array)[0]):
        current_distance = distance(x_center, y_center, x_values[int(x_array[counter])], y_values[int(y_array[counter])])
        if current_distance < shortest_distance:
            shortest_distance = current_distance
            temp_counter = counter
    return temp_counter
    
def find_OT(input_matrix, node_array, x_values, y_values, x_origin_center, y_origin_center, x_origin_increment, y_origin_increment, x_target_center, y_target_center, x_target_increment, y_target_increment, origin_menu_var, target_menu_var, points_in_minima_groups):
    input_matrix_shape = numpy.shape(input_matrix)
    x_array, y_array, z_array = generate_three_column_format(get_index_array(numpy.empty(input_matrix_shape[0])), get_index_array(numpy.empty(input_matrix_shape[1])), input_matrix)
    
    if origin_menu_var == "USE SEARCH":
        origin_temp_counter = node_search(x_array, y_array, z_array, node_array, x_values, y_values, x_origin_center, x_origin_increment, y_origin_center, y_origin_increment)
    elif origin_menu_var == "ARBITRARY":
        origin_temp_counter = node_arbitrary(x_array, y_array, x_values, y_values, x_origin_center, y_origin_center)
    else:
        origin_temp_counter = points_in_minima_groups[int(origin_menu_var)][0]

    if target_menu_var == "USE SEARCH":
        target_temp_counter = node_search(x_array, y_array, z_array, node_array, x_values, y_values, x_target_center, x_target_increment, y_target_center, y_target_increment)
    elif target_menu_var == "ARBITRARY":
        target_temp_counter = node_arbitrary(x_array, y_array, x_values, y_values, x_target_center, y_target_center)
    else:
        target_temp_counter = points_in_minima_groups[int(target_menu_var)][0]

    origin_node_to_plot = ([0,0,0])
    target_node_to_plot = ([0,0,0])
    origin_temp_counter = int(origin_temp_counter)
    target_temp_counter = int(target_temp_counter)
    if origin_temp_counter != -1:
        origin_node_to_plot[0] = x_values[int(x_array[origin_temp_counter])]
        origin_node_to_plot[1] = y_values[int(y_array[origin_temp_counter])]
        origin_node_to_plot[2] = z_array[origin_temp_counter]
    if target_temp_counter != -1:
        target_node_to_plot[0] = x_values[int(x_array[target_temp_counter])]
        target_node_to_plot[1] = y_values[int(y_array[target_temp_counter])]
        target_node_to_plot[2] = z_array[target_temp_counter]

    if origin_temp_counter == -1 or target_temp_counter == -1 or origin_temp_counter == target_temp_counter:
        return 1, origin_node_to_plot, target_node_to_plot, origin_temp_counter, target_temp_counter
    else:
        print("ORIGIN: " + str(origin_node_to_plot[0]) + " " + str(origin_node_to_plot[1]))
        print("TARGET: " + str(target_node_to_plot[0]) + " " + str(target_node_to_plot[1]))
        return 0, origin_node_to_plot, target_node_to_plot, origin_temp_counter, target_temp_counter

def rectangle_stamp(input_matrix, sorted_x_axis, sorted_y_axis, center_x, center_y, x_increment, y_increment, value_to_add):
    if value_to_add != 0:
        last_pos = numpy.shape(input_matrix)[0]-1
        out_matrix = numpy.empty((1,numpy.shape(sorted_x_axis)[0],numpy.shape(sorted_y_axis)[0]))
        out_matrix[0] = input_matrix[last_pos]
        x_up_limit, x_down_limit = find_limits(sorted_x_axis,center_x,x_increment)
        y_up_limit, y_down_limit = find_limits(sorted_y_axis,center_y,y_increment)
        for x in range(numpy.shape(sorted_x_axis)[0]):
            if sorted_x_axis[x]<=x_up_limit and sorted_x_axis[x]>=x_down_limit:
                for y in range(numpy.shape(sorted_y_axis)[0]):
                    if sorted_y_axis[y]<=y_up_limit and sorted_y_axis[y]>=y_down_limit:
                        out_matrix[0][x][y] += value_to_add
        return numpy.append(input_matrix, out_matrix, axis = 0)
    else:
        return input_matrix

def distance(a_x, a_y, b_x, b_y):
    return  math.sqrt(((a_x-b_x) * (a_x-b_x))+((a_y-b_y) * (a_y-b_y)))

def circle_stamp(input_matrix, sorted_x_axis, sorted_y_axis, center_x, center_y, radius, value_to_add):
    if value_to_add != 0:
        last_pos = numpy.shape(input_matrix)[0]-1
        out_matrix = numpy.empty((1,numpy.shape(sorted_x_axis)[0],numpy.shape(sorted_y_axis)[0]))
        out_matrix[0] = input_matrix[last_pos]
        x_up_limit, x_down_limit = find_limits(sorted_x_axis,center_x,radius)
        y_up_limit, y_down_limit = find_limits(sorted_y_axis,center_y,radius)
        for x in range(numpy.shape(sorted_x_axis)[0]):
            if sorted_x_axis[x]<=x_up_limit and sorted_x_axis[x]>=x_down_limit:
                for y in range(numpy.shape(sorted_y_axis)[0]):
                    if sorted_y_axis[y]<=y_up_limit and sorted_y_axis[y]>=y_down_limit:
                        if(distance(center_x, center_y, sorted_x_axis[x], sorted_y_axis[y]) <= radius):
                            out_matrix[0][x][y] += value_to_add
        return numpy.append(input_matrix, out_matrix, axis = 0)
    else:
        return input_matrix
        
def invert_map(input_matrix, sorted_x_axis, sorted_y_axis):
    last_pos = numpy.shape(input_matrix)[0]-1
    out_matrix = numpy.empty((1,numpy.shape(sorted_x_axis)[0],numpy.shape(sorted_y_axis)[0]))
    out_matrix[0] = input_matrix[last_pos]
    for x in range(numpy.shape(sorted_x_axis)[0]):
        for y in range(numpy.shape(sorted_y_axis)[0]):
            out_matrix[0][x][y] = -out_matrix[0][x][y]
    return numpy.append(input_matrix, out_matrix, axis = 0)
            
def surface_plot(xi, yi, x_to_plot, y_to_plot, z_to_plot):
    print("Generating surface plot")
    try:
        plt.close(1)
    except:
        pass  
    plt.figure(1)
    xmax, xmin = max(xi), min(xi)
    ymax, ymin = max(yi), min(yi)
    zi = ml.griddata(x_to_plot, y_to_plot, z_to_plot, xi, yi, interp='linear')
    plt.contour(xi, yi, zi, 15, linewidths = 0.5, colors = 'k')
    plt.pcolormesh(xi, yi, zi, cmap = plt.get_cmap('rainbow'))
    plt.colorbar()
    incr = (xmax-xmin)/100
    plt.xlim(xmin-incr, xmax+incr)
    incr = (ymax-ymin)/100
    plt.ylim(ymin-incr, ymax+incr)
    
def add_points_to_surface_plot(*args, **kwargs):
    point_counter = 0
    if ('color' in kwargs):
        in_color = kwargs['color']   
    else:
        in_color = ["black",]
    if ('data_type' in kwargs):
        data_type = kwargs['data_type']   
    else:
        data_type = "numpy_array"
    color_counter = 0
    color_array = ("black","red","#00ff00","blue","yellow","cyan","#ff748c","#f5f5f5","#421010")
    if data_type == "list":
        point_set_counter = 0
        for arg in args:
            point_set_counter += 1
            for points_list in arg:
                #print("Adding points")
                temp_arg = numpy.asarray(points_list)
                if (len(in_color)>1):
                    temp_color = in_color[color_counter]
                else:
                    temp_color = in_color[0]
                    if (temp_color == "multi"):
                        if (color_counter > numpy.shape(color_array)[0]-1):
                            color_counter = 0
                        temp_color = color_array[color_counter]
                x = temp_arg.transpose()[0]
                y = temp_arg.transpose()[1]
                plt.scatter(x,y,color = temp_color, alpha=0.5)
                color_counter += 1
                #print("Done")
    elif data_type == "annotate":
        if ('color' in kwargs):
            pass
        else:
            in_color = ["yellow",]
        for arg in args:
            #print("Annotating points")
            if (len(in_color)>1):
                temp_color = in_color[color_counter]
            else:
                temp_color = in_color[0]
                if (temp_color == "multi"):
                    if (color_counter > 7):
                        color_counter = 0
                    temp_color = color_array[color_counter]
            if len(numpy.shape(arg)) == 2:
                for counter in range(numpy.shape(arg)[0]):
                    point_counter += 1
                    print_progress (point_counter,numpy.shape(arg)[0])
                    plt.annotate( str(arg[counter][2]), xy = (arg[counter][0],arg[counter][1]), xytext = (-20, 20), textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.5', fc = temp_color, alpha = 0.5), arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
            elif len(numpy.shape(arg)) == 1:
                point_counter += 1
                print_progress (point_counter,numpy.shape(arg)[0])
                plt.annotate( str(arg[2]), xy = (arg[0],arg[1]), xytext = (-20, 20), textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.5', fc = temp_color, alpha = 0.5), arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
            color_counter+=1
            #print("Done")
    else:
        for arg in args:
            #print("Adding points")
            if (len(in_color)>1):
                temp_color = in_color[color_counter]
            else:
                temp_color = in_color[0]
                if (temp_color == "multi"):
                    if (color_counter > 7):
                        color_counter = 0
                    temp_color = color_array[color_counter]
            if len(numpy.shape(arg)) == 2:
                x = arg.transpose()[0]
                y = arg.transpose()[1]
                plt.scatter(x,y,color = temp_color, alpha=0.5)
            elif len(numpy.shape(arg)) == 1:
                point_counter += 1
                print_progress (point_counter,numpy.shape(arg)[0])
                plt.scatter(arg[0],arg[1],color = temp_color, alpha=0.5)
            color_counter+=1
            #print("Done")

def line_plot(*args, **kwargs):
    print("Line plot")
    if ('color' in kwargs):
        in_color = kwargs['color']   
    else:
        in_color = ["black",]
    if ('data_type' in kwargs):
        data_type = kwargs['data_type']   
    else:
        data_type = "numpy_array"
    try:
        plt.close(2)
    except:
        pass  
    plt.figure(2)
    color_counter = 0
    color_array = ("black","red","#00ff00","blue","orange","magenta","cyan","yellow","#ff748c","#f5f5f5","#421010")
    if data_type == "list":
        for arg in args:
            for points_list in arg:
                pre_temp_arg = numpy.asarray(points_list)
                temp_arg = numpy.empty(numpy.shape(pre_temp_arg)[0])
                for counter in range (numpy.shape(temp_arg)[0]):
                    temp_arg[counter] = pre_temp_arg[counter][2]
                if (len(in_color)>1):
                    temp_color = in_color[color_counter]
                else:
                    temp_color = in_color[0]
                    if (temp_color == "multi"):
                        if (color_counter > 7): 
                            color_counter = 0
                        temp_color = color_array[color_counter]
                x = numpy.arange (0,numpy.shape(temp_arg)[0],1)
                plt.plot(x,temp_arg,color = temp_color)
                color_counter += 1
    else:
        for arg in args:
            if (len(in_color)>1):
                temp_color = in_color[color_counter]
            else:
                temp_color = in_color[0]
                if (temp_color == "multi"):
                    if (color_counter > 7):
                        color_counter = 0
                    temp_color = color_array[color_counter]
            x = numpy.arange (0,numpy.shape(arg)[0],1)
            plt.plot(x,arg,color = temp_color)
            color_counter+=1

def generate_three_column_format(sorted_x_array, sorted_y_array, z_matrix):
    sorted_x_length = numpy.shape(sorted_x_array); sorted_y_length = numpy.shape(sorted_y_array)
    x_to_plot = numpy.empty((sorted_x_length[0]*sorted_y_length[0]))
    y_to_plot = numpy.empty((sorted_x_length[0]*sorted_y_length[0]))
    z_to_plot = numpy.empty((sorted_x_length[0]*sorted_y_length[0]))
    counter_to_plot = 0
    for counter_x in range(0, sorted_x_length[0]):
       for counter_y in range (0, sorted_y_length[0]):
           x_to_plot[counter_to_plot] = sorted_x_array[counter_x]
           y_to_plot[counter_to_plot] = sorted_y_array[counter_y]
           z_to_plot[counter_to_plot] = z_matrix[counter_x][counter_y]
           counter_to_plot+=1
    return x_to_plot, y_to_plot, z_to_plot

def sort_and_clean(in_array):
    sorted_array = numpy.sort(in_array)
    array_length = numpy.shape(sorted_array)
    sort_index = [False] * array_length[0]
    sort_index[0]=True
    for counter_1 in range(1, array_length[0]):
        if sorted_array[counter_1] != sorted_array[counter_1-1]:
            sort_index[counter_1]=True
    out_array = []
    for counter_1 in range(0, array_length[0]):
        if sort_index[counter_1] == True:
            out_array.append(sorted_array[counter_1])
    out_numpy_array = numpy.array(out_array)
    return out_numpy_array

def three_dimension_sorter(raw_x_array, raw_y_array, raw_z_array):
    raw_x_length = numpy.shape(raw_x_array); raw_y_length = numpy.shape(raw_y_array); raw_z_length = numpy.shape(raw_z_array)
    if raw_x_length[0] <= 0 or raw_x_length[0] != raw_y_length[0] or raw_x_length[0] != raw_z_length[0]:
        raise
    else:
        sorted_x = sort_and_clean(raw_x_array); sorted_y = sort_and_clean(raw_y_array)
        sorted_x_length = numpy.shape(sorted_x); sorted_y_length = numpy.shape(sorted_y)
        out_matrix = numpy.empty((1,sorted_x_length[0],sorted_y_length[0]))
        x_index_array = numpy.array(raw_x_array); y_index_array = numpy.array(raw_y_array)
        for counter_x in range(0, sorted_x_length[0]):
            x_index_array[raw_x_array == sorted_x[counter_x]] = counter_x
        for counter_y in range(0, sorted_y_length[0]):
            y_index_array[raw_y_array == sorted_y[counter_y]] = counter_y
        for counter_1 in range(0, raw_z_length[0]):
            out_matrix[0][int(x_index_array[counter_1])][int(y_index_array[counter_1])] = raw_z_array[counter_1]  
        return sorted_x, sorted_y, out_matrix

########################################
########################################


def main_exit_handler():
     if tkinter.messagebox.askokcancel("Quit?", "Are you sure you want to quit?"):
        try:
            plt.close(1)
        except:
            pass  
        try:
            plt.close(2)
        except:
            pass 
        try:
            root.map_editor_top.destroy()
        except:
            pass
        try:
            root.path_generator_top.destroy()
        except:
            pass
        root.destroy()
########################################
####################### GUI


class TkMainDialog(tkinter.Frame):

    def __init__(self, root):
        print("\n\n#####################################################################\nMEPSA (Minimum Energy Path Surface Analysis) v." + version + "\n#####################################################################\nCopyright (C) 2015, Iñigo Marcos Alcalde\n#####################################################################\n#                            LICENSE                                #\n#####################################################################        \nMEPSA is free software: you can redistribute it and/or modify\nit under the terms of the GNU General Public License as published by\nthe Free Software Foundation, either version 3 of the License, or\n(at your option) any later version.\n\nThis program is distributed in the hope that it will be useful,\nbut WITHOUT ANY WARRANTY; without even the implied warranty of\nMERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the\nGNU General Public License for more details.\n\nYou should have received a copy of the GNU General Public License\nalong with this program. If not, see <http://www.gnu.org/licenses/>.\n#####################################################################\n#####################################################################\nContact: pagomez@cbm.csic.es and imarcos@cbm.csic.es\n#####################################################################\nCitation: \nIñigo Marcos-Alcalde, Javier Setoain, Jesús I. Mendieta-Moreno, Jesús Mendieta, and Paulino Gómez-Puertas. (2015) MEPSA: minimum energy pathway analysis for energy landscapes. Bioinformatics 31: 3853-3855\n#####################################################################\n\n")
        tkinter.Frame.__init__(self, root)
        root.protocol("WM_DELETE_WINDOW", main_exit_handler)
        self.file_path = 'none'
        root.title("mepsa")
        # predefined variables
        self.path_not_found = 1
        self.smooth_path_not_found = 1
        self.nodes_not_found = 1
        self.OT_nodes_not_found = 1
        self.global_connectivity_not_found = 1
        self.empty_stack = 1
        self.well_sampling_not_done = 1
        self.autoplot_string = "Auto-plot: On"
        self.is_circle_string = "Rectangle"
        self.is_circle = False
        self.map_mod_center = [0,0]
        self.map_mod_increment = [0,0]
        self.map_mod_value_to_add = [0,0] #only the fists postion is used, but the second is kept to make the list work with some functions
        self.origin_center = [0,0]
        self.origin_increment = [0,0]
        self.target_center = [0,0]
        self.target_increment = [0,0]
        self.is_4_axes_string = "2 axis smooth" #deprecated option - before it was selectable in the same way as self.is_4_axes_smooth_string and self.is_4_axes_smooth_bool
        self.is_4_axes_bool = False #deprecated option - before it was selectable in the same way as self.is_4_axes_smooth_string and self.is_4_axes_smooth_bool
        self.is_4_axes_smooth_string = "2 axis smooth"
        self.is_4_axes_smooth_bool = False
        self.path_points = ([0,])
        self.map_editor_is_open = False
        self.path_generator_is_open = False

        # options for buttons that use pack (the ones in self)
        self.button_opt = {'fill': tkinter.constants.BOTH, 'padx': 5, 'pady': 5}

        # define buttons
        font = tkinter.font.Font(size = 10, weight = "bold")
        self.mepsa_label = tkinter.Label(self, text = "MEPSA", font = font)
        self.mepsa_label.pack(fill = tkinter.constants.BOTH, padx = 10, pady = 10)
        self.autoplot_button = tkinter.Button(self, text=self.autoplot_string, command=self.autoplot_change)
        self.autoplot_button.pack(self.button_opt)
        self.load_button = tkinter.Button(self, text='Load map file', command=self.ask_for_input_map)
        self.load_button.pack(self.button_opt)
        self.map_label = tkinter.Label(self, text = self.file_path)
        self.map_label.pack(self.button_opt)
        self.unload_button = tkinter.Button(self, text='Unload map file', command=self.unload_map)
        self.unload_button.config(state ='disabled')
        self.unload_button.pack(self.button_opt)
        self.plot_button = tkinter.Button(self, text='Plot map', command=self.plot_map)
        self.plot_button.config(state ='disabled')
        self.plot_button.pack(self.button_opt)
        self.map_editor_button = tkinter.Button(self, text='Map editor', command=self.map_editor)
        self.map_editor_button.config(state ='disabled')
        self.map_editor_button.pack(self.button_opt)
        self.path_generator_button = tkinter.Button(self, text='Connectivity analyses', command=self.path_generator    )
        self.path_generator_button.config(state ='disabled')
        self.path_generator_button.pack(self.button_opt)                
    
        # define options for opening a file
        self.file_openin_opt = options = {}
        options['filetypes'] = [('all files', '.*'), ('map files', '.dat')]
        options['initialfile'] = 'my_map.dat'
        options['parent'] = root
        options['title'] = 'Input map file'
  
        # define options for saving a file
        self.file_saving_opt = options = {}
        options['filetypes'] = [('all files', '.*'), ('map files', '.dat')]
        options['initialfile'] = 'my_out_path_file.dat'
        options['parent'] = root
        options['title'] = 'Save path file'
  
    ######## button functions
    def autoplot_change(self):
        if self.autoplot_string == "Auto-plot: On":
            self.autoplot_string = "Auto-plot: Off"
        else:
            self.autoplot_string = "Auto-plot: On"
        self.autoplot_button.config(text = self.autoplot_string)
        self.update_idletasks()
    
    def ask_for_input_map(self):
        self.file_path = tkinter.filedialog.askopenfilename()
        if self.file_path is None or len(self.file_path) == 0:
            return
        if self.file_path != '' and self.file_path != 'none':
            self.map_label.config(text = 'Loading...')
            self.update_idletasks()
            try:
                temp_x, temp_y, temp_z = numpy.loadtxt(self.file_path, unpack=True)
            except:
                print((sys.exc_info()))
                tkinter.messagebox.showwarning(
                    "Input map file",
                    "There was a problem loading the map\nCheck the format of the input file")
                self.file_path = 'none'
                self.map_label.config(text = self.file_path)
                self.update_idletasks()
                return
            try:
                self.x, self.y, self.maps_list = three_dimension_sorter(temp_x, temp_y, temp_z)
            except:
                print((sys.exc_info()))
                tkinter.messagebox.showwarning(
                    "Input map file",
                    "There was a problem in three_dimension_sorter")
                self.file_path = 'none'
                self.map_label.config(text = self.file_path)
                self.update_idletasks()
                return
            self.unload_button.config(state ='normal')
            self.unload_button.pack(self.button_opt)
            self.plot_button.config(state ='normal')
            self.plot_button.pack(self.button_opt)
            self.map_editor_button.config(state ='normal')
            self.map_editor_button.pack(self.button_opt)
            self.path_generator_button.config(state ='normal')
            self.path_generator_button.pack(self.button_opt)
            self.load_button.config(state ='disabled')
            self.load_button.pack(self.button_opt)
            self.map_label.config(text = self.file_path)
            self.update_idletasks()    
            self.path_not_found = 1
            self.nodes_not_found = 1
            self.smooth_path_not_found = 1
            self.OT_nodes_not_found = 1
            self.global_connectivity_not_found  = 1
            self.well_sampling_not_done = 1
            self.path_points = ([0,])
            if (self.autoplot_string == "Auto-plot: On"):
                self.plot_map()
        else:
            self.file_path = 'none'
            self.map_label.config(text = self.file_path)
            self.update_idletasks()
    
    def unload_map(self):
        self.file_path = 'none'
        self.map_label.config(text = 'Unloading...')
        self.update_idletasks()
        self.unload_button.config(state ='disabled'); self.unload_button.pack(self.button_opt)
        self.plot_button.config(state ='disabled')
        self.plot_button.pack(self.button_opt)
        self.map_editor_button.config(state ='disabled')
        self.map_editor_button.pack(self.button_opt)
        self.path_generator_button.config(state ='disabled')
        self.path_generator_button.pack(self.button_opt)
        del self.x
        del self.y
        del self.maps_list
        try:
            del self.paths_list
        except:
            pass
        try:
            plt.close(1)
        except:
            pass
        try:
            plt.close(2)
        except:
            pass
        self.load_button.config(state ='normal'); self.load_button.pack(self.button_opt)
        self.map_label.config(text = self.file_path)
        self.update_idletasks()

    def plot_map(self):
        self.plot_button.config(text='PLOTTING...')
        self.plot_button.pack()
        self.plot_button.update()
        x_to_plot, y_to_plot, z_to_plot = generate_three_column_format(self.x, self.y, self.maps_list[numpy.shape(self.maps_list)[0]-1])
        surface_plot(self.x, self.y, x_to_plot, y_to_plot, z_to_plot)
        self.plot_button.config(text='Plot map')
        self.plot_button.pack()
        self.update_idletasks()
        plt.show()

########################################
####################### Map editor
    #main
    def map_editor(self):
        self.map_editor_top = tkinter.Toplevel(master=root)
        self.map_editor_top.entry_strings = ["",""]
        self.map_editor_top.protocol("WM_DELETE_WINDOW", self.map_editor_exit_handler)
        self.map_editor_top.title("Map editor")
        self.map_editor_is_open = True
        
        ####widgets
        self.map_editor_top.spacing_label_0 = tkinter.Label(self.map_editor_top, text = "_____________________________\n\nMODIFY MAP\n_____________________________\n")
        self.map_editor_top.spacing_label_0.grid(row=0,columnspan=2)
        self.map_editor_top.circle_button = tkinter.Button(self.map_editor_top, text=self.is_circle_string, command=self.is_circle_button)
        self.map_editor_top.circle_button.grid(row=1,columnspan=2, pady=5)
        self.map_editor_top.center_label = tkinter.Label(self.map_editor_top, text = "\nStamp center coordinates")
        self.map_editor_top.center_label.grid(row=2,columnspan=2)
        self.map_editor_top.center_values_label = tkinter.Label(self.map_editor_top, text = "X | Y")
        self.map_editor_top.center_values_label.grid(row=3,columnspan=2)
        self.map_editor_top.center_x_entry = tkinter.Entry(self.map_editor_top, width = 10)
        self.map_editor_top.center_y_entry = tkinter.Entry(self.map_editor_top, width = 10)
        self.map_editor_top.center_x_entry.grid(row=4, column=0)
        self.map_editor_top.center_y_entry.grid(row=4, column=1)

        def callback_center():
            self.map_editor_top.entry_strings[0] = self.map_editor_top.center_x_entry.get()
            self.map_editor_top.entry_strings[1] = self.map_editor_top.center_y_entry.get()
            self.set_two_values(self.map_editor_top.entry_strings, self.map_mod_center)
            self.map_editor_top.center_values_label.config(text = "X | Y")
            self.map_editor_top.update_idletasks()

        self.map_editor_top.stamp_increments_label = tkinter.Label(self.map_editor_top, text = "\nX Y increments")
        self.map_editor_top.stamp_increments_label.grid(row=5,columnspan=2)
        self.map_editor_top.stamp_increments_values_label = tkinter.Label(self.map_editor_top, text = "X | Y")
        self.map_editor_top.stamp_increments_values_label.grid(row=6,columnspan=2)
        self.map_editor_top.increment_x_entry = tkinter.Entry(self.map_editor_top, width = 10)
        self.map_editor_top.increment_y_entry = tkinter.Entry(self.map_editor_top, width = 10)
        self.map_editor_top.increment_x_entry.grid(row=7, column=0)
        self.map_editor_top.increment_y_entry.grid(row=7, column=1)

        def callback_increments():
            self.map_editor_top.entry_strings[0] = self.map_editor_top.increment_x_entry.get()
            self.map_editor_top.entry_strings[1] = self.map_editor_top.increment_y_entry.get()
            self.set_two_values(self.map_editor_top.entry_strings, self.map_mod_increment)
            if self.is_circle == True:
                self.map_editor_top.stamp_increments_values_label.config(text = "")
            else:
                self.map_editor_top.stamp_increments_values_label.config(text = "X | Y")
            self.map_editor_top.update_idletasks()
        self.map_editor_top.value_to_add_label = tkinter.Label(self.map_editor_top, text = "\nValue to add")
        self.map_editor_top.value_to_add_label.grid(row=8,columnspan=2)
        self.map_editor_top.value_to_add_entry = tkinter.Entry(self.map_editor_top, width = 10)
        self.map_editor_top.value_to_add_entry.grid(row=10,columnspan=2,pady=10)

        def callback_value_to_add():
            self.map_editor_top.entry_strings[0] = self.map_editor_top.value_to_add_entry.get()
            self.set_two_values(self.map_editor_top.entry_strings, self.map_mod_value_to_add)
            self.map_editor_top.update_idletasks()

        def callback_set_all():
            callback_value_to_add()
            callback_increments()
            callback_center()
            
        def callback_modify():
            callback_set_all()
            if self.is_circle == True:
                self.maps_list = (circle_stamp(self.maps_list, self.x, self.y, self.map_mod_center[0], self.map_mod_center[1],self.map_mod_increment[0], self.map_mod_value_to_add[0]))
            else:
                self.maps_list = (rectangle_stamp(self.maps_list, self.x, self.y, self.map_mod_center[0], self.map_mod_center[1],self.map_mod_increment[0],self.map_mod_increment[1], self.map_mod_value_to_add[0]))
            if numpy.shape(self.maps_list)[0] > 1:
                self.map_editor_top.set_undo_button.config(state="normal")
                self.map_editor_top.update_idletasks()
                self.path_not_found = 1
                self.nodes_not_found = 1
                self.smooth_path_not_found = 1
                self.OT_nodes_not_found = 1
                self.global_connectivity_not_found = 1
                self.well_sampling_not_done = 1
                try:
                    if 'normal' == self.path_generator_top.state():
                        self.path_generator_top.destroy()
                        self.path_generator()
                except:
                    pass
                if (self.autoplot_string == "Auto-plot: On"):
                    self.plot_map()

        def callback_invert():
            self.maps_list = invert_map(self.maps_list, self.x, self.y)
            if numpy.shape(self.maps_list)[0] > 1:
                self.map_editor_top.set_undo_button.config(state="normal")
                self.map_editor_top.update_idletasks()
                #try:
                self.path_not_found = 1
                self.nodes_not_found = 1
                self.smooth_path_not_found = 1
                self.OT_nodes_not_found = 1
                self.global_connectivity_not_found = 1
                self.well_sampling_not_done = 1
                try:
                    if 'normal' == self.path_generator_top.state():
                        self.path_generator_top.destroy()
                        self.path_generator()
                except:
                    pass
                if (self.autoplot_string == "Auto-plot: On"):
                    self.plot_map()

        self.map_editor_top.set_modify_map_button = tkinter.Button(self.map_editor_top, text="MODIFY MAP", command=callback_modify)
        self.map_editor_top.set_modify_map_button.grid(row=12,columnspan=2,pady=10)
        
        self.map_editor_top.set_modify_map_button = tkinter.Button(self.map_editor_top, text="INVERT MAP", command=callback_invert)
        self.map_editor_top.set_modify_map_button.grid(row=13,columnspan=2,pady=10)
        
        self.map_editor_top.spacing_label_1 = tkinter.Label(self.map_editor_top, text = "_____________________________\n\nSMOOTH MAP\n_____________________________\n")
        self.map_editor_top.spacing_label_1.grid(row=14,columnspan=2)
        
        self.map_editor_top.is_4_axes_smooth_button = tkinter.Button(self.map_editor_top, text=self.is_4_axes_smooth_string, command=self.is_4_axes_smooth_button)
        self.map_editor_top.is_4_axes_smooth_button.grid(row=15,columnspan=2, pady=10)

        def callback_smooth_map():
            iterations = 0
            iterations = int(self.map_editor_top.smooth_map_spin.get())
            if iterations > 0:
                self.map_editor_top.smooth_map_button.config(text='RUNNING...')
                self.map_editor_top.smooth_map_button.grid()
                self.map_editor_top.smooth_map_button.update()
                self.maps_list = smooth_map(self.maps_list ,self.is_4_axes_smooth_bool, iterations)
                self.map_editor_top.smooth_map_button.config(text='SMOOTH MAP')
                self.map_editor_top.smooth_map_button.grid()
                if numpy.shape(self.maps_list)[0] > 1:
                    self.map_editor_top.set_undo_button.config(state="normal")
                    self.map_editor_top.update_idletasks()
                self.path_not_found = 1
                self.nodes_not_found = 1
                self.smooth_path_not_found = 1
                self.OT_nodes_not_found = 1
                self.global_connectivity_not_found = 1
                self.well_sampling_not_done = 1
                try:
                    if 'normal' == self.path_generator_top.state():
                        self.path_generator_top.destroy()
                        self.path_generator()
                except:
                    pass
                if (self.autoplot_string == "Auto-plot: On"):
                    self.plot_map()
            else:
                tkinter.messagebox.showwarning(
                    "ERROR",
                    "SMOOTH ITERATIONS VALUE HAS TO BE GREATER THAN 0\n")

        self.map_editor_top.smooth_map_spin = tkinter.Spinbox(self.map_editor_top, from_ = 1, to = 100, width=5)
        self.map_editor_top.smooth_map_spin.grid(row=16,column=0,pady=10)
        self.map_editor_top.smooth_map_button = tkinter.Button(self.map_editor_top, text="SMOOTH MAP", command=callback_smooth_map)
        self.map_editor_top.smooth_map_button.grid(row=16,column=1,pady=10)
        self.map_editor_top.spacing_label_2 = tkinter.Label(self.map_editor_top, text = "_____________________________\n")
        self.map_editor_top.spacing_label_2.grid(row=17,columnspan=2)

        def callback_undo():
            if numpy.shape(self.maps_list)[0] > 1:
                self.maps_list = numpy.delete(self.maps_list,numpy.shape(self.maps_list)[0]-1,0)
            if numpy.shape(self.maps_list)[0] == 1:
                self.map_editor_top.set_undo_button.config(state="disabled")
                self.map_editor_top.update_idletasks()
            self.path_not_found = 1
            self.nodes_not_found = 1
            self.smooth_path_not_found = 1
            self.OT_nodes_not_found = 1
            self.global_connectivity_not_found = 1
            self.well_sampling_not_done = 1
            try:
                if 'normal' == self.path_generator_top.state():
                    self.path_generator_top.destroy()
                    self.path_generator()
            except:
                pass
            if (self.autoplot_string == "Auto-plot: On"):
                self.plot_map()
        self.map_editor_top.set_undo_button = tkinter.Button(self.map_editor_top, text="UNDO", command=callback_undo)
        self.map_editor_top.set_undo_button.grid(row=18,column = 0,pady=5)
        self.map_editor_top.update_idletasks()

        def callback_save_map():
            file_to_save = tkinter.filedialog.asksaveasfile(mode='w', defaultextension=".txt")
            if file_to_save is None == 0:
                return
            path_counter = 0
            for x in range(numpy.shape(self.maps_list[numpy.shape(self.maps_list)[0]-1])[0]):
                for y in range(numpy.shape(self.maps_list[numpy.shape(self.maps_list)[0]-1])[1]):
                    file_to_save.write(str(self.x[x]) + " " + str(self.y[y]) + " " + str(self.maps_list[numpy.shape(self.maps_list)[0]-1][x][y]) + "\n")
            file_to_save.close()
        
        self.map_editor_top.save_map_button = tkinter.Button(self.map_editor_top, text="SAVE", command=callback_save_map)
        self.map_editor_top.save_map_button.grid(row=18,column = 1,pady=5)
        
        if self.is_circle == True:
            self.is_circle_string = "Circle"
            self.map_editor_top.circle_button.config(text = self.is_circle_string)
            self.map_editor_top.stamp_increments_label.config (text = "_____________________________\n\nRadius\n_____________________________\n")
            self.map_editor_top.stamp_increments_values_label.config(text = "")
            self.map_editor_top.increment_y_entry.config(state="disabled")
            self.map_editor_top.update_idletasks()
        if numpy.shape(self.maps_list)[0] == 1:
            self.map_editor_top.set_undo_button.config(state="disabled")
            self.map_editor_top.update_idletasks()

        ####end of widgets
        self.unload_button.config(state ='disabled')
        self.unload_button.pack(self.button_opt)
        self.map_editor_button.config(state ='disabled')
        self.map_editor_button.pack(self.button_opt)
        self.update_idletasks()

    def map_editor_exit_handler(self):
        self.map_editor_is_open = False
        if self.path_generator_is_open == False:
            self.unload_button.config(state ='normal')
            self.unload_button.pack(self.button_opt)
        self.map_editor_button.config(state ='normal')
        self.map_editor_button.pack(self.button_opt)
        self.map_editor_top.increment_y_entry = tkinter.Entry(self.map_editor_top)
        self.update_idletasks()
        self.map_editor_top.destroy()

    def is_circle_button(self):
        if self.is_circle == True:
            self.is_circle = False
            self.is_circle_string = "Rectangle"
            self.map_editor_top.circle_button.config(text = self.is_circle_string)
            self.map_editor_top.stamp_increments_label.config (text = "\nX Y increments")
            self.map_editor_top.stamp_increments_values_label.config(text = "X | Y")
            self.map_editor_top.increment_y_entry.config(state="normal")
        else:
            self.is_circle = True
            self.is_circle_string = "Circle"
            self.map_editor_top.circle_button.config(text = self.is_circle_string)
            self.map_editor_top.stamp_increments_label.config (text = "\nRadius")
            self.map_editor_top.stamp_increments_values_label.config(text = "")
            self.map_editor_top.increment_y_entry.config(state="disabled")
        self.map_editor_top.update_idletasks()

    def set_two_values(self, input_list, output_list):
        try:    
            x = float(input_list[0])
        except:
            x = 0.0
        try:
            y = float(input_list[1])
        except:
            y = 0.0
        output_list[0] = x
        output_list[1] = y

    def is_4_axes_smooth_button(self):
            if self.is_4_axes_smooth_bool== True:
                self.is_4_axes_smooth_bool= False
                self.is_4_axes_smooth_string = "2 axis smooth"
                self.map_editor_top.is_4_axes_smooth_button.config(text = self.is_4_axes_smooth_string)
            else:
                self.is_4_axes_smooth_bool= True
                self.is_4_axes_smooth_string = "4 axis smooth"
                self.map_editor_top.is_4_axes_smooth_button.config(text = self.is_4_axes_smooth_string)
                tkinter.messagebox.showwarning(
                    "WARNING",
                    "4 AXES SMOOTHING MODE IS NOT FULLY SUPPORTED.\nTHE CONTRIBUTION OF EACH SURROUNDING POINT IS NOT DISTANCE WEIGHTED.\nUSE IT AT YOUR OWN RISK.\n")
########################################
####################### Connectivity analyses
    #main
    def path_generator(self):
        self.path_generator_top = tkinter.Toplevel(master=root)
        self.path_generator_top.entry_strings = ["",""]
        self.path_generator_top.protocol("WM_DELETE_WINDOW", self.path_generator_exit_handler)
        self.path_generator_top.title("Connectivity analyses")
        self.path_generator_is_open = True

        #######STATUS RELATED BUTTON AVAILABILITY
        def OT_selection_availability(state_in):
            self.path_generator_top.origin_center_label.config(state = state_in)
            self.path_generator_top.origin_center_label.grid()
            self.path_generator_top.target_center_label.config(state = state_in)
            self.path_generator_top.target_center_label.grid()
            self.path_generator_top.origin_center_values_X_label.config(state = state_in)
            self.path_generator_top.origin_center_values_X_label.grid()
            self.path_generator_top.origin_center_values_Y_label.config(state = state_in)
            self.path_generator_top.origin_center_values_Y_label.grid()
            self.path_generator_top.target_center_values_X_label.config(state = state_in)
            self.path_generator_top.target_center_values_X_label.grid()
            self.path_generator_top.target_center_values_Y_label.config(state = state_in)
            self.path_generator_top.target_center_values_Y_label.grid()
            self.path_generator_top.stamp_origin_increments_label.config(state = state_in)
            self.path_generator_top.stamp_origin_increments_label.grid()
            self.path_generator_top.stamp_target_increments_label.config(state = state_in)
            self.path_generator_top.stamp_target_increments_label.grid()
            self.path_generator_top.stamp_origin_increments_values_X_label.config(state = state_in)
            self.path_generator_top.stamp_origin_increments_values_X_label.grid()
            self.path_generator_top.stamp_origin_increments_values_Y_label.config(state = state_in)
            self.path_generator_top.stamp_origin_increments_values_Y_label.grid()
            self.path_generator_top.stamp_target_increments_values_X_label.config(state = state_in)
            self.path_generator_top.stamp_target_increments_values_X_label.grid()
            self.path_generator_top.stamp_target_increments_values_Y_label.config(state = state_in)
            self.path_generator_top.stamp_target_increments_values_Y_label.grid()
            self.path_generator_top.origin_top_label.config(state = state_in)
            self.path_generator_top.origin_top_label.grid()
            self.path_generator_top.origin_menu_button.config(state = state_in)
            self.path_generator_top.origin_menu_button.grid()
            self.path_generator_top.target_top_label.config(state = state_in)
            self.path_generator_top.target_top_label.grid()
            self.path_generator_top.target_menu_button.config(state = state_in)
            self.path_generator_top.target_menu_button.grid()
            self.path_generator_top.find_OT_button.config(state = state_in)
            self.path_generator_top.find_OT_button.grid()

        def nodes_plot_availability(state_in):
            self.path_generator_top.plot_nodes_button.config(state = state_in)
            self.path_generator_top.plot_nodes_button.grid()

        def connectivity_analysis_availability(state_in):
            self.path_generator_top.analyse_global_connectivity_button.config(state = state_in)
            self.path_generator_top.analyse_global_connectivity_button.grid()

        def connectivity_plot_availability(state_in):
            self.path_generator_top.global_connectivity_plot_button.config(state = state_in)
            self.path_generator_top.global_connectivity_plot_button.grid()
            
        def OT_plot_availability(state_in):
            self.path_generator_top.OT_surf_plot_button.config(state = state_in)
            self.path_generator_top.OT_surf_plot_button.grid()

        def path_generator_availability(state_in):
            self.path_generator_top.gen_path_menu_button.config(state = state_in)
            self.path_generator_top.gen_path_menu_button.grid()

        def path_plots_availability(state_in):
            self.path_generator_top.full_surf_plot_button.config(state = state_in)
            self.path_generator_top.full_surf_plot_button.grid()
            self.path_generator_top.path_surf_plot_button.config(state = state_in)
            self.path_generator_top.path_surf_plot_button.grid()
            self.path_generator_top.path_line_plot_button.config(state = state_in)
            self.path_generator_top.path_line_plot_button.grid()

        def path_inverts_availability(state_in):
            self.path_generator_top.invert_path_values_button.config(state = state_in)
            self.path_generator_top.invert_path_values_button.grid()
            self.path_generator_top.invert_path_order_button.config(state = state_in)
            self.path_generator_top.invert_path_order_button.grid()

        def path_add_to_stack_availability(state_in):
            self.path_generator_top.path_add_to_stack_button.config(state = state_in)
            self.path_generator_top.path_add_to_stack_button.grid()

        def path_saves_availability(state_in):
            self.path_generator_top.save_path_button.config(state = state_in)
            self.path_generator_top.save_path_button.grid()
            self.path_generator_top.save_interest_points_button.config(state = state_in)
            self.path_generator_top.save_interest_points_button.grid()

        def path_smooth_availability(state_in):
            self.path_generator_top.smooth_button.config(state = state_in)
            self.path_generator_top.smooth_button.grid()

        def path_smooth_save_availability(state_in):
            self.path_generator_top.save_smooth_button.config(state = state_in)
            self.path_generator_top.save_smooth_button.grid()

        def path_smooth_plot_availability(state_in):
            self.path_generator_top.smooth_plot_button.config(state = state_in)
            self.path_generator_top.smooth_plot_button.grid()

        def well_sampling_availability(state_in):
            self.path_generator_top.calculate_well_sampling_button.config(state = state_in)
            self.path_generator_top.calculate_well_sampling_button.grid()

        def well_sampling_post_process_availability(state_in):
            self.path_generator_top.well_sampling_plot_button.config(state = state_in)
            self.path_generator_top.well_sampling_plot_button.grid()
            self.path_generator_top.save_well_sampling_button.config(state = state_in)
            self.path_generator_top.save_well_sampling_button.grid()

        def delete_stack_availability(state_in):
            self.path_generator_top.delete_last_from_stack_button.config(state = state_in)
            self.path_generator_top.delete_last_from_stack_button.grid()

        def save_stack_availability(state_in):
            self.path_generator_top.save_stack_button.config(state = state_in)
            self.path_generator_top.save_stack_button.grid()

        def stack_plots_availability(state_in):    
            self.path_generator_top.plot_surface_stack_button.config(state = state_in)
            self.path_generator_top.plot_surface_stack_button.grid()
            self.path_generator_top.plot_line_stack_button.config(state = state_in)
            self.path_generator_top.plot_line_stack_button.grid()

        def check_status():
            if self.nodes_not_found == 1:
                OT_selection_availability('disabled')
                nodes_plot_availability('disabled')
                connectivity_analysis_availability('disabled')
                self.global_connectivity_not_found = 1
                self.OT_nodes_not_found = 1
            else:
                OT_selection_availability('normal')
                nodes_plot_availability('normal')
                connectivity_analysis_availability('normal')
                
            if self.global_connectivity_not_found == 1:
                connectivity_plot_availability('disabled')
            else:
                connectivity_plot_availability('normal')
                    
            if self.OT_nodes_not_found == 1:
                OT_plot_availability('disabled')
                path_generator_availability('disabled')
                well_sampling_availability('disabled')
                self.well_sampling_not_done = 1
                self.path_not_found = 1
            else:
                OT_plot_availability('normal')
                path_generator_availability('normal')
                well_sampling_availability('normal')
                
                
            if self.path_not_found == 1:
                path_plots_availability('disabled')
                path_inverts_availability('disabled')
                path_add_to_stack_availability('disabled')
                path_saves_availability('disabled')
                path_smooth_availability('disabled')
                self.smooth_path_not_found = 1
            else:
                path_plots_availability('normal')
                path_inverts_availability('normal')
                path_add_to_stack_availability('normal')
                path_saves_availability('normal')
                path_smooth_availability('normal')
                
            if self.smooth_path_not_found == 1:
                path_smooth_save_availability('disabled')
                path_smooth_plot_availability('disabled')
            else:
                path_smooth_save_availability('normal')
                path_smooth_plot_availability('normal')
                
            if self.well_sampling_not_done == 1:
                well_sampling_post_process_availability('disabled')
            else:
                well_sampling_post_process_availability('normal')
                
            if self.empty_stack == 1:
                delete_stack_availability('disabled')
                save_stack_availability('disabled')
                stack_plots_availability('disabled')
            else:
                delete_stack_availability('normal')
                save_stack_availability('normal')
                stack_plots_availability('normal')
            self.path_generator_top.update_idletasks()
        #######END OF STATUS RELATED BUTTON AVAILABILITY

        ####widgets
        def callback_global_connectivity_plot():
            self.path_generator_top.global_connectivity_plot_button.config(text='PLOTTING...')
            self.path_generator_top.global_connectivity_plot_button.grid()
            self.path_generator_top.global_connectivity_plot_button.update()
            x_to_plot, y_to_plot, z_to_plot = generate_three_column_format(self.x, self.y, self.maps_list[numpy.shape(self.maps_list)[0]-1])
            surface_plot(self.x, self.y, x_to_plot, y_to_plot, z_to_plot)
            add_points_to_surface_plot(self.connectivity_points_stack, color = ["multi",], data_type = "list")
            add_points_to_surface_plot(self.connectivity_barrier_points, self.connectivity_minima_points, color = ["magenta", "orange"])
            add_points_to_surface_plot(self.connectivity_barrier_points, self.connectivity_minima_points, data_type = "annotate", color = ["magenta", "orange"])
            self.path_generator_top.global_connectivity_plot_button.config(text='CONNECTIVITY PLOT')
            self.path_generator_top.global_connectivity_plot_button.grid()
            plt.show()

        def callback_analyse_global_connectivity(mode):
            self.path_generator_top.analyse_global_connectivity_var.set("RUNNING...")
            self.path_generator_top.analyse_global_connectivity_button.grid()
            self.path_generator_top.analyse_global_connectivity_button.update()
            if mode == "FULL":
                well_sampling = False
            elif mode == "MINIMAL":
                well_sampling = True
            self.path_generator_top.analyse_global_connectivity_button.config(text='RUNNING...')
            self.path_generator_top.analyse_global_connectivity_button.grid()
            self.path_generator_top.analyse_global_connectivity_button.update()
            self.connectivity_points_stack, self.connectivity_barrier_points, self.connectivity_minima_points, self.connectivity_propagation_array = global_connectivity_analyis(self.maps_list[numpy.shape(self.maps_list)[0]-1], self.node_array, self.points_in_minima_groups, self.node_matrix, self.is_4_axes_bool, self.x, self.y,well_sampling)
            self.path_generator_top.analyse_global_connectivity_var.set("ANALYSE CONNECT.")
            self.path_generator_top.analyse_global_connectivity_button.grid()
            self.global_connectivity_not_found = 0
            check_status()
            print("Connectivity analysis done")
            if (self.autoplot_string == "Auto-plot: On"):
                callback_global_connectivity_plot()

        def callback_node_plot():
            self.path_generator_top.plot_nodes_button.config(text='PLOTTING...')
            self.path_generator_top.plot_nodes_button.grid()
            self.path_generator_top.plot_nodes_button.update()
            x_to_plot, y_to_plot, z_to_plot = generate_three_column_format(self.x, self.y, self.maps_list[numpy.shape(self.maps_list)[0]-1])
            surface_plot(self.x, self.y, x_to_plot, y_to_plot, z_to_plot)
            add_points_to_surface_plot(self.node_points_to_plot)
            add_points_to_surface_plot(self.node_points_to_plot, data_type = "annotate")
            self.path_generator_top.plot_nodes_button.config(text='NODES PLOT')
            self.path_generator_top.plot_nodes_button.grid()
            plt.show()

        def callback_find_nodes(mode):
            self.path_generator_top.gen_node_menu_var.set("RUNNING...")
            self.path_generator_top.find_nodes_button.grid()
            self.path_generator_top.find_nodes_button.update()
            if (mode == "MIN AND FLAT"):
                flat_nodes_bool = True
            else:
                flat_nodes_bool = False
            self.nodes_not_found, self.node_points_to_plot, self.node_array, self.points_in_minima_groups, self.node_matrix = node_finder (self.maps_list[numpy.shape(self.maps_list)[0]-1], self.is_4_axes_bool, self.x, self.y, flat_nodes_bool)
            self.path_generator_top.gen_node_menu_var.set("FIND NODES")
            self.path_generator_top.find_nodes_button.grid()
            if (self.nodes_not_found != 0):
                check_status()
                tkinter.messagebox.showwarning(
                    "ERROR",
                    "NODE FINDING FAILED\n")
            else:
                self.path_generator_top.origin_menu_button['menu'].delete(0, 'end')
                self.path_generator_top.origin_menu_button['menu'].add_command(label="USE SEARCH", command=tkinter._setit(self.path_generator_top.origin_menu_var, "USE SEARCH"))
                self.path_generator_top.origin_menu_button['menu'].add_command(label="ARBITRARY", command=tkinter._setit(self.path_generator_top.origin_menu_var, "ARBITRARY"))
                for counter in range(numpy.shape(self.points_in_minima_groups)[0]):
                    self.path_generator_top.origin_menu_button['menu'].add_command(label=counter, command=tkinter._setit(self.path_generator_top.origin_menu_var, counter))
                self.path_generator_top.origin_menu_var.set("USE SEARCH")
                self.path_generator_top.origin_menu_button.grid()
                self.path_generator_top.target_menu_button['menu'].delete(0, 'end')
                self.path_generator_top.target_menu_button['menu'].add_command(label="USE SEARCH", command=tkinter._setit(self.path_generator_top.target_menu_var, "USE SEARCH", "ARBITRARY"))
                self.path_generator_top.target_menu_button['menu'].add_command(label="ARBITRARY", command=tkinter._setit(self.path_generator_top.target_menu_var, "ARBITRARY"))
                for counter in range(numpy.shape(self.points_in_minima_groups)[0]):
                    self.path_generator_top.target_menu_button['menu'].add_command(label=counter, command=tkinter._setit(self.path_generator_top.target_menu_var, counter))
                check_status()
                if (self.autoplot_string == "Auto-plot: On"):
                    callback_node_plot()


        self.path_generator_top.gen_node_menu_var = tkinter.StringVar(self.path_generator_top)
        self.path_generator_top.gen_node_menu_var.set("FIND NODES")
        self.path_generator_top.find_nodes_button = tkinter.OptionMenu(self.path_generator_top, self.path_generator_top.gen_node_menu_var, "MIN ONLY", "MIN AND FLAT", command=callback_find_nodes)
        self.path_generator_top.find_nodes_button.grid(row=0,column=0,pady=15)
        self.path_generator_top.plot_nodes_button = tkinter.Button(self.path_generator_top, text="NODES PLOT", command=callback_node_plot)
        self.path_generator_top.plot_nodes_button.grid(row=0,column=1, pady=15)

        self.path_generator_top.analyse_global_connectivity_var = tkinter.StringVar(self.path_generator_top)
        self.path_generator_top.analyse_global_connectivity_var.set("ANALYSE CONNECT.")
        self.path_generator_top.analyse_global_connectivity_button = tkinter.OptionMenu(self.path_generator_top, self.path_generator_top.analyse_global_connectivity_var, "FULL", "MINIMAL", command=callback_analyse_global_connectivity)


#        self.path_generator_top.gen_path_menu_var = tkinter.StringVar(self.path_generator_top)
#        self.path_generator_top.gen_path_menu_var.set("GENERATE PATH")
#        self.path_generator_top.gen_path_menu_button = tkinter.OptionMenu(self.path_generator_top, self.path_generator_top.gen_path_menu_var, "GLOBAL", "NODE BY NODE", command=callback_gen_path)
#        self.path_generator_top.gen_path_menu_button.grid(row=10,column=0,pady=15)



#        self.path_generator_top.analyse_global_connectivity_button = tkinter.Button(self.path_generator_top, text="ANALYSE CONNECT.", command=callback_analyse_global_connectivity)
        self.path_generator_top.analyse_global_connectivity_button.grid(row=0,column=3, pady=15)
 

        self.path_generator_top.global_connectivity_plot_button = tkinter.Button(self.path_generator_top, text="CONNECT. PLOT", command=callback_global_connectivity_plot)
        self.path_generator_top.global_connectivity_plot_button.grid(row=0,column=4,columnspan=2, pady=15)
        self.path_generator_top.origin_top_label = tkinter.Label(self.path_generator_top, text = "\nOrigin\n_____________________________\n")
        self.path_generator_top.origin_top_label.grid(row=1,column= 0, columnspan=2)
        self.path_generator_top.origin_menu_var = tkinter.StringVar(self.path_generator_top)
        self.path_generator_top.origin_menu_var.set("USE SEARCH")
        self.path_generator_top.origin_menu_button = tkinter.OptionMenu(self.path_generator_top, self.path_generator_top.origin_menu_var, "USE SEARCH")
        self.path_generator_top.origin_menu_button.grid(row=2,column= 0, columnspan=2)
        self.path_generator_top.target_top_label = tkinter.Label(self.path_generator_top, text = "\nTarget\n_____________________________\n")
        self.path_generator_top.target_top_label.grid(row=1,column= 3, columnspan=2)
        self.path_generator_top.target_menu_var = tkinter.StringVar(self.path_generator_top)
        self.path_generator_top.target_menu_var.set("USE SEARCH")
        self.path_generator_top.target_menu_button = tkinter.OptionMenu(self.path_generator_top, self.path_generator_top.target_menu_var, "USE SEARCH")
        self.path_generator_top.target_menu_button.grid(row=2,column= 3, columnspan=2)
        self.path_generator_top.origin_center_label = tkinter.Label(self.path_generator_top, text = "\nOrigin search center coordinates\n_____________________________\n")
        self.path_generator_top.origin_center_label.grid(row=3,column= 0, columnspan=2)
        self.path_generator_top.origin_center_values_X_label = tkinter.Label(self.path_generator_top, text = "X", width = 10)
        self.path_generator_top.origin_center_values_Y_label = tkinter.Label(self.path_generator_top, text = "Y", width = 10)
        self.path_generator_top.origin_center_values_X_label.grid(row=4, column=0)
        self.path_generator_top.origin_center_values_Y_label.grid(row=4, column=1)
        self.path_generator_top.origin_center_x_entry = tkinter.Entry(self.path_generator_top, width = 10)
        self.path_generator_top.origin_center_y_entry = tkinter.Entry(self.path_generator_top, width = 10)
        self.path_generator_top.origin_center_x_entry.grid(row=5, column=0)
        self.path_generator_top.origin_center_y_entry.grid(row=5, column=1)

        def callback_origin_center():
            self.path_generator_top.entry_strings[0] = self.path_generator_top.origin_center_x_entry.get()
            self.path_generator_top.entry_strings[1] = self.path_generator_top.origin_center_y_entry.get()
            self.set_two_values(self.path_generator_top.entry_strings, self.origin_center)
        self.path_generator_top.stamp_origin_increments_label = tkinter.Label(self.path_generator_top, text = "_____________________________\n\nOrigin rectangle search X Y increments\n_____________________________\n")
        self.path_generator_top.stamp_origin_increments_label.grid(row=6,column= 0,columnspan=2)
        self.path_generator_top.stamp_origin_increments_values_X_label = tkinter.Label(self.path_generator_top, text = "X", width = 10)
        self.path_generator_top.stamp_origin_increments_values_Y_label = tkinter.Label(self.path_generator_top, text = "Y", width = 10)
        self.path_generator_top.stamp_origin_increments_values_X_label.grid(row=7, column=0)
        self.path_generator_top.stamp_origin_increments_values_Y_label.grid(row=7, column=1)
        self.path_generator_top.origin_increment_x_entry = tkinter.Entry(self.path_generator_top, width = 10)
        self.path_generator_top.origin_increment_y_entry = tkinter.Entry(self.path_generator_top, width = 10)
        self.path_generator_top.origin_increment_x_entry.grid(row=8, column=0)
        self.path_generator_top.origin_increment_y_entry.grid(row=8, column=1)
        def callback_origin_increments():
            self.path_generator_top.entry_strings[0] = self.path_generator_top.origin_increment_x_entry.get()
            self.path_generator_top.entry_strings[1] = self.path_generator_top.origin_increment_y_entry.get()
            self.set_two_values(self.path_generator_top.entry_strings, self.origin_increment)

        self.path_generator_top.target_center_label = tkinter.Label(self.path_generator_top, text = "\nTarget search center coordinates\n_____________________________\n")
        self.path_generator_top.target_center_label.grid(row=3,column= 3,columnspan=2)
        self.path_generator_top.target_center_values_X_label = tkinter.Label(self.path_generator_top, text = "X", width = 10)
        self.path_generator_top.target_center_values_Y_label = tkinter.Label(self.path_generator_top, text = "Y", width = 10)
        self.path_generator_top.target_center_values_X_label.grid(row=4, column=3)
        self.path_generator_top.target_center_values_Y_label.grid(row=4, column=4)
        self.path_generator_top.target_center_x_entry = tkinter.Entry(self.path_generator_top, width = 10)
        self.path_generator_top.target_center_y_entry = tkinter.Entry(self.path_generator_top, width = 10)
        self.path_generator_top.target_center_x_entry.grid(row=5, column=3)
        self.path_generator_top.target_center_y_entry.grid(row=5, column=4)

        def callback_target_center():
            self.path_generator_top.entry_strings[0] = self.path_generator_top.target_center_x_entry.get()
            self.path_generator_top.entry_strings[1] = self.path_generator_top.target_center_y_entry.get()
            self.set_two_values(self.path_generator_top.entry_strings, self.target_center)
            
        self.path_generator_top.stamp_target_increments_label = tkinter.Label(self.path_generator_top, text = "_____________________________\n\nTarget rectangle search X Y increments\n_____________________________\n")
        self.path_generator_top.stamp_target_increments_label.grid(row=6,column= 3, columnspan=2)

        self.path_generator_top.stamp_target_increments_values_X_label = tkinter.Label(self.path_generator_top, text = "X", width = 10)
        self.path_generator_top.stamp_target_increments_values_Y_label = tkinter.Label(self.path_generator_top, text = "Y", width = 10)
        self.path_generator_top.stamp_target_increments_values_X_label.grid(row=7, column=3)
        self.path_generator_top.stamp_target_increments_values_Y_label.grid(row=7, column=4)

        self.path_generator_top.target_increment_x_entry = tkinter.Entry(self.path_generator_top, width = 10)
        self.path_generator_top.target_increment_y_entry = tkinter.Entry(self.path_generator_top, width = 10)
        self.path_generator_top.target_increment_x_entry.grid(row=8, column=3)
        self.path_generator_top.target_increment_y_entry.grid(row=8, column=4)
        def callback_target_increments():
            self.path_generator_top.entry_strings[0] = self.path_generator_top.target_increment_x_entry.get()
            self.path_generator_top.entry_strings[1] = self.path_generator_top.target_increment_y_entry.get()
            self.set_two_values(self.path_generator_top.entry_strings, self.target_increment)

        def callback_set_all():
            callback_origin_center()
            callback_origin_increments()
            callback_target_center()
            callback_target_increments()

        def callback_find_OT():
            self.path_generator_top.find_OT_button.config(text='RUNNING...')
            self.path_generator_top.find_OT_button.grid()
            self.path_generator_top.find_OT_button.update()
            callback_set_all()
            
            self.origin_selection_method = self.path_generator_top.origin_menu_var.get()
            self.target_selection_method = self.path_generator_top.target_menu_var.get()
            
            self.OT_nodes_not_found, self.origin_point_to_plot, self.target_point_to_plot, self.origin_point, self.target_point = find_OT(self.maps_list[numpy.shape(self.maps_list)[0]-1], self.node_array,self.x, self.y, self.origin_center[0], self.origin_center[1], self.origin_increment[0], self.origin_increment[1], self.target_center[0], self.target_center[1], self.target_increment[0], self.target_increment[1], self.origin_selection_method, self.target_selection_method, self.points_in_minima_groups)
            self.path_generator_top.find_OT_button.config(text='SET O&T')
            self.path_generator_top.find_OT_button.grid()
            self.path_not_found = 1
            check_status()
            if self.OT_nodes_not_found == 1:
                if self.origin_point == -1 and self.target_point == -1:
                    tkinter.messagebox.showwarning(
                        "SET O&T",
                        "No origin or target could be found with the arguments used\n")
                elif self.origin_point == -1:
                    tkinter.messagebox.showwarning(
                        "SET O&T",
                        "No origin could be found with the arguments used\n")
                elif self.target_point == -1:
                    tkinter.messagebox.showwarning(
                        "SET O&T",
                        "No target could be found with the arguments used\n")
                else:
                    tkinter.messagebox.showwarning(
                        "SET O&T",
                        "Target and origin seem to be the same\n")
            else:
                if (self.autoplot_string == "Auto-plot: On"):
                    callback_OT_surf_plot()

        def callback_gen_path(mode):
            if mode == "GLOBAL":
                if_node_by_node = 0
            elif mode == "NODE BY NODE":
                if_node_by_node = 1
        
            self.path_generator_top.gen_path_menu_var.set("RUNNING...")
            self.path_generator_top.gen_path_menu_button.grid()
            self.path_generator_top.gen_path_menu_button.update()
            if ((self.origin_selection_method == "ARBITRARY" or self.target_selection_method == "ARBITRARY") and if_node_by_node == 1):
                self.path_not_found = 1
                tkinter.messagebox.showwarning(
                    "ERROR",
                    "NODE BY NODE SAMPLING NOT SUPPORTED WITH ARBITRARY SELECTIONS\n")
            else:
                self.path_not_found, self.path_points, self.visited_points = find_path(self.maps_list[numpy.shape(self.maps_list)[0]-1], self.origin_point, self.target_point, self.node_array, self.points_in_minima_groups, self.node_matrix, self.is_4_axes_bool, self.x, self.y, if_node_by_node)
                callback_find_interest_points()
            self.path_generator_top.gen_path_menu_var.set("GENERATE PATH")
            self.path_generator_top.gen_path_menu_button.grid()
            check_status()
            if (self.path_not_found != 0):
                tkinter.messagebox.showwarning(
                    "ERROR",
                    "PATH FINDING FAILED\n")
            else:
                if (self.autoplot_string == "Auto-plot: On"):
                    callback_path_surf_plot()

        def callback_OT_surf_plot():
            self.path_generator_top.OT_surf_plot_button.config(text='PLOTTING...')
            self.path_generator_top.OT_surf_plot_button.grid()
            self.path_generator_top.OT_surf_plot_button.update()
            x_to_plot, y_to_plot, z_to_plot = generate_three_column_format(self.x, self.y, self.maps_list[numpy.shape(self.maps_list)[0]-1])
            surface_plot(self.x, self.y, x_to_plot, y_to_plot, z_to_plot)
            add_points_to_surface_plot(self.node_points_to_plot, self.origin_point_to_plot, self.target_point_to_plot, color = ["black", "#00ff00", "red"])
            add_points_to_surface_plot(self.origin_point_to_plot, self.target_point_to_plot, data_type = "annotate")
            self.path_generator_top.OT_surf_plot_button.config(text='O&T PLOT')
            self.path_generator_top.OT_surf_plot_button.grid()
            plt.show()

        def callback_full_surf_plot():
            self.path_generator_top.full_surf_plot_button.config(text='PLOTTING...')
            self.path_generator_top.full_surf_plot_button.grid()
            self.path_generator_top.full_surf_plot_button.update()
            x_to_plot, y_to_plot, z_to_plot = generate_three_column_format(self.x, self.y, self.maps_list[numpy.shape(self.maps_list)[0]-1])
            surface_plot(self.x, self.y, x_to_plot, y_to_plot, z_to_plot)
            add_points_to_surface_plot(self.visited_points, self.path_points, self.node_points_to_plot, self.interest_points, color = ["#00ff00", "black", "orange","magenta"])
            add_points_to_surface_plot(self.interest_points, data_type = "annotate")
            self.path_generator_top.full_surf_plot_button.config(text='FULL PLOT')
            self.path_generator_top.full_surf_plot_button.grid()
            plt.show()

        def callback_path_surf_plot():
            self.path_generator_top.path_surf_plot_button.config(text='PLOTTING...')
            self.path_generator_top.path_surf_plot_button.grid()
            self.path_generator_top.path_surf_plot_button.update()
            x_to_plot, y_to_plot, z_to_plot = generate_three_column_format(self.x, self.y, self.maps_list[numpy.shape(self.maps_list)[0]-1])
            surface_plot(self.x, self.y, x_to_plot, y_to_plot, z_to_plot)
            add_points_to_surface_plot(self.path_points)
            self.path_generator_top.path_surf_plot_button.config(text='PATH PLOT')
            self.path_generator_top.path_surf_plot_button.grid()
            plt.show()

        def callback_find_interest_points():
            temp_interest_points = numpy.empty(numpy.shape(self.path_points)[0], dtype = int);
            max_check = 0
            min_check = 0
            total_interest_points = 1
            temp_interest_points[0] = 0
            for counter in range(1,numpy.shape(self.path_points)[0]-1):
                max_check = 0
                min_check = 0
                if self.path_points[counter][2] < self.path_points[counter-1][2]:
                    min_check += 1
                elif self.path_points[counter][2] > self.path_points[counter-1][2]:
                    max_check += 1
                if self.path_points[counter][2] < self.path_points[counter+1][2]:
                    min_check += 1
                elif self.path_points[counter][2] > self.path_points[counter+1][2]:
                    max_check += 1
                if max_check != min_check:
                    temp_interest_points[total_interest_points] = counter
                    total_interest_points+=1
            temp_interest_points[total_interest_points] = numpy.shape(self.path_points)[0]-1
            total_interest_points+=1
            out_points_position = 0
            self.interest_points = numpy.empty((total_interest_points,3))
            for counter in range (total_interest_points):
                self.interest_points[out_points_position][0] = self.path_points[temp_interest_points[counter]][0]
                self.interest_points[out_points_position][1] = self.path_points[temp_interest_points[counter]][1]
                self.interest_points[out_points_position][2] = self.path_points[temp_interest_points[counter]][2]
                out_points_position += 1

        def callback_path_line_plot():
            self.path_generator_top.path_line_plot_button.config(text='PLOTTING...')
            self.path_generator_top.path_line_plot_button.grid()
            self.path_generator_top.path_line_plot_button.update()
            z_points = numpy.empty(numpy.shape(self.path_points)[0])
            for counter in range(numpy.shape(self.path_points)[0]):
                z_points[counter] = self.path_points[counter][2]
            line_plot(z_points, color = ["red",])
            self.path_generator_top.path_line_plot_button.config(text='ENERGY PLOT')
            self.path_generator_top.path_line_plot_button.grid()
            plt.show()

        def callback_save_path():
            file_to_save = tkinter.filedialog.asksaveasfile(mode='w', defaultextension=".txt")
            if file_to_save is None == 0:
                return
            for counter in range(numpy.shape(self.path_points)[0]):
                file_to_save.write(str(self.path_points[counter][0]) + " " + str(self.path_points[counter][1]) + " " + str(self.path_points[counter][2]) + "\n")
            file_to_save.close()

        def callback_save_interest_points():
            file_to_save = tkinter.filedialog.asksaveasfile(mode='w', defaultextension=".txt")
            if file_to_save is None == 0:
                return
            for counter in range(numpy.shape(self.interest_points)[0]):
                file_to_save.write(str(self.interest_points[counter][0]) + " " + str(self.interest_points[counter][1]) + " " + str(self.interest_points[counter][2]) + "\n")
            file_to_save.close()

        def callback_save_smooth():
            file_to_save = tkinter.filedialog.asksaveasfile(mode='w', defaultextension=".txt")
            if file_to_save is None == 0:
                return
            for counter in range(numpy.shape(self.smooth_path_points)[0]):
                file_to_save.write(str(self.smooth_path_points[counter][0]) + " " + str(self.smooth_path_points[counter][1]) + "\n")
            file_to_save.close()

        def callback_invert_path_values():
            for counter in range(numpy.shape(self.path_points)[0]):
                self.path_points[counter][2] = -self.path_points[counter][2]
            for counter in range(numpy.shape(self.interest_points)[0]):
                self.interest_points[counter][2] = -self.interest_points[counter][2]
            if (self.autoplot_string == "Auto-plot: On"):
                callback_path_line_plot()

        def callback_invert_path_order():
            temp_path_points = numpy.copy(self.path_points)
            order_counter = numpy.shape(self.path_points)[0]-1
            for counter in range(numpy.shape(self.path_points)[0]):
                self.path_points[counter][0] = temp_path_points[order_counter][0]
                self.path_points[counter][1] = temp_path_points[order_counter][1]
                self.path_points[counter][2] = temp_path_points[order_counter][2]
                order_counter -= 1
            if (self.autoplot_string == "Auto-plot: On"):
                callback_path_line_plot()

        def callback_smooth():
            iterations = 0
            iterations = int(self.path_generator_top.smooth_spin.get())
            if iterations > 0:
                self.path_generator_top.smooth_button.config(text='RUNNING...')
                self.path_generator_top.smooth_button.grid()
                self.path_generator_top.smooth_button.update()
                self.smooth_path_not_found, self.smooth_path_points = smooth_path(self.path_points, iterations)
                self.path_generator_top.smooth_button.config(text='SMOOTH')
                self.path_generator_top.smooth_button.grid()
                check_status()
            else:
                tkinter.messagebox.showwarning(
                    "ERROR",
                    "SMOOTH ITERATIONS VALUE HAS TO BE GREATER THAN 0\n")
            if (self.autoplot_string == "Auto-plot: On"):
                callback_smooth_plot()

        def callback_smooth_plot():
            self.path_generator_top.smooth_plot_button.config(text='PLOTTING...')
            self.path_generator_top.smooth_plot_button.grid()
            self.path_generator_top.smooth_plot_button.update()
            x_to_plot, y_to_plot, z_to_plot = generate_three_column_format(self.x, self.y, self.maps_list[numpy.shape(self.maps_list)[0]-1])
            surface_plot(self.x, self.y, x_to_plot, y_to_plot, z_to_plot)
            add_points_to_surface_plot(self.smooth_path_points)
            self.path_generator_top.smooth_plot_button.config(text='SMOOTH PLOT')
            self.path_generator_top.smooth_plot_button.grid()
            plt.show()

        def callback_add_to_stack():
            if self.empty_stack == 1:
                points_list = self.path_points.tolist()
                self.path_points_stack = [points_list,]
                self.empty_stack = 0
            else:
                points_list = self.path_points.tolist()
                self.path_points_stack.append(points_list)
            check_status()

        def callback_delete_last_from_stack():
            if len(self.path_points_stack) == 1:
                self.empty_stack = 1
            else:
                self.path_points_stack.pop()
            check_status()

        def callback_surface_stack_plot():
            self.path_generator_top.plot_surface_stack_button.config(text='PLOTTING...')
            self.path_generator_top.plot_surface_stack_button.grid()
            self.path_generator_top.plot_surface_stack_button.update()
            x_to_plot, y_to_plot, z_to_plot = generate_three_column_format(self.x, self.y, self.maps_list[numpy.shape(self.maps_list)[0]-1])
            surface_plot(self.x, self.y, x_to_plot, y_to_plot, z_to_plot)
            add_points_to_surface_plot(self.path_points_stack, color = ["multi",], data_type = "list")
            self.path_generator_top.plot_surface_stack_button.config(text='PATH STACK PLOT')
            self.path_generator_top.plot_surface_stack_button.grid()
            plt.show()

        def callback_line_stack_plot():
            self.path_generator_top.plot_line_stack_button.config(text='PLOTTING...')
            self.path_generator_top.plot_line_stack_button.grid()
            self.path_generator_top.plot_line_stack_button.update()
            line_plot(self.path_points_stack, color = ["multi",], data_type = "list")
            self.path_generator_top.plot_line_stack_button.config(text='ENERGY STACK PLOT')
            self.path_generator_top.plot_line_stack_button.grid()
            plt.show()

        def callback_save_stack():
            file_to_save = tkinter.filedialog.asksaveasfile(mode='w', defaultextension=".txt")
            if file_to_save is None == 0:
                return
            path_counter = 0
            for points_list in self.path_points_stack:
                temp_array = numpy.asarray(points_list)
                file_to_save.write("PATH_" + str(path_counter) + "\n")
                path_counter+=1
                for counter in range(numpy.shape(temp_array)[0]):
                    file_to_save.write(str(temp_array[counter][0]) + " " + str(temp_array[counter][1]) + " " + str(temp_array[counter][2]) + "\n")
            file_to_save.close()
            
##################################################
        def callback_calculate_well_sampling():
            self.path_generator_top.calculate_well_sampling_button.config(text='RUNNING...')
            self.path_generator_top.calculate_well_sampling_button.grid()
            self.path_generator_top.calculate_well_sampling_button.update()
            self.origin_sampling_points, self.target_sampling_points = well_sampling_analysis(self.origin_point, self.target_point, self.maps_list[numpy.shape(self.maps_list)[0]-1], self.node_array, self.node_matrix, self.is_4_axes_bool, self.x, self.y)
            total_points_temp = numpy.shape(self.origin_sampling_points)[0]
            print(str(total_points_temp) + " points were sampled to get from the origin node to the barrier of the target well")
            total_points_temp = numpy.shape(self.target_sampling_points)[0]
            print(str(total_points_temp) + " points were sampled to get from the target node to the barrier of the origin well")
            self.well_sampling_not_done = 0
            check_status()
            self.path_generator_top.calculate_well_sampling_button.config(text='CALCULATE WELL SAMPLING')
            self.path_generator_top.calculate_well_sampling_button.grid()
            self.path_generator_top.calculate_well_sampling_button.update()
            if (self.autoplot_string == "Auto-plot: On"):
                callback_well_sampling_plot()

        def callback_well_sampling_plot():
            self.path_generator_top.well_sampling_plot_button.config(text='PLOTTING...')
            self.path_generator_top.well_sampling_plot_button.grid()
            self.path_generator_top.well_sampling_plot_button.update()
            x_to_plot, y_to_plot, z_to_plot = generate_three_column_format(self.x, self.y, self.maps_list[numpy.shape(self.maps_list)[0]-1])
            surface_plot(self.x, self.y, x_to_plot, y_to_plot, z_to_plot)
            add_points_to_surface_plot(self.origin_sampling_points, self.target_sampling_points, color = ["#00ff00", "red"])
            self.path_generator_top.well_sampling_plot_button.config(text='WELL SAMPLING PLOT')
            self.path_generator_top.well_sampling_plot_button.grid()
            self.path_generator_top.well_sampling_plot_button.update()
            plt.show()

        def callback_save_well_sampling():
            x_to_plot, y_to_plot, z_to_plot = generate_three_column_format(self.x, self.y, self.maps_list[numpy.shape(self.maps_list)[0]-1])
            file_to_save = tkinter.filedialog.asksaveasfile(mode='w', defaultextension=".txt")
            if file_to_save is None == 0:
                return
            path_counter = 0
            file_to_save.write("TARGET_TO_ORIGIN\n")
            for counter in range(numpy.shape(self.origin_sampling_points)[0]):
                file_to_save.write(str(self.origin_sampling_points[counter][0]) + " " + str(self.origin_sampling_points[counter][1]) + " " + str(self.origin_sampling_points[counter][2]) + "\n")
            file_to_save.write("ORIGIN_TO_TARGET\n")
            for counter in range(numpy.shape(self.target_sampling_points)[0]):
                file_to_save.write(str(self.target_sampling_points[counter][0]) + " " + str(self.target_sampling_points[counter][1]) + " " + str(self.target_sampling_points[counter][2]) + "\n")
            file_to_save.close()

##################################################

        self.path_generator_top.find_OT_button = tkinter.Button(self.path_generator_top, text="SET O&T", command=callback_find_OT)
        self.path_generator_top.find_OT_button.grid(row=9,column=0,columnspan=2, pady=15)
        self.path_generator_top.OT_surf_plot_button = tkinter.Button(self.path_generator_top, text="O&T PLOT", command=callback_OT_surf_plot)
        self.path_generator_top.OT_surf_plot_button.grid(row=9,column=3,columnspan=2, pady=15)
        self.path_generator_top.gen_path_menu_var = tkinter.StringVar(self.path_generator_top)
        self.path_generator_top.gen_path_menu_var.set("GENERATE PATH")
        self.path_generator_top.gen_path_menu_button = tkinter.OptionMenu(self.path_generator_top, self.path_generator_top.gen_path_menu_var, "GLOBAL", "NODE BY NODE", command=callback_gen_path)
        self.path_generator_top.gen_path_menu_button.grid(row=10,column=0,pady=15)
        self.path_generator_top.full_surf_plot_button = tkinter.Button(self.path_generator_top, text="FULL PLOT", command=callback_full_surf_plot)
        self.path_generator_top.full_surf_plot_button.grid(row=10,column=1,pady=15)
        self.path_generator_top.path_surf_plot_button = tkinter.Button(self.path_generator_top, text="PATH PLOT", command=callback_path_surf_plot)
        self.path_generator_top.path_surf_plot_button.grid(row=10,column=2,pady=15)
        self.path_generator_top.path_line_plot_button = tkinter.Button(self.path_generator_top, text="ENERGY PLOT", command=callback_path_line_plot)
        self.path_generator_top.path_line_plot_button.grid(row=10,column=3,pady=15)
        self.path_generator_top.save_path_button = tkinter.Button(self.path_generator_top, text="SAVE PATH", command=callback_save_path)
        self.path_generator_top.save_path_button.grid(row=10,column=4,pady=15)
        self.path_generator_top.invert_path_values_button = tkinter.Button(self.path_generator_top, text="INVERT PATH VALUES", command=callback_invert_path_values)
        self.path_generator_top.invert_path_values_button.grid(row=11,column=1,pady=15)
        self.path_generator_top.invert_path_order_button = tkinter.Button(self.path_generator_top, text="INVERT PATH ORDER", command=callback_invert_path_order)
        self.path_generator_top.invert_path_order_button.grid(row=11,column=3,pady=15)
        self.path_generator_top.smooth_spin = tkinter.Spinbox(self.path_generator_top, from_ = 1, to = 100, width=5)
        self.path_generator_top.smooth_spin.grid(row=12,column=0,pady=15)
        self.path_generator_top.smooth_button = tkinter.Button(self.path_generator_top, text="SMOOTH", command=callback_smooth)
        self.path_generator_top.smooth_button.grid(row=12,column=1,pady=15)
        self.path_generator_top.smooth_plot_button = tkinter.Button(self.path_generator_top, text="SMOOTH PLOT", command=callback_smooth_plot)
        self.path_generator_top.smooth_plot_button.grid(row=12,column=2,pady=15)
        self.path_generator_top.save_smooth_button = tkinter.Button(self.path_generator_top, text="SAVE SMOOTH", command=callback_save_smooth)
        self.path_generator_top.save_smooth_button.grid(row=12,column=3,pady=15)
        self.path_generator_top.save_interest_points_button = tkinter.Button(self.path_generator_top, text="SAVE POIs", command=callback_save_interest_points)
        self.path_generator_top.save_interest_points_button.grid(row=12,column=4,pady=15)
        self.path_generator_top.path_add_to_stack_button = tkinter.Button(self.path_generator_top, text="ADD TO STACK", command=callback_add_to_stack)
        self.path_generator_top.path_add_to_stack_button.grid(row=13,column = 0,pady=15)
        self.path_generator_top.delete_last_from_stack_button = tkinter.Button(self.path_generator_top, text="REMOVE LAST", command=callback_delete_last_from_stack)
        self.path_generator_top.delete_last_from_stack_button.grid(row=13,column = 1,pady=15)
        self.path_generator_top.plot_surface_stack_button = tkinter.Button(self.path_generator_top, text="PATH STACK PLOT", command=callback_surface_stack_plot)
        self.path_generator_top.plot_surface_stack_button.grid(row=13,column = 2,pady=15)
        self.path_generator_top.plot_line_stack_button = tkinter.Button(self.path_generator_top, text="ENERGY STACK PLOT", command=callback_line_stack_plot)
        self.path_generator_top.plot_line_stack_button.grid(row=13,column = 3,pady=15)
        self.path_generator_top.save_stack_button = tkinter.Button(self.path_generator_top, text="SAVE STACK", command=callback_save_stack)
        self.path_generator_top.save_stack_button.grid(row=13,column = 4,pady=15)
        self.path_generator_top.calculate_well_sampling_button = tkinter.Button(self.path_generator_top, text="CALCULATE WELL SAMPLING", command=callback_calculate_well_sampling)
        self.path_generator_top.calculate_well_sampling_button.grid(row=14,column = 1,pady=15)
        self.path_generator_top.well_sampling_plot_button = tkinter.Button(self.path_generator_top, text="WELL SAMPLING PLOT", command=callback_well_sampling_plot)
        self.path_generator_top.well_sampling_plot_button.grid(row=14,column = 2,pady=15)
        self.path_generator_top.save_well_sampling_button = tkinter.Button(self.path_generator_top, text="SAVE WELL SAMPLING", command=callback_save_well_sampling)
        self.path_generator_top.save_well_sampling_button.grid(row=14,column = 3,pady=15)
        
        ####end of widgets
        self.unload_button.config(state ='disabled')
        self.unload_button.pack(self.button_opt)
        self.path_generator_button.config(state ='disabled')
        self.path_generator_button.pack(self.button_opt)
        check_status()
        self.update_idletasks()
        
    def path_generator_exit_handler(self):
        self.path_generator_is_open = False
        if self.map_editor_is_open == False:
            self.unload_button.config(state ='normal')
            self.unload_button.pack(self.button_opt)
        self.path_generator_button.config(state ='normal')
        self.path_generator_button.pack(self.button_opt)
        self.path_generator_top.increment_y_entry = tkinter.Entry(self.path_generator_top)
        self.update_idletasks()
        self.path_generator_top.destroy()
        
        self.path_generator_top.update_idletasks()

########################################
####################### Main loop

if __name__=='__main__':
    root = tkinter.Tk()
    TkMainDialog(root).pack()
    root.mainloop()
