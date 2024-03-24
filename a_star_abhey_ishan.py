#Import important libraries 
import time
import cv2
import heapq as hq
import numpy as np
import matplotlib.pyplot as plt
import copy
import math


def obstacle_map(canvas):

    #Creating rectangle 1
    cv2.rectangle(canvas,pt1=(100,500),pt2=(175,100),color=(179,41,43),thickness=-1)

    #Creating rectangle 2
    cv2.rectangle(canvas,pt1=(275,400),pt2=(350,0),color=(179,41,43),thickness=-1)

    # Draw hexagon
    cv2.fillPoly(canvas, [np.array([(650, 400), (775, 325), (775, 175), (650, 100), (525, 175), (525, 325)])], color=(179,41,43))

    # Draw polygon

    cv2.fillPoly(canvas, [np.array([(900, 50), (900, 125), (1020, 125), (1020, 375), (900, 375), (900, 450), (1100, 450), (1100, 50)])], color=(179,41,43))

    return canvas


def input_coordinates():

    #check if start and goal nodes are valid
    while True:
        start_node_str = input("Enter the coordinates of starting node (x,y,theta):")
        goal_node_str = input("Enter the coordinates of Goal node (x,y,theta):")
        
        start_node = tuple(map(int, start_node_str.split(',')))
        goal_node = tuple(map(int, goal_node_str.split(',')))


        #Check if the start and goal node are valid
        if is_valid(start_node[0],start_node[1]):

            if is_valid(goal_node[0],goal_node[1]):
                break
            else:
                print("Invalid goal node. Please enter valid coordinates.")
                continue
        else:
            print("Invalid start node. Please enter valid coordinates.")
            continue

    return start_node,goal_node

def input_step():
    while True:
        step_size = float(input("Enter the step size:"))

        if step_size>=1 and step_size<=10:
            break
        else:
            print("Invalid step size.")
    
    return step_size


#checking if the robot is at a valid positon
def is_valid(x,y):

    #check if the coordinates are in bounds of canvas
    if (0 <= x <= width and 0 <= y <= height):
        pass
    else:
        return False

    #Here (x-5) and (y-5) is used to account for 5 mm clearance
    #check if the coordinates are within bounds of rectangle 1
    if ((100-5) <= x <= (175+5)) and (0 <= y <= (400+5)):
        return False
       
    #check if the coordinates are within bounds of rectangle 2
    if ((275-5) <= x <= (350+5)) and ((100-5) <= y <= (500+5)):
        return False

    #Here (x-3.53) and (y-3.53) is used to account for 5 mm clearance in diagonal dirn
    #check if the coordinates are within bounds of hexagon
    if (((y+3.53)+(3/5)*(x+3.53)-490 >=0) and ((y+3.53)-(3/5)*(x-3.53)+290>=0) and (y)<=175) or ((525-5) <= (x) <= (775+5) and (175 <= (y) <= 325)) or (((y-3.53)-(3/5)*(x+3.53)-10<=0) and ((y-3.53)+(3/5)*(x-3.53) - 790 <=0) and (y)>=325):
        return False

    #check if the coordinates are within bounds of polygon
    if ((900-5) <= (x) <= (1100+5) and (50-5) <= (y) <= (125+5)) or ((1020-5) <= (x) <= (1100+5) and 125 <= (y) <= 375) or ((900-5) <= (x) <= (1100+5) and (375-5) <= (y) <= (450+5)):
        return False
    
    return True


def action(step_size,current_theta):

    current_theta = math.radians(current_theta)

    movements=[]

    # 5 actions
    for angle in range(-60,90,30):
        x = step_size*math.cos(current_theta + math.radians(angle))
        y = step_size*math.sin(current_theta + math.radians(angle))

        movements.append((x,y,angle))

    return movements

# moving nodes
def move_node(present_node,move):

    next_node = []

    #x, y, theta coordinates of next_node
    new_x = present_node[0] + move[0]
    new_y = present_node[1] + move[1]
    new_theta = present_node[2] + move[2]

    #rounding x,y to upto 2 decimal places
    new_x = round(new_x,1)
    new_y = round(new_y,1)

    # keeping theta in 0 to 360 range
    new_theta = new_theta - (new_theta//360)*360

    if is_valid(new_x, new_y):
        next_node = (new_x, new_y,new_theta)
        return next_node
    
    else:
        return None



def heuristic_cost(current_pos,goal_pos):

    cost = math.sqrt((goal_pos[1]-current_pos[1])**2+(goal_pos[0]-current_pos[0])**2) 
    return round(cost,1)

def round_node(node):
    x = round(node[0] * 2) / 2
    y = round(node[1] * 2) / 2
    theta = node[2]
    return x,y,theta

def in_orientation_threshold(current_orientation,goal_orientation,threshold):
    if current_orientation > 180:
        current_orientation = 360 - current_orientation
        delta  = current_orientation + goal_orientation

    else:
        delta = abs(current_orientation - goal_orientation)


    if delta <=threshold:
        return True
    else:
        return False 



#backtracking 
def get_path(start_position, goal_position,closed_list):
    
    path = []
    current_node = goal_position
    
    # Backtrack from goal node to start node
    while current_node != start_position:
        path.append(current_node)
        current_node = tuple(closed_list[current_node])
    
    # Add the start node to the path
    path.append(start_position)
    
    # Reverse the path to get it in the correct order (from start to goal)
    path.reverse()
    
    return path,closed_list


def visualization(path, closed_list, canvas, start_position, goal_position, frame_skip=50):
    output_video = cv2.VideoWriter('visualization.mp4', cv2.VideoWriter_fourcc(*'XVID'), 1000, (canvas.shape[1], canvas.shape[0]))
    skip_counter = 0

    # Draw start and goal node
    cv2.circle(canvas, start_position[:2], 5, (0, 255, 0), -1)
    cv2.circle(canvas, goal_position[:2], 5, (0, 0, 255), -1)

    # Draw visited nodes
    for visited_node in closed_list:
        canvas[int(visited_node[1])-1][int(visited_node[0])-1] = [57, 131, 196]

        skip_counter += 1
        if skip_counter == frame_skip:
            vid = cv2.flip(canvas, 0) 
            output_video.write(vid)
            skip_counter = 0
    
    # Draw optimal Path
    optimal_path = copy.deepcopy(path)
    optimal_path.reverse()
    for _ in range(len(optimal_path)):
        node = optimal_path.pop()
        canvas[int(node[1])-1][int(node[0])-1] = [0, 0, 255]       

        skip_counter += 1
        if skip_counter == frame_skip:
            vid = cv2.flip(canvas, 0)
            output_video.write(vid)
            skip_counter = 0

    output_video.release()


#A star algorithm
def a_star(start_position, goal_position, canvas,step_size, goal_threshold_distance=1.5,goal_threshold_angle=30):

    # List of nodes to be explored
    open_list = []
    
    # Dictionary stores explored and its parent node
    closed_list = {}

    # Dictionary to store node information {present_node: [parent_node, cost_to_come]}
    node_info = {}

    # visited  nodes (as nearest 0.5 multiple)
    visited_nodes = np.zeros((1000, 2400, 12), dtype=int)

    # heap to store the nodes based on their cost value
    hq.heapify(open_list)

    # Inserting the initial node with its [total_cost, present_node]
    hq.heappush(open_list, [ 0+heuristic_cost(start_position,goal_position), start_position])

    # Set the node_info for the start position
    node_info[start_position] = [None, 0]

    # visited_set.add(round_node(start_position))
    index = round_node(start_position)
    visited_nodes[int(index[0]*2)][int(index[1]*2)][int(index[2]/30)] = 1

    #while open list is not empty
    while open_list:

        # total_cost,cost2come,parent_node, present_node = hq.heappop(open_list)

        total_cost, present_node = hq.heappop(open_list)
        parent_node, cost2come = node_info[present_node]

        # Adding the present node to closed list with its parent node - {present_node:parent_node}

        # rounded_closed_node = round_node(present_node)
        # closed_list[(present_node[0], present_node[1],present_node[2])] = parent_node

        closed_list[present_node] = parent_node
        # closed_list[rounded_closed_node[0], rounded_closed_node[1], rounded_closed_node[2]] = parent_node

        # closed_set.add(round_node(present_node))
  

        # print(present_node)
        cost2goal = total_cost-cost2come
        #if goal reached
        # if list(present_node) == list(goal_position):
        # if heuristic_cost(present_node, goal_position) <= goal_threshold_distance:
        if cost2goal <= goal_threshold_distance:

            if in_orientation_threshold(present_node[2],goal_position[2],goal_threshold_angle) :

            # closed_list[(goal_position[0], goal_position[1],goal_position[2])] = present_node
            
                closed_list[goal_position] = present_node
                print("goal reached")
                return get_path(start_position, goal_position,closed_list)
        
        #Add neighbouring nodes to the open_list
        for direction in action(step_size,present_node[2]):

            next_node = move_node(present_node,direction)

            if next_node is not None:

                rounded_next_node = round_node(next_node)

                scaled_x = int(rounded_next_node[0] * 2)
                scaled_y = int(rounded_next_node[1] * 2)
                scaled_theta_index = int(rounded_next_node[2] / 30)

                # if (rounded_next_node not in closed_set):
                if np.sum(visited_nodes[scaled_x,scaled_y,:]) < 5:

                    # if (rounded_next_node not in visited_set):
                    if (visited_nodes[scaled_x,scaled_y,scaled_theta_index] == 0):

                        new_cost2come = cost2come + step_size
                        new_total_cost = new_cost2come + heuristic_cost(next_node, goal_position)

                        # hq.heappush(open_list, [cost2come+step_size+heuristic_cost(next_node,goal_position),cost2come + step_size, present_node, list(next_node)])
                        # hq.heapify(open_list)

                        hq.heappush(open_list, [new_total_cost, next_node])
                        node_info[next_node] = [present_node, new_cost2come]

                        # visited_set.add(rounded_next_node)
                        visited_nodes[scaled_x,scaled_y,scaled_theta_index] = 1
                
                    # if node is already in open list we need to compare cost and update if needed
                    else:

                        if (next_node in node_info) and (cost2come + step_size) < node_info[next_node][1]:

                            new_cost2come = cost2come + step_size
                            new_total_cost = new_cost2come + heuristic_cost(next_node, goal_position)
                            hq.heappush(open_list, [new_total_cost, next_node])
                            node_info[next_node] = [present_node, new_cost2come]        

    return "Solution does not exist"

                        

if __name__=="__main__":

    start_time = time.time() 

    # create blank  canvas
    width = 1200
    height = 500
    canvas = np.ones((height,width,3), dtype=np.uint8) * 255

    # draw the obstacle map
    canvas = obstacle_map(canvas)

    # input start and goal node coordinates
    start_position,goal_position = input_coordinates()

    #Step size
    step_size = input_step()

    # A*
    path,closed_list = a_star(start_position, goal_position, canvas, step_size)

    #goal reached
    goal_reached_time = time.time()
    print("Total time taken to reach the goal: ",goal_reached_time-start_time)

    # Display Node exploration and Optimal path
    visualization(path,closed_list,canvas,start_position,goal_position)

    end_time = time.time()

    print("Total time taken to execute the code: ",end_time-start_time) 
    print(f"closed_list : {closed_list}")
    print(f"path : {path}")
    