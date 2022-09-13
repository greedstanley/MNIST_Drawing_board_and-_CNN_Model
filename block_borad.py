import cv2
import numpy as np
import mediapipe as mp
import keyboard
import time
# load CNN model 
import tensorflow as tf
CNN_model = tf.keras.models.load_model('./mnist_dect_CNN_model.h5')

# select camera
cap = cv2.VideoCapture(0)
# initial mediapipe function
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mpDraw = mp.solutions.drawing_utils

# created black board
board_x_init, board_y_init = 300, 50
board_height, board_width = 200, 300
board = None

# pen status
pen_sta=0
pen_x_pre, pen_y_pre = 0, 0

# label detect
labels_detect = {1:"Answer" ,99:"None"}
label = 99
numbers = ''


#%% number of island
def numsIsland(board, board_height, board_width):
    count = 0
    n = board_height
    m = board_width
    dic = {}
    for i in range(n):
        for j in range(m):
            if board[i][j]>0:
                count+=1
                y_list, x_list = DFS(board, i, j, n, m, [], [])
                dic.update({str(count):[y_list, x_list]})      
    return dic
    

#%% DFS
def DFS(board, y, x, board_height, board_width, y_list, x_list):
    if x < 0 or x >= board_width or y < 0 or y >= board_height or board[y][x]==0:
        return
    
    y_list.append(y)
    x_list.append(x)
    
    
    board[y][x] = 0
    DFS(board, y+1, x, board_height, board_width, y_list, x_list)
    DFS(board, y-1, x, board_height, board_width, y_list, x_list)
    DFS(board, y, x+1, board_height, board_width, y_list, x_list)
    DFS(board, y, x-1, board_height, board_width, y_list, x_list)
    
    return y_list,x_list
    
#%% Perdict numbers
def predictNum(gary_frame, board_height, board_width):
    num = ''    
    
    dic = numsIsland(gary_frame.copy(), board_height, board_width)
    
    for i in range(len(dic.keys())):
        # Created new canvas
        y_height = (max(dic[str(i+1)][0]) - min(dic[str(i+1)][0]))
        canvas = np.zeros( (int(y_height/(0.7)),int(y_height/(0.7)) ), np.uint8)
        # canvas_x_center = can_x_cen , canvas_y_center = can_y_cen
        can_x_cen, can_y_cen = (canvas.shape[1])//2, (canvas.shape[0])//2
        # cut the number of image
        x_max = max(dic[str(i+1)][1])
        x_min = min(dic[str(i+1)][1])
        y_max = max(dic[str(i+1)][0])
        y_min = min(dic[str(i+1)][0])
        cut_num = gary_frame[y_min:y_max,x_min:x_max]
        
        cv2.imshow('cut_image'+str(i),cut_num)
        x_delta = cut_num.shape[1]//2
        y_delta = cut_num.shape[0]//2
        canvas[can_y_cen-y_delta:can_y_cen-y_delta+cut_num.shape[0],
               can_x_cen-x_delta:can_x_cen-x_delta+cut_num.shape[1]] = cut_num
        
        cv2.imshow("img"+str(i),canvas)
        
        # predict numbers
        canvas = cv2.resize(canvas, (28,28), fx=1, fy=1)
        canvas_shape = canvas.reshape(1,28,28,1)
        pred=CNN_model.predict(canvas_shape)
        num += str(int(np.argmax(pred,axis=1)))
        cut_num[:]=0
        
    print(num)
    
    return num
 
#%% function of pen
def findpen(handslandmark, f_height, f_width, board_x_init, board_y_init, board_height, board_width):
    global pen_x_pre, pen_y_pre
    
    for i, lm in enumerate(handslandmark):
        if i==8:
            pen_x = int(lm.x*f_width)
            pen_y = int(lm.y*f_height)
            cv2.circle(copy_frame, (pen_x, pen_y), 8, [255, 255, 0], cv2.FILLED)
            
        else:
            continue
        
    if pen_x_pre==0 and pen_y_pre==0:
        pen_x_pre, pen_y_pre = pen_x, pen_y
        
    if board_x_init<=pen_x and pen_x<=(board_x_init+board_width) and board_y_init<=pen_y and pen_y<=(board_y_init+board_height) and pen_sta==1:
        cv2.line(board, (pen_x_pre-board_x_init, pen_y_pre-board_y_init),
                 (pen_x-board_x_init, pen_y-board_y_init), 
                 [255, 255, 255], 8)
    pen_x_pre, pen_y_pre = pen_x, pen_y


      
#%% Camera action
while True:
    
    ret, frame = cap.read()

    # initialize the blackboard
    if board is None:
        board = np.zeros((board_height, board_width, 3), np.uint8)
    
    if ret==True:
    
        frame = cv2.resize(frame, (0,0), fx=1, fy=1)
        frame = cv2.flip(frame, 1)
        copy_frame = frame.copy()
        
        # frame Height and Width
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
        
        # dect hands
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hands_result = hands.process(frameRGB)
        
        # draw hands land marks (points)
        if hands_result.multi_hand_landmarks:
            for hands_land_mark in hands_result.multi_hand_landmarks:
                #mpDraw.draw_landmarks(copy_frame, hands_land_mark, mp_hands.HAND_CONNECTIONS)
                findpen(hands_land_mark.landmark, frameHeight, 
                        frameWidth,board_x_init, board_y_init,
                        board_height, board_width)
        
        # Show board frame
        cv2.rectangle(copy_frame, (board_x_init, board_y_init),
                      (board_x_init+board_width, board_y_init+board_height),
                      (0, 0, 255), 4)
        cv2.putText(copy_frame, labels_detect[label]+numbers,
                    (400, 290), cv2.FONT_HERSHEY_SIMPLEX,
                    1 ,[255,255,100] ,2)
        
        # Smooth drawing on frame
        _ , draw_mask = cv2.threshold(cv2.cvtColor(board, cv2.COLOR_BGR2GRAY),
                                      20, 255, cv2.THRESH_BINARY)
        foreground = cv2.bitwise_and(board,board,mask = draw_mask)
        background = copy_frame[board_y_init:board_y_init+board_height,
                                board_x_init:board_x_init+board_width]
        background = cv2.bitwise_and(background,background,
                                     mask = cv2.bitwise_not(draw_mask))
        test_frame = cv2.add(foreground,background)
        copy_frame[board_y_init:board_y_init+board_height,
                   board_x_init:board_x_init+board_width] = test_frame
        
    # Show frame
    cv2.imshow("black board", board)
    cv2.imshow("draw_frame", copy_frame)

    
    # Predict digit number
    if keyboard.is_pressed('p'):
        time.sleep(0.2)
        gary_frame = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
        numbers = predictNum(gary_frame, board_height, board_width)
        label = 1

        
    # Clear blackboard
    if keyboard.is_pressed('enter'):
        time.sleep(0.2)
        print('Clear black board ')
        numbers = ''
        board = None
        label = 99
        

    # switch pen status
    if keyboard.is_pressed('space'):
        time.sleep(0.2)
        if pen_sta == 0:
            pen_sta=1
        else:
            pen_sta=0
    
    # Took a picture
    if keyboard.is_pressed('c'):
        file_name = labels_detect[label]+numbers+'.png'
        cv2.imwrite(file_name,copy_frame)
    
    
    # Quit
    if cv2.waitKey(5)== ord('q'):
        break
cv2.destroyAllWindows()
cap.release()

