import numpy as np
import mediapipe as mp
import cv2

hands = mp.solutions.hands
hand_landmark = hands.Hands(max_num_hands=1)
frame_shape = (720, 1280, 3)
blackboard_shape = (720, 1280, 3)
blackboard = np.zeros(blackboard_shape, dtype='uint8')
mask = np.zeros(blackboard_shape, dtype='uint8')
colour = (255, 0, 255)
thickness = 4
curr_tool = 'draw'
start_point = None
draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)
prevxy = None

tools = cv2.imread("tool.png")
tools = tools.astype('uint8')
print(tools.shape)

# Row and Column for toolbar
midCol = 1280 // 2
max_row = 50
min_col = midCol - 125
max_col = midCol + 125

# Define color palette
colors = [
    (0, 0, 255),    # Red
    (0, 255, 0),    # Green
    (255, 0, 0),    # Blue
    (0, 255, 255),  # Yellow
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (255, 255, 255),# White
    (128, 128, 128),# Gray
    (0, 165, 255)   # Orange
]

# Check if distance between 2 points is less than 60 pixels
def get_is_clicked(point1, point2):
    (x1, y1) = point1
    (x2, y2) = point2
    dis = (x1 - x2) ** 2 + (y1 - y2) ** 2
    dis = np.sqrt(dis)
    return dis < 60

# Return tool based on column location
def get_Tool(point, prev_tool):
    (x, y) = point
    if min_col < x < max_col and y < max_row:
        if x < 50 + min_col:
            return "line"
        elif x < 100 + min_col:
            return "rectangle"
        elif x < 150 + min_col:
            return "draw"
        elif x < 200 + min_col:
            return "circle"
        else:
            return "erase"
    return prev_tool

# Return color based on row location
def get_Color(point):
    (x, y) = point
    if y < max_row and x >= max_col:
        color_index = (x - max_col) // 50
        if color_index < len(colors):
            return colors[color_index]
    return None

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    
    # Ensure frame has the same shape as `frame_shape`
    if frame.shape[:2] != frame_shape[:2]:
        frame = cv2.resize(frame, (frame_shape[1], frame_shape[0]))

    # Preprocess Image
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    op = hand_landmark.process(rgb)
    
    # Clear the blackboard before drawing new hand landmarks
    blackboard[:] = 0

    # Draw the live video feed in a small box
    live_feed_small = cv2.resize(frame, (320, 240))
    blackboard[0:240, 0:320] = live_feed_small

    # Draw hand landmarks on the blackboard and handle drawing tools
    if op.multi_hand_landmarks:
        for all_landmarks in op.multi_hand_landmarks:
            draw.draw_landmarks(blackboard, all_landmarks, hands.HAND_CONNECTIONS)

            # Index finger location
            x = int(all_landmarks.landmark[8].x * blackboard_shape[1])
            y = int(all_landmarks.landmark[8].y * blackboard_shape[0])
            
            # Middle finger location
            middle_x = int(all_landmarks.landmark[12].x * blackboard_shape[1])
            middle_y = int(all_landmarks.landmark[12].y * blackboard_shape[0])
            
            is_clicked = get_is_clicked((x, y), (middle_x, middle_y))
            new_tool = get_Tool((x, y), curr_tool)
            new_color = get_Color((x, y))
            
            if new_color:
                colour = new_color
            else:
                curr_tool = new_tool

            # Select tool and draw for that
            if curr_tool == 'draw':
                if is_clicked and prevxy is not None:
                    cv2.line(mask, prevxy, (x, y), colour, thickness)

            elif curr_tool == 'line':
                if is_clicked and start_point is None:
                    start_point = (x, y)
                elif is_clicked:
                    cv2.line(blackboard, start_point, (x, y), colour, thickness)
                elif not is_clicked and start_point:
                    cv2.line(mask, start_point, (x, y), colour, thickness)
                    start_point = None

            elif curr_tool == 'rectangle':
                if is_clicked and start_point is None:
                    start_point = (x, y)
                elif is_clicked:
                    cv2.rectangle(blackboard, start_point, (x, y), colour, thickness)
                elif not is_clicked and start_point:
                    cv2.rectangle(mask, start_point, (x, y), colour, thickness)
                    start_point = None
            
            elif curr_tool == 'circle':
                if is_clicked and start_point is None:
                    start_point = (x, y)
                
                if start_point:
                    rad = int(((start_point[0] - x) ** 2 + (start_point[1] - y) ** 2) ** 0.5)
                if is_clicked:
                    cv2.circle(blackboard, start_point, rad, colour, thickness)
                
                if not is_clicked and start_point:
                    cv2.circle(mask, start_point, rad, colour, thickness)
                    start_point = None
            
            elif curr_tool == "erase":
                cv2.circle(blackboard, (x, y), 30, (0, 0, 0), -1)  # -1 means fill
                if is_clicked:
                    cv2.circle(mask, (x, y), 30, 0, -1)
            prevxy = (x, y)    

    # Merge Frame and Mask
    blackboard = np.where(mask != 0, mask, blackboard)
    
    blackboard[0:max_row, min_col:max_col] = tools

    # Draw color palette
    for i, color in enumerate(colors):
        palette_x = max_col + i * 50
        cv2.rectangle(blackboard, (palette_x, 0), (palette_x + 50, max_row), color, -1)

    cv2.imshow('Blackboard', blackboard)
    if cv2.waitKey(1) == 27:
        break
  
cap.release()
cv2.destroyAllWindows()
