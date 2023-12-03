import tkinter as tk
from tkinter.font import Font
import cv2
import numpy as np
import time
import threading
import multiprocessing
import psutil
import os
from PIL import Image, ImageTk
import sys
y1=0
#redirecting the standard output to the file 'Final 612 Project Log.txt'
with open('Final 612 Project Log.txt', 'w') as f:
    sys.stdout = f

        #function to set button styles
    def set_button_style(button):
        button.config(bg="#4CAF50", fg="white", font=button_font, bd=0, padx=10, pady=5, activebackground="#3E8E41", highlightthickness=2, highlightcolor="#2E7D32")

    def perform_lane_detection():
        #thread for lane detection
        lane_thread = threading.Thread(target=lane_detection)
        #CODE TO EXPLICITLY SPECIFY THE CPU CORE TO RUN ON (IF NOT DONE PYTHON BY DEFAULT USES ALL POSSIBLE CORES)
        psutil.Process().cpu_affinity([])
        lane_thread.start()
        lane_thread.join()
        print("Thread nameeeeeeeeeeeeeeeeeeee:", lane_thread.name)
        print("Thread ID:", lane_thread.ident)
        #psutil.Process(lane_thread.ident).cpu_affinity([])
        print("Thread status:", lane_thread.is_alive())

    def perform_pedestrian_detection():
        #psutil.Process().cpu_affinity([0,9,12,13])
        psutil.Process().cpu_affinity([])
        #thread for pedestrian_detection
        pedestrain_thread = threading.Thread(target=pedestrian_detection)
        pedestrain_thread.start()
        pedestrain_thread.join()
        print("Thread name:", pedestrain_thread.name)
        print("Thread ID:", pedestrain_thread.ident)
        print("Thread status:", pedestrain_thread.is_alive())

    def perform_stop_sign_detection(): 
        #thread for stop_sign_detection
        sign_thread = threading.Thread(target=stop_sign_detection)
        sign_thread.start()
        sign_thread.join()
        print("Thread name:", sign_thread.name)
        print("Thread ID:", sign_thread.ident)
        print("Thread status:", sign_thread.is_alive())

    def perform_car_detection():
        #thread for stop_sign_detection
        car_thread = threading.Thread(target=car_detection)
        psutil.Process().cpu_affinity([0])
        car_thread.start()
        car_thread.join()
        print("Thread name:", car_thread.name)
        print("Thread ID:", car_thread.ident)
        print("Thread status:", car_thread.is_alive())

    #function for lane detection
    def lane_detection():
        thread = threading.current_thread()
        print("Thread name:", thread.name)
        print("Thread ID:", thread.ident)
        print("Thread status:", thread.is_alive())
        p = psutil.Process(os.getpid())
        print(f"Lane Detection thread is running on CPU core {p.cpu_affinity()}")

        p = psutil.Process()
        x=0
        #Get the threads associated with the process
        threads = p.threads()   

        for thread in threads:
            print(thread)
        #cap for cali vid short1
        #cap = cv2.VideoCapture("cali_vid_short1.mp4")
        #cap for drive.mp4
        cap = cv2.VideoCapture("drive.mp4")
        max_line_Gap = 50
        max_line_angle = np.pi / 180
        stop_flag = False
        while not stop_flag:
            
            ret, frame = cap.read()
            #time.sleep(0.2)
            key = cv2.waitKey(1)
            cv2.namedWindow("Lane Departure Warning System")
            while True:
                ret, frame = cap.read()
                if ret:
                    t = cv2.getTickCount()
                    width, height = frame.shape[1], frame.shape[0]

                    
                    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    #cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
                    #for (x, y, w, h) in cars:
                    #    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    
                    
                    
                    #CUDA FOR GPU
                    #hsv = cv2.cuda.cvtColor(cv2.cuda_GpuMat(frame), cv2.COLOR_BGR2HSV)
                    
                    
                    
                    
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    lower = np.uint8([180, 18, 255])
                    upper = np.uint8([0, 0, 231])

                    mask = cv2.inRange(hsv, lower, upper)
                    edges = cv2.Canny(mask, 75, 250)
                    
                    lines = cv2.HoughLinesP(edges, 1, max_line_angle, 50, max_line_Gap)

                    if lines is not None:
                        for i in range(0, len(lines)):
                            l = lines[i][0]
                            cv2.line(frame, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)
                    #change perspective
                    #perspective points for DRIVE.mp4
                    pts1 = np.float32([[560, 440],[710, 440],[200, 640],[1000, 640]])
                    pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])
                    #perspective points for CALI_VID_SHORT.mp4
                    #pts1 = np.float32([[560, 400],[710, 400],[200, 500],[1000, 500]])
                    #pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])
                    #perspective points for Self-driving.AVI
                    #pts1 = np.float32([[560, 400],[710, 400],[200, 500],[1000, 500]])
                    #pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])


                    #change perspective 
                    matrix = cv2.getPerspectiveTransform(pts1,pts2)
                    birds_eye = cv2.warpPerspective(frame, matrix, (width, height))
                    #end change birds eye perspective
                    grayscale = cv2.cvtColor(birds_eye, cv2.COLOR_BGR2GRAY)
                    #smooth image
                    kernel_size = 5
                    blur = cv2.GaussianBlur(grayscale, (kernel_size, kernel_size), 0)
                    #canny edge dedection
                    low_t = 50
                    high_t = 95
                    edges = cv2.Canny(blur, low_t, high_t)
                    #detect line with houg line transform
                    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, maxLineGap=50)
                    #draaw lines
                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        #lassification right and left line
                        if x1 < 640 or x2 < 640:
                            x1_left = x1
                            x2_left = x2
                            y1_left = y1
                            y2_left = y2
                        
                        elif x1 > 640 or x2 > 640:
                            x1_right = x1
                            x2_right = x2
                            y1_right = y1
                            y2_right = y2
                        try:
                            #calculate middle points
                            x1_mid = int((x1_right + x1_left)/2)
                            x2_mid = int((x2_right + x2_left)/2)
                        
                            # y1_mid = int((y1_right + y1_left)/2)
                            # y2_mid = int((y2_right + y2_left)/2)
                            #UNCOMMENT THIS
                            cv2.line(birds_eye, (640, 300), (x2_mid, 420), (0, 255, 0), 2)
                            
                            #create straight pipe line in middle of the frame
                            x_1, x_2 = 640, 640
                            y_1, y_2 = 300, 420
                            cv2.line(birds_eye, (x_1,y_1), (x_2, y_2), (0, 0, 255), 2)
                            #calculate 3 point beetween angle
                            point_1 = [x_1, y_1]
                            point_2 = [x_2, y_2]
                            point_3 = [x2_mid, 420]
                            
                            radian = np.arctan2(point_2[1] - point_1[1], point_2[0] - point_1[0]) - np.arctan2(point_3[1] - point_1[1], point_3[0] - point_1[0])
                            angle = (radian *180 / np.pi)
                            print("Angle : ", angle)
                            # cv2.putText(frame, str(int(angle)), (15,35),cv2.FONT_HERSHEY_SIMPLEX,1 , (255,0,0), 2, cv2.LINE_AA )
                            cv2.putText(frame, "LANE DEPARTURE WARNING SYSTEM ", (15,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,155,255), 3, cv2.LINE_AA )
                            cv2.putText(frame, "ARROWS SHOW DIRECTION TO TURN TOWARDS TO AVOID LANE DISPLACEMENT ", (15,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,155,255), 3, cv2.LINE_AA )

                            if angle < -30:
                                cv2.putText(frame, "<<", (120,250), cv2.FONT_HERSHEY_SIMPLEX, 4, (0,0,255), 3, cv2.LINE_AA )

                            elif angle > 25:
                                cv2.putText(frame, ">>", (1000,250), cv2.FONT_HERSHEY_SIMPLEX, 4, (255,0,00), 3, cv2.LINE_AA )

                        except NameError:
                            continue
                        #lane draw line red
                        cv2.line(birds_eye, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    #DRAWING POINTS FOR DRIVE.mp4
                    drawing_pts = np.array([[[550, 440],[690, 440],[1000, 640],[200, 640]]], np.int32)
                    #DRAWING POINTS FOR CALI_VID.MP4
                    #drawing_pts = np.array([[[550, 400],[690, 400],[1000, 500],[200, 500]]], np.int32)
                    #REAL DRIVEMP4
                    #drawing_pts = np.array([[[560, 340],[710, 440],[1000, 640],[200, 640],]], np.int32)

                    #SELF_DRIVING.AVI
                    #drawing_pts = np.array([[[560, 340],[710, 440],[1000, 640],[200, 640],]], np.int32)
                    cv2.polylines(frame, [drawing_pts],True, (0,255,),3)
                    # cv2.fillPoly(frame, [drawing_pts], (0,255,0))
                        
                    birds_eye = cv2.resize(birds_eye, (640,360))
                    t = cv2.getTickCount() - t
                    fps = cv2.getTickFrequency() / float(t)
                    cv2.putText(frame, "FPS : " + str(int(fps)), (900, 50), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.imshow("Top_view", birds_eye)
                    cv2.imshow("Lane Departure Warning System", frame)
                    x+=1
                    #time.sleep(0.4)
                    print(x)
                key = cv2.waitKey(1)

                if key == ord('d'):
                    print("D was pressed by user , exiting Lane Departure Warning System")
                    cv2.destroyWindow("Lane Departure Warning System")
                    cv2.destroyWindow("Top_view")
                    cap.release()
                    stop_flag = True
                    break
                elif key == ord('q'):
                    print("Q was pressed by user , exiting Lane Departure Warning System")
                    cv2.destroyWindow("Lane Departure Warning System")
                    cv2.destroyWindow("Top_view")
                    cap.release()
                    stop_flag = True
                    break

    #function for pedestrian detection
    def pedestrian_detection():
        y1=0
        thread = threading.current_thread()
        print("Thread name:", thread.name)
        print("Thread ID:", thread.ident)
        print("Thread status:", thread.is_alive())
        p = psutil.Process(os.getpid())
        print(f"Pedestrian Detection thread is running on CPU core {p.cpu_affinity()}")
        cap = cv2.VideoCapture("pedestrian_detect123.mp4")
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        cv2.namedWindow("Pedestrian Detection Window", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Pedestrian Detection Window", 600, 400)
        stop_flag = False
        while not stop_flag:
            t = cv2.getTickCount()
            ret, frame = cap.read()
            if not ret:
                break
            image = frame.copy()
            image = cv2.resize(frame, (600,400), interpolation=cv2.INTER_LINEAR)
            mask = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
            image_height, image_width = frame.shape[:2]
            p1 = (0, image_height)
            p2 = (0, 0)
            p3 = (image_width, 0)
            p4 = (image_width, image_height)
            vertices = np.array([p1, p2, p3, p4], np.int32)
            verticesToFill = [vertices]
            cv2.fillPoly(mask, verticesToFill, (255, 255, 255))
            maskedIm = cv2.bitwise_and(image, mask)
            found, _ = hog.detectMultiScale(maskedIm, winStride=(8, 8), padding=(0, 0), scale=1.1)
            for (x, y, w, h) in found:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                #text = f"Pedestrian ({x}, {y})"
                text = f"Pedestrian"
                cv2.putText(image, text, (x, y + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
            t = cv2.getTickCount() - t
            fps = cv2.getTickFrequency() / float(t)
            cv2.putText(image, "Frames: {:.2f}".format(fps), (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2)

            y1=y1+1
            print("Total Frames ",y1)
            cv2.imshow("Pedestrian Detection Window", image)
            #time.sleep(0.1)
            key = cv2.waitKey(1)

            if key == ord('d'):
                print("D was pressed by user , exiting Pedestrian Detection System")
                cv2.destroyWindow("Pedestrian Detection Window")
                cap.release()
                stop_flag = True
                break
            elif key == ord('q'):
                print("D was pressed by user , exiting Pedestrian Detection System")
                cv2.destroyWindow("Pedestrian Detection Window")
                cap.release()
                stop_flag = True
                break
        
    #function for stop sign detection
    def stop_sign_detection1():
        #t = cv2.getTickCount()
        thread = threading.current_thread()
        print("Thread name:", thread.name)
        print("Thread ID:", thread.ident)
        print("Thread status:", thread.is_alive())
        p = psutil.Process(os.getpid())
        print(f"Stop Sign detection System thread is running on CPU core {p.cpu_affinity()}")
        cap = cv2.VideoCapture("video_edited.mp4")
        cv2.namedWindow("Stop Sign detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Stop Sign detection", 640, 480)
        print("here0")
        while True:
            print("here1")
            ret, frame1 = cap.read()
            if not ret:
                break
            t = cv2.getTickCount()
            frame = cv2.resize(frame1, (480, 320), interpolation=cv2.INTER_LINEAR)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower_red = np.array([0, 210, 20])
            upper_red = np.array([179, 255, 255])
            red = cv2.inRange(hsv, lower_red, upper_red)
            blur = cv2.medianBlur(red, 5)
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            vertices = np.array([[20, 110], [20, 0], [440, 0], [440, 110]], np.int32)
            cv2.fillPoly(mask, [vertices], (255, 255, 255))
            maskedIm = cv2.bitwise_and(blur, blur, mask=mask)
            circles = cv2.HoughCircles(maskedIm, cv2.HOUGH_GRADIENT, 1, 50, param1=425, param2=9, minRadius=4, maxRadius=10)
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                for c in circles:
                    center = (c[0], c[1])
                    cv2.circle(frame, center, 10, (0, 0, 255), 3, cv2.LINE_AA)
                    radius = c[2]
                    cv2.putText(frame, "STOP Sign", (120, 150), cv2.FONT_HERSHEY_PLAIN, 3.0, (0, 0, 255), 5, cv2.LINE_AA)
            t = cv2.getTickCount() - t
            fps = cv2.getTickFrequency() / t
            cv2.putText(frame, "FPS: {:.1f}".format(fps), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow("STOP_LIGHT_DETECTION", frame)
            key = cv2.waitKey(1)
            if key == ord('d'):
                print("D was pressed by user , exiting Stop Sign detection System")
                cv2.destroyWindow("STOP_LIGHT_DETECTION")
                cap.release()
                stop_flag = True
                break
            elif key == ord('q'):
                print("D was pressed by user , exiting Stop Sign detection System")
                cv2.destroyWindow("STOP_LIGHT_DETECTION")
                cap.release()
                stop_flag = True
                break

    def stop_sign_detection():
        thread = threading.current_thread()
        print("Thread name:", thread.name)
        print("Thread ID:", thread.ident)
        print("Thread status:", thread.is_alive())
        p = psutil.Process(os.getpid())
        print(f"Stop Sign detection System thread is running on CPU core {p.cpu_affinity()}")
        stop_sign_cascade = cv2.CascadeClassifier('stop_sign_classifier_2.xml')
        cap = cv2.VideoCapture('video_edited.mp4')
        stop_flag = False
        while not stop_flag:
            
            ret,frame = cap.read()
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_gray = cv2.equalizeHist(frame_gray)
            t = cv2.getTickCount()
            stopSigns = stop_sign_cascade.detectMultiScale(frame_gray, 1.1, 4, 0 | cv2.CASCADE_SCALE_IMAGE, minSize=(50, 50))
            for (x, y, w, h) in stopSigns:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 6)
            t = cv2.getTickCount() - t
            fps = cv2.getTickFrequency() / float(t)
            cv2.putText(frame, "Frames: {:.2f}".format(fps), (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2)
            cv2.imshow("Stop Sign Detection", frame)
            time.sleep(0.1)
            key = cv2.waitKey(1)
            if key == ord('d'):
                print("D was pressed by user , exiting Stop Sign Detection System")
                cv2.destroyWindow("Stop Sign Detection")
                cap.release()
                stop_flag = True
                break
            elif key == ord('q'):
                print("D was pressed by user , exiting Stop Sign Detection System")
                cv2.destroyWindow("Stop Sign Detection")
                cap.release()
                stop_flag = True
                break

    #function for car detection
    def car_detection():
        thread = threading.current_thread()
        print("Thread name:", thread.name)
        print("Thread ID:", thread.ident)
        print("Thread status:", thread.is_alive())
        p = psutil.Process(os.getpid())
        print(f"Car detection System thread is running on CPU core {p.cpu_affinity()}")
        car_cascade = cv2.CascadeClassifier('cars.xml')
        #Open the video capture device
        cap = cv2.VideoCapture('dataset_video2.avi')
        while True:
            #Read a frame from the video capture device
            ret, frame = cap.read()
            #If there's an error or the video has ended, break out of the loop
            if not ret:
                break
            
            #Convert the frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            t = cv2.getTickCount()
            #Detect cars in the grayscale frame
            cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1)
            #Draw rectangles around the detected carsq
            for (x, y, w, h) in cars:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            #Display the frame
            
            t = (cv2.getTickCount() - t) / cv2.getTickFrequency()
            fps = 1.0 / t
            cv2.putText(frame, "Frames: {:.2f}".format(fps), (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2)
            cv2.imshow('Car Detection', frame)
            time.sleep(0.4)
            key = cv2.waitKey(1)
            if key == ord('d'):
                print("D was pressed by user , exiting Car Detection System")
                cv2.destroyWindow("Car Detection")
                cap.release()
                stop_flag = True
                break
            elif key == ord('q'):
                print("D was pressed by user , exiting Car Detection System")
                cv2.destroyWindow("Car Detection")
                cap.release()
                stop_flag = True
                break

    #Define function to exit the program
    def exit_program():
        print("Exiting program")
        root.destroy()

    root = tk.Tk()
    root.title("CSCI 612 Final Project")
    root.geometry("800x500")
    thread = threading.current_thread()
    print("Thread name:", thread.name)
    print("Thread ID:", thread.ident)
    print("Thread status:", thread.is_alive())
    button_font = Font(family="Helvetica", size=14)
    title_font = Font(family="Helvetica", size=18, weight="bold")
    bg_image = Image.open("bg.jpg")
    bg_photo = ImageTk.PhotoImage(bg_image)
    bg_label = tk.Label(root, image=bg_photo)
    bg_label.place(x=10, y=0, relwidth=0.4, relheight=1)


    title_label = tk.Label(root, text="Choose from available features:", font=title_font)
    title_label.pack(padx=20)

    lane_button = tk.Button(root, text="Lane Detection", command=perform_lane_detection)
    set_button_style(lane_button)
    lane_button.pack(pady=20,padx=(200,0))

    pedestrian_button = tk.Button(root, text="Pedestrian Detection", command=perform_pedestrian_detection)
    set_button_style(pedestrian_button)
    pedestrian_button.pack(pady=20,padx=(200,0))

    stop_sign_button = tk.Button(root, text="Stop Sign Detection", command=perform_stop_sign_detection)
    set_button_style(stop_sign_button)
    stop_sign_button.pack(pady=20,padx=(200,0))

    car_button = tk.Button(root, text="Car Detection", command=perform_car_detection)
    set_button_style(car_button)
    car_button.pack(pady=10,padx=(200,0))

    exit_button = tk.Button(root, text="Exit", command=exit_program)
    set_button_style(exit_button)
    exit_button.pack(side="bottom", padx=20, pady=20)

    root.mainloop()

    print("Running threads:", threading.enumerate())

