# PART-1 = text to sign
import os
import string
import subprocess
import webbrowser

import imageio.v3 as iio  # combining gifs
import numpy as np  # for converting the animation into matrix
import tkinter as tk
from tkinter import *  # creating frame UI
from PIL import Image, ImageTk, ImageSequence
from googletrans import Translator
import time

# PART-2 = sign to text
import cv2
import mediapipe as mp              # model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


def func(text, root_input):
    isl_gif = ['hello', 'welcome','indian','sign','language']
    print(text)

    try:
        translator = Translator()
        detect = translator.detect(text)
        print(detect.lang)
        text_pass = text
        if detect.lang == 'hi':
            text_pass = translator.translate(text, dest='en', src="hi")
            text_pass = text_pass.text
        elif detect.lang == 'en':
            text_pass = text
        a = text_pass.lower()
        if len(a) == 0:
            print("NO TEXT FOUND!!")

        else:
            print("you said " + a.lower())
        for c in string.punctuation:
            a = a.replace(c, "")
        print(a)
        split_a = list(a.split(" "))
        lst = [i.replace('\n', '') for i in split_a]
        print(type(lst))
        frames_lst = [iio.imread("Animations/{}.gif".format(a)) for a in lst if a.lower() in isl_gif]
        frames = np.vstack(frames_lst)
        duration = iio.immeta("Animations/hello.gif")["duration"]

        iio.imwrite("Animations/result.gif", frames, duration=duration)

        root_display = tk.Toplevel()
        root_display.title("ISL Animated Translator and Desktop Virtual Assistant")
        root_display.geometry("1366x768")
        Label2 = tk.Label(root_display, text="ISL Animated Translator and Desktop Virtual Assistant",
                          font=("times new roman", 40, "bold"), bd=1, bg="grey", fg='white')
        Label2.pack(ipadx=10, ipady=10)

        root_input.wm_withdraw()

        def play_gif():
            global imge
            imge = Image.open("Animations/result.gif")

            lbl = Label(root_display)
            lbl.place(x=30, y=130)

            for imge in ImageSequence.Iterator(imge):
                imge = imge.resize((900, 500))
                imge = ImageTk.PhotoImage(imge)
                lbl.config(image=imge)
                root_display.update()
                time.sleep(0.05)

        def exit():
            root_display.destroy()
            return root_input.wm_deiconify()


        # play button
        def playgif_button_hover(e):
            play_button.config(background='OrangeRed3', foreground="white")
        def playgif_hover_leave(e):
            play_button.config(background='SystemButtonFace', foreground="black")
        play_button = Button(root_display, text="Play", font=("Constantia", 20), bg='#ffffff', width=12, height=1,
               activebackground="#B1CBF6", command=play_gif)
        play_button.place(x=1000, y=200)
        play_button.bind("<Enter>", playgif_button_hover)
        play_button.bind("<Leave>", playgif_hover_leave)
        # again button
        def againgif_button_hover(e):
            again_button.config(background='OrangeRed3', foreground="white")
        def againgif_hover_leave(e):
            again_button.config(background='SystemButtonFace', foreground="black")
        again_button = Button(root_display, text="Replay", font=("Constantia", 20), bg='#ffffff', width=12, height=1,
               activebackground="#B1CBF6", command=play_gif)
        again_button.place(x=1000, y=300)
        again_button.bind("<Enter>", againgif_button_hover)
        again_button.bind("<Leave>", againgif_hover_leave)
        # exit button
        def exitgif_button_hover(e):
            exit_button.config(background='OrangeRed3', foreground="white")
        def exitgif_hover_leave(e):
            exit_button.config(background='SystemButtonFace', foreground="black")
        exit_button = Button(root_display, text="Exit", font=("Constantia", 20), bg='#ffffff', width=12, height=1,
               activebackground="#B1CBF6", command=exit)
        exit_button.place(x=1000, y=400)
        exit_button.bind("<Enter>", exitgif_button_hover)
        exit_button.bind("<Leave>", exitgif_hover_leave)

    except:
        print("error")


def Application():
    mp_holistic = mp.solutions.holistic  # Holistic model
    mp_drawing = mp.solutions.drawing_utils  # Drawing utilities

    def mediapipe_detection(image, model):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
        image.flags.writeable = False  # Image is no longer writeable
        results = model.process(image)  # Make prediction
        image.flags.writeable = True  # Image is now writeable
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR COVERSION RGB 2 BGR
        return image, results

    def draw_landmarks(image, results):
        mp_drawing.draw_landmarks(image, results.face_landmarks,
                                  mp_holistic.FACEMESH_TESSELATION)  # Draw face connections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)  # Draw pose connections
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks,
                                  mp_holistic.HAND_CONNECTIONS)  # Draw left-hand connections
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks,
                                  mp_holistic.HAND_CONNECTIONS)  # Draw right-hand connections

    def draw_styled_landmarks(image, results):
        # Draw face connections
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                                  mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                  mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                                  )
        # Draw pose connections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                                  )
        # Draw left-hand connections
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                                  )
        # Draw right-hand connections
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )

    def extract_keypoints(results):
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                         results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
        face = np.array([[res.x, res.y, res.z] for res in
                         results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
        lh = np.array([[res.x, res.y, res.z] for res in
                       results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(
            21 * 3)
        rh = np.array([[res.x, res.y, res.z] for res in
                       results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
            21 * 3)
        return np.concatenate([pose, face, lh, rh])

    # Path for exported data, numpy arrays
    # DATA_PATH = os.path.join('MP_Data')

    # Actions that we try to detect
    actions = np.array(['bye', 'hello', 'indian', 'language', 'sign'])

    # Thirty videos worth of data
    no_sequences = 30

    # Videos are going to be 30 frames in length
    sequence_length = 30

    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))

    res = [.7, 0.2, 0.1]

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    model.load_weights('action.h5')

    colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245), (252, 186, 3), (148, 3, 252)]

    def prob_viz(res, actions, input_frame, colors):
        output_frame = input_frame.copy()
        for num, prob in enumerate(res):
            cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
            cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                        cv2.LINE_AA)
        return output_frame

    root_sign = tk.Toplevel()
    root_sign.title("Sign Language To Text Conversion")
    root_sign.geometry("1366x768")
    label1 = tk.Label(root_sign, text="ISL Animated Translator and Desktop Virtual Assistant",
                      font=("times new roman", 40, "bold"), bd=1, bg="grey", fg='white')
    label1.pack(ipadx=10, ipady=10)
    mainFrame = Frame(root_sign)
    mainFrame.place(x=700, y=100)
    root_front.wm_withdraw()

    # Capture video frames
    lmain = tk.Label(mainFrame)
    lmain.grid(row=0, column=0)
    text_var = tk.StringVar()
    label2 = Label(root_sign, textvariable=text_var, font=("times new roman", 20, "bold"), bd=1, bg="grey", fg='white')
    label2.pack(pady=50, padx=50, ipadx=10, ipady=10, anchor=tk.W)
    text_var.set("Identifying Word = None")

    para_var = tk.StringVar()
    label3 = Label(root_sign, textvariable=para_var, font=("times new roman", 20, "bold"), bd=1, bg="grey", fg='white',
                   justify=LEFT)
    label3.pack(pady=50, padx=50, ipadx=10, ipady=10, anchor=tk.W)
    para_var.set("Sentence = None")
    global result

    def sentences(word):

        result.append(word)
        time.sleep(2)
        if len(result) >= 35:
            result.clear()
        r = "\n".join([" ".join(result[i:i + 6]) for i in range(0, len(result), 6)])

        return r

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    result = []


    def again(result):
        result.clear()
        para_var.set("Sentence = None")
        return result

    def againsign_button_hover(e):
        againButton.config(background='OrangeRed3', foreground="white")

    def againsign_hover_leave(e):
        againButton.config(background='SystemButtonFace', foreground="black")

    againButton = Button(root_sign, text="REFRESH", font=("Constantia", 12), bg='#ffffff', width=20, height=1)
    againButton.configure(command=lambda: again(result))
    againButton.place(x=50, y=500)
    againButton.bind("<Enter>", againsign_button_hover)
    againButton.bind("<Leave>", againsign_hover_leave)

    def homeFromSign(root_sign):
        cap.release()
        cv2.destroyAllWindows()
        root_sign.destroy()
        return root_front.wm_deiconify()

    # close button
    def homesign_button_hover(e):
        HomeButton.config(background='OrangeRed3', foreground="white")

    def homesign_hover_leave(e):
        HomeButton.config(background='SystemButtonFace', foreground="black")

    HomeButton = Button(root_sign, text="HOME", font=("Constantia", 12), bg='#ffffff', width=20, height=1)
    HomeButton.configure(command=lambda: homeFromSign(root_sign))
    HomeButton.place(x=300, y=500)
    HomeButton.bind("<Enter>", homesign_button_hover)
    HomeButton.bind("<Leave>", homesign_hover_leave)

    """ des_var = tk.StringVar()
    label3 = Label(root_sign, textvariable=des_var, font=("times new roman", 10, "bold"), justify=LEFT)
    label3.place(x=10, y=600)
    description = 'NOTE:\n\n(1) Place finger on the camera for SPACE \n(2) Click twice on Home to redirect to home page' \
                  '\n\n\n\n COMPUTER COMMANDS AVAILABLE NOW:\n\nNotepad | Calculator | Google | Camera | Youtube |' \
                  ' Log off/Shutdown/Sign out'
    des_var.set(description)"""

    # 1. New detection variables
    sequence = []
    sentence = []
    threshold = 0.8

    # Set mediapipe model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():

            # Read feed
            ret, frame = cap.read()
            frame = cv2.resize(frame, (800, 650))
            # Make detections
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(img))
            lmain['image'] = img
            image, results = mediapipe_detection(frame, holistic)
            # print(results)
            """# stop sign button
            def stopsign_button_hover(e):
                stopButton.config(background='OrangeRed3', foreground="white")


            def stopsign_hover_leave(e):
                stopButton.config(background='SystemButtonFace', foreground="black")


            stopButton = Button(root_sign, text="STOP", font=("Constantia", 12), bg='#ffffff', width=20, height=1)
            stopButton.configure(command=lambda: root_sign.after_cancel(frame))
            stopButton.place(x=50, y=400)
            stopButton.bind("<Enter>", stopsign_button_hover)
            stopButton.bind("<Leave>", stopsign_hover_leave)"""
            # Draw landmarks
            # draw_styled_landmarks(image, results)

            # 2. Prediction logic
            keypoints = extract_keypoints(results)
            #         sequence.insert(0,keypoints)
            #         sequence = sequence[:30]
            sequence.append(keypoints)
            sequence = sequence[-30:]

            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                # print(actions[np.argmax(res)])

                # 3. Viz logic
                if res[np.argmax(res)] > threshold:
                    if len(sentence) > 0:
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                            print(actions[np.argmax(res)])
                            T = str(actions[np.argmax(res)])
                            text_var.set("Current Letter = {}".format(T))
                            S = sentences(str(actions[np.argmax(res)]))
                            para_var.set("Sentence = {}".format(S))
                    else:
                        sentence.append(actions[np.argmax(res)])
                        print(actions[np.argmax(res)])

                if len(sentence) > 5:
                    sentence = sentence[-5:]

                # Viz probabilities
                image = prob_viz(res, actions, image, colors)

            cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)

            # Show to screen
            # cv2.imshow('OpenCV Feed', image)

            root_sign.update()


def Assistant():
    mp_holistic = mp.solutions.holistic  # Holistic model
    mp_drawing = mp.solutions.drawing_utils  # Drawing utilities

    def mediapipe_detection(image, model):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
        image.flags.writeable = False  # Image is no longer writeable
        results = model.process(image)  # Make prediction
        image.flags.writeable = True  # Image is now writeable
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR COVERSION RGB 2 BGR
        return image, results

    def draw_landmarks(image, results):
        mp_drawing.draw_landmarks(image, results.face_landmarks,
                                  mp_holistic.FACEMESH_TESSELATION)  # Draw face connections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)  # Draw pose connections
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks,
                                  mp_holistic.HAND_CONNECTIONS)  # Draw left-hand connections
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks,
                                  mp_holistic.HAND_CONNECTIONS)  # Draw right-hand connections

    def draw_styled_landmarks(image, results):
        # Draw face connections
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                                  mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                  mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                                  )
        # Draw pose connections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                                  )
        # Draw left-hand connections
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                                  )
        # Draw right-hand connections
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )

    def extract_keypoints(results):
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                         results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
        face = np.array([[res.x, res.y, res.z] for res in
                         results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
        lh = np.array([[res.x, res.y, res.z] for res in
                       results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(
            21 * 3)
        rh = np.array([[res.x, res.y, res.z] for res in
                       results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
            21 * 3)
        return np.concatenate([pose, face, lh, rh])

    # Path for exported data, numpy arrays
    # DATA_PATH = os.path.join('MP_Data')

    # Actions that we try to detect
    actions = np.array(['camera', 'google','whatsapp','youtube'])

    # Thirty videos worth of data
    no_sequences = 30

    # Videos are going to be 30 frames in length
    sequence_length = 30

    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))

    res = [.7, 0.2, 0.1]

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    model.load_weights('virtualassistant.h5')

    colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245), (252, 186, 3), (148, 3, 252)]

    def prob_viz(res, actions, input_frame, colors):
        output_frame = input_frame.copy()
        for num, prob in enumerate(res):
            cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
            cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                        cv2.LINE_AA)
        return output_frame

    root_sign = tk.Toplevel()
    root_sign.title("Sign Language To Text Conversion")
    root_sign.geometry("1366x768")
    label1 = tk.Label(root_sign, text="ISL Animated Translator and Desktop Virtual Assistant",
                      font=("times new roman", 40, "bold"), bd=1, bg="grey", fg='white')
    label1.pack(ipadx=10, ipady=10)
    mainFrame = Frame(root_sign)
    mainFrame.place(x=700, y=100)
    root_front.wm_withdraw()

    # Capture video frames
    lmain = tk.Label(mainFrame)
    lmain.grid(row=0, column=0)
    text_var = tk.StringVar()
    label2 = Label(root_sign, textvariable=text_var, font=("times new roman", 20, "bold"), bd=1, bg="grey", fg='white')
    label2.pack(pady=50, padx=50, ipadx=10, ipady=10, anchor=tk.W)
    text_var.set("Identifying Word = None")

    para_var = tk.StringVar()
    label3 = Label(root_sign, textvariable=para_var, font=("times new roman", 20, "bold"), bd=1, bg="grey", fg='white',
                   justify=LEFT)
    label3.pack(pady=50, padx=50, ipadx=10, ipady=10, anchor=tk.W)
    para_var.set("Sentence = None")
    global result

    def again(result):
        result.clear()
        para_var.set("Sentence = None")
        return result

    def againsign_button_hover(e):
        againButton.config(background='OrangeRed3', foreground="white")

    def againsign_hover_leave(e):
        againButton.config(background='SystemButtonFace', foreground="black")

    againButton = Button(root_sign, text="REFRESH", font=("Constantia", 12), bg='#ffffff', width=20, height=1)
    againButton.configure(command=lambda: again(result))
    againButton.place(x=50, y=500)
    againButton.bind("<Enter>", againsign_button_hover)
    againButton.bind("<Leave>", againsign_hover_leave)

    def homeFromSign(root_sign):
        cap.release()
        cv2.destroyAllWindows()
        root_sign.destroy()
        return root_front.wm_deiconify()

    # close button
    def homesign_button_hover(e):
        HomeButton.config(background='OrangeRed3', foreground="white")

    def homesign_hover_leave(e):
        HomeButton.config(background='SystemButtonFace', foreground="black")

    HomeButton = Button(root_sign, text="HOME", font=("Constantia", 12), bg='#ffffff', width=20, height=1)
    HomeButton.configure(command=lambda: homeFromSign(root_sign))
    HomeButton.place(x=300, y=500)
    HomeButton.bind("<Enter>", homesign_button_hover)
    HomeButton.bind("<Leave>", homesign_hover_leave)

    des_var = tk.StringVar()
    label3 = Label(root_sign, textvariable=des_var, font=("times new roman", 10, "bold"), justify=LEFT)
    label3.place(x=10, y=550)
    description = 'NOTE:\n' \
                  'COMPUTER COMMANDS AVAILABLE NOW:\nGoogle | Camera | Youtube | Whatsapp' \

    des_var.set(description)


    def sentences(word):
        result.append(word)
        if len(result) >= 35:
            result.clear()
        r = "\n".join([" ".join(result[i:i + 5]) for i in range(0, len(result), 5)])
        query = r.lower()
        # Virtual Assistant
        if 'google' in query:
            webbrowser.open_new_tab("https://www.google.com")
            time.sleep(5)
            # os.system("taskkill /im chrome.exe /f")"""
            return again(result)
        elif 'youtube' in query:
            webbrowser.open_new_tab("https://www.youtube.com")
            time.sleep(5)
            #os.system("taskkill /im chrome.exe /f")"""
            return again(result)
            # speak("Google chrome is open now")
        elif 'whatsapp' in query:
            webbrowser.open_new_tab("https://web.whatsapp.com/")
            time.sleep(5)
            #os.system("taskkill /im chrome.exe /f")"""
            return again(result)
        elif 'camera' in query:
            subprocess.run('start microsoft.windows.camera:', shell=True)
            time.sleep(5)
            #subprocess.run('Taskkill /IM WindowsCamera.exe /F', shell=True)"""
            return again(result)
        else:
            print('something went wrong')

        return r

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    result = []

    # 1. New detection variables
    sequence = []
    sentence = []
    threshold = 0.9

    # Set mediapipe model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():

            # Read feed
            ret, frame = cap.read()
            frame = cv2.resize(frame, (800, 650))
            # Make detections
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(img))
            lmain['image'] = img
            image, results = mediapipe_detection(frame, holistic)
            # print(results)
            """# stop sign button
            def stopsign_button_hover(e):
                stopButton.config(background='OrangeRed3', foreground="white")


            def stopsign_hover_leave(e):
                stopButton.config(background='SystemButtonFace', foreground="black")


            stopButton = Button(root_sign, text="STOP", font=("Constantia", 12), bg='#ffffff', width=20, height=1)
            stopButton.configure(command=lambda: root_sign.after_cancel(frame))
            stopButton.place(x=50, y=400)
            stopButton.bind("<Enter>", stopsign_button_hover)
            stopButton.bind("<Leave>", stopsign_hover_leave)"""
            # Draw landmarks
            # draw_styled_landmarks(image, results)

            # 2. Prediction logic
            keypoints = extract_keypoints(results)
            #         sequence.insert(0,keypoints)
            #         sequence = sequence[:30]
            sequence.append(keypoints)
            sequence = sequence[-30:]

            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                # print(actions[np.argmax(res)])

                # 3. Viz logic
                if res[np.argmax(res)] > threshold:
                    if len(sentence) > 0:
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                            print(actions[np.argmax(res)])
                            T = str(actions[np.argmax(res)])
                            text_var.set("Current Letter = {}".format(T))
                            S = sentences(str(actions[np.argmax(res)]))
                            para_var.set("Sentence = {}".format(S))
                    else:
                        sentence.append(actions[np.argmax(res)])
                        print(actions[np.argmax(res)])

                if len(sentence) > 5:
                    sentence = sentence[-5:]

                # Viz probabilities
                image = prob_viz(res, actions, image, colors)

            cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)

            # Show to screen
            # cv2.imshow('OpenCV Feed', image)

            root_sign.update()


def homeFromInput(root_input):
    root_input.destroy()
    return root_front.wm_deiconify()


def text_input():
    root_input = tk.Toplevel()
    root_input.title("ISL Animated Translator and Desktop Virtual Assistant")
    root_input.geometry("1366x768")
    LabelInput = tk.Label(root_input, text="ISL Animated Translator and Desktop Virtual Assistant",
                          font=("times new roman", 30, "bold"), bd=1, bg="grey", fg='white')
    LabelInput.pack(ipadx=10, ipady=10)
    root_front.wm_withdraw()
    ent = Entry(root_input)
    # Textbox
    TextInput = Text(root_input, wrap=WORD, height=10, width=83, font=("Constantia", 20), padx=20, pady=10)
    TextInput.place(x=120, y=100)

    # home
    def home_button_hover(e):
        homeButton.config(background='OrangeRed3', foreground="white")
    def home_hover_leave(e):
        homeButton.config(background='SystemButtonFace', foreground="black")
    homeButton = Button(root_input, text="HOME", font=("Constantia", 20), bg='#ffffff', width=20, height=1, activebackground="#B1CBF6")
    homeButton.configure(command=lambda: homeFromInput(root_input))
    homeButton.place(x=250, y=500)
    homeButton.bind("<Enter>", home_button_hover)
    homeButton.bind("<Leave>", home_hover_leave)

    # Confirm text
    def confirm_button_hover(e):
        confirmButton.config(background='OrangeRed3', foreground="white")
    def confirm_hover_leave(e):
        confirmButton.config(background='SystemButtonFace', foreground="black")
    confirmButton = Button(root_input, text="NEXT", font=("Constantia", 20), bg='#ffffff', width=20, height=1, activebackground="#B1CBF6")
    confirmButton.configure(command=lambda: func(TextInput.get("1.0", END), root_input))
    confirmButton.place(x=750, y=500)
    confirmButton.bind("<Enter>",confirm_button_hover)
    confirmButton.bind("<Leave>",confirm_hover_leave)


root_front = tk.Tk()
root_front.title("ISL Animated Translator and Desktop Virtual Assistant")
root_front.geometry("1530x900")
"""Label1 = tk.Label(root_front, text="ISL Animated Translator and Desktop Virtual Assistant",
                  font=("times new roman", 40, "bold"), bd=1, bg="grey", fg='white')
Label1.pack(ipadx=10, ipady=10)"""

# UI image
img = Image.open("finalimg.jpg")
img = img.resize((1500, 700))
img = ImageTk.PhotoImage(img)
panel = Label(root_front, image=img)
panel.image = img
panel.place(x=60, y=0)

# text button
def text_button_hover(e):
    textToSignButton.config(background='OrangeRed3', foreground= "white")
def text_hover_leave(e):
    textToSignButton.config(background='SystemButtonFace', foreground= "black")
textToSignButton = Button(root_front, text="TEXT TO SIGN", font=("Constantia", 20), bg='#ffffff', width=20, height=1, activebackground="#B1CBF6")
textToSignButton.configure(command=lambda: text_input())
textToSignButton.place(x=50, y=200)
textToSignButton.bind("<Enter>", text_button_hover)
textToSignButton.bind("<Leave>", text_hover_leave)

# sign button
def sign_button_hover(e):
    signToTextButton.config(background='OrangeRed3', foreground= "white")
def sign_hover_leave(e):
    signToTextButton.config(background='SystemButtonFace', foreground="black")
signToTextButton = Button(root_front, text="SIGN TO TEXT", font=("Constantia", 20), bg='#ffffff', width=20, height=1, activebackground="#B1CBF6")
signToTextButton.configure(command=lambda: Application())
signToTextButton.place(x=50, y=300)
signToTextButton.bind("<Enter>", sign_button_hover)
signToTextButton.bind("<Leave>", sign_hover_leave)

# Assistant button
def assistant_button_hover(e):
    assistantButton.config(background='OrangeRed3', foreground= "white")
def assistant_button_hover_leave(e):
    assistantButton.config(background='SystemButtonFace', foreground="black")
assistantButton = Button(root_front, text="VIRTUAL ASSISTANT", font=("Constantia", 20), bg='#ffffff', width=20, height=1, activebackground="#B1CBF6")
assistantButton.configure(command=lambda: Assistant())
assistantButton.place(x=50, y=400)
assistantButton.bind("<Enter>", assistant_button_hover)
assistantButton.bind("<Leave>", assistant_button_hover_leave)

# exit button
def exit_button_hover(e):
    exitButton.config(background='OrangeRed3', foreground= "white")
def exit_hover_leave(e):
    exitButton.config(background='SystemButtonFace', foreground="black")
exitButton = Button(root_front, text="EXIT", font=("Constantia", 20), bg='#ffffff', width=20, height=1, activebackground="#B1CBF6")
exitButton.configure(command=lambda: root_front.destroy())
exitButton.place(x=50, y=500)
exitButton.bind("<Enter>", exit_button_hover)
exitButton.bind("<Leave>", exit_hover_leave)

root_front.mainloop()
