# 🪄 Magic Click — User Guide

Welcome to Magic Click! This system acts as your intelligent photographer it watches the camera, automatically captures the best picture of every person in the room (eyes open, sharp, facing forward), and instantly organizes them by person.

Here is how to use it in 3 easy steps.

---

## 🚀 1. Starting the System

You do not need to install complicated software or open any terminal windows.

### Windows Users
* Double-click the **`MagicClick.exe`** file (or `MagicClick.bat`).
* A small loading window will appear. The very first time you run it, it might take a few minutes to download the camera models.
* Once the bar finishes, your web browser will automatically open and the camera feed will start.

### Mac Users
* Open the `launchers/macos` folder and double-click **`MagicClick.command`** (or the `MagicClick.app` icon if provided to you).
* A terminal window will briefly flash, followed by the loading screen. 
* Your browser will automatically open with the dashboard.

---

## 🎛️ 2. Understanding the Dashboard

When the system opens in your browser (at `http://localhost:5001`), you will see the **Control Center**.

### The Main Camera Window
A new window will pop up showing your actual live camera feed.
* As people walk into the frame, the camera will automatically draw boxes around them.
* You don't have to click anything! When the system thinks it has a "great shot", it will save it in the background automatically.

### Adding and Removing Cameras
In the browser dashboard, scroll down to the **Camera Management** section:
* **To add your Phone as a camera:** Download "IP Camera Lite" on your phone. Connect to your Wi-Fi, turn it on, and type the URL it gives you (like `http://192.168.1.15:8080/video`) into the "Source" box. Click **Test**, then **Save**.
* **Toggle Cameras:** Click the green switch next to any camera to turn it off or on instantly.

---

## 🖼️ 3. Viewing, Naming, and Downloading Photos

The system saves every great photo and silently groups them by person. Because it doesn't know their names yet, they are initially assigned a long ID code.

### Option A: The "Photobooth" Mode
Click the **Retrieval** tab at the bottom of the dashboard (or go to `http://localhost:5001/camera`).
1. Stand in front of the camera and click **Capture**.
2. The system will look at your face, instantly search its memory, and show you **every good photo it has taken of you today**.
3. It will only show *your* photos, never anyone else's!

### Option B: The "Organized Files" Mode
If you just want to see the photos the system has saved:
1. Open the folder where Magic Click is installed on your computer.
2. Go to **`data`** ➡️ **`images`**.
3. Inside, you will see a folder created for every unique person who walked by the camera today. Inside each folder are their best photos.

### Naming Someone
If you know a person's name, you can tell the system!
Currently, the easiest way to give someone a name is to open the `metadata.db` file with a free SQLite viewer app, find their long ID number, and type their name in the "name" column. Future updates will let you do this directly in the browser!

---

## 🛑 How to Stop Magic Click Safely

When your event is over, you need to turn off the smart cameras.
* Note: Just closing your browser **does not** stop the camera!
* **To stop:** Go to the live video window (the one showing the actual camera feed) and press the **`q`** key on your keyboard. 
* The system will gracefully save everything it is working on, close the cameras, and shut down.

---

### FAQ / Troubleshooting

**Q: The camera window isn't opening, but the browser did.**
* Check the **Camera Management** panel in your browser. Make sure at least one camera is added and toggled "On". Make sure your laptop's privacy settings allow apps to use the camera.

**Q: I stood in front of the camera, but it didn't save any photos of me.**
* Magic Click is picky! It is looking for a "good" portrait. Stand still, look generally toward the camera, make sure the room isn't too dark, and wait a second.

**Q: The browser page says "No matching person found" when I click Capture.**
* This means the system hasn't captured a high-quality "base" image of you yet for its memory bank. Walk around for a minute, let the live camera see you clearly, and try again.
