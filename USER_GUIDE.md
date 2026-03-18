# Magic Click — Complete User Guide
### For Everyone (No Technical Knowledge Required)

---

## 📋 Table of Contents

1. [What Is Magic Click?](#what-is-magic-click)
2. [Before You Start — Checklist](#before-you-start)
3. [Part 1: Securing Your Project Files](#part-1-securing-your-project-files)
4. [Part 2: Installing Magic Click](#part-2-installing-magic-click)
5. [Part 3: Launching the System](#part-3-launching-the-system)
6. [Part 4: Using the Dashboard](#part-4-using-the-dashboard)
7. [Part 5: Where Logs Are Stored](#part-5-where-logs-are-stored)
8. [Part 6: Verifying Everything Is Working](#part-6-verifying-everything-is-working)
9. [Part 7: Shutting Down Safely](#part-7-shutting-down-safely)
10. [Troubleshooting Common Issues](#troubleshooting-common-issues)

---

## What Is Magic Click?

Magic Click is an AI-powered security and monitoring system. It uses your computer's camera to:
- **Detect people** in live video
- **Capture and score** images automatically
- **Store results** in a secure local database accessible from your web browser

You never need to open a terminal or write any code. Everything works with one click after setup.

---

## Before You Start

Please confirm all of the following before installing:

| ✅ | Requirement |
|---|---|
| ☐ | You are using **macOS 12+**, **Windows 10+**, or **Ubuntu 20.04+** |
| ☐ | Your computer has at least **8 GB of RAM** |
| ☐ | You have at least **3 GB of free disk space** |
| ☐ | You are connected to **Wi-Fi or Ethernet** (required for first-time setup) |
| ☐ | Your computer has a **webcam** (built-in or USB) |
| ☐ | You know your computer's **administrator password** |

> **Why do I need the internet?** The first setup downloads AI models (~500 MB, one-time only). After that, Magic Click works completely offline.

---

## Part 1: Securing Your Project Files

Before installing, keep your Magic Click folder safe.

### Step 1.1 — Store the Folder Safely
- Keep the `Magic_Click` folder in a location **only you can access**
- Recommended: your **Home folder** or **Documents folder**
- ❌ **Do NOT** put it on a shared network drive or a USB stick you pass around

### Step 1.2 — Do Not Share These Files
The following files contain sensitive session data and should never be shared:

| File | Why It's Sensitive |
|---|---|
| `mc_database/data/auth.json` | Contains your hashed login password |
| `mc_database/data/mc_database.db` | Contains all captured images and person records |
| `magic_click_install.log` | May contain system path details |

### Step 1.3 — Set Folder Permissions (macOS)
Open **Terminal** (press `⌘ Space`, type "Terminal") and run:
```bash
chmod -R 700 ~/Magic_Click
```
This makes the folder readable and writable only by you.

### Step 1.4 — Set Folder Permissions (Windows)
- Right-click the `Magic_Click` folder → **Properties**
- Click the **Security** tab → **Edit**
- Remove access for **Everyone** or **Users** if present
- Keep only your username with **Full control**

---

## Part 2: Installing Magic Click

### 🍎 macOS — Using the .pkg Installer

**Step 2.1** — Find the file named `MagicClick_Installer.pkg` in the `installers/macos/dist/` folder.

**Step 2.2** — Double-click the `.pkg` file to open it.

> ⚠️ **First time opening?** macOS may say *"cannot be opened because the developer cannot be verified."*
> - **Right-click** (or Control-click) the `.pkg` file → **Open**
> - Click **Open** in the popup that appears
> - If that doesn't work: **System Settings → Privacy & Security** → scroll down → click **Open Anyway**

**Step 2.3** — Follow the on-screen installer steps:
1. Welcome screen → Click **Continue**
2. Read the setup info (internet, timing) → Click **Continue**
3. Click **Install**
4. Enter your **Mac password** when prompted → Click **Install Software**

**Step 2.4** — A setup window will appear automatically showing progress. Wait for it to finish (5–15 minutes on first run).

**Step 2.5** — When you see **"Setup Complete!"**, click **▶ Launch Magic Click** to open the dashboard.

---

### 🪟 Windows — Using the .exe Installer

**Step 2.1** — Find the file `MagicClick_Setup.exe` in the `installers/windows/dist/` folder.

**Step 2.2** — Right-click the `.exe` → **Run as administrator**.

> ⚠️ **Windows may show a SmartScreen warning:** *"Windows protected your PC."*
> - Click **More info**
> - Click **Run anyway**
> - This is normal for independently distributed software.

**Step 2.3** — Follow the installer wizard:
1. Click **Next** on the welcome screen
2. Accept the default install location → **Next**
3. Check the boxes to create Desktop and Start Menu shortcuts → **Next**
4. Click **Install**

**Step 2.4** — A progress window will appear. Wait for everything to install (5–15 min).

**Step 2.5** — When done, leave **"Launch Magic Click now"** checked and click **Finish**.

---

### 🐧 Linux — Using the .deb Package

**Step 2.1** — Open a terminal in the folder containing `magic-click_2.0.0_amd64.deb`.

**Step 2.2** — Run:
```bash
sudo dpkg -i magic-click_2.0.0_amd64.deb
```

**Step 2.3** — The setup will run automatically. Look for **Magic Click** in your applications menu when done.

---

## Part 3: Launching the System

After installation, you launch Magic Click the same way every time:

| Platform | How to Launch |
|---|---|
| **macOS** | Double-click **MagicClick** in your Applications folder or Launchpad |
| **Windows** | Double-click the **Magic Click** shortcut on your Desktop |
| **Linux** | Click **Magic Click** in your application menu |

**What happens when you launch:**
1. A small window briefly appears (the startup manager)
2. The AI camera pipeline starts automatically in the background
3. Your web browser opens to: `http://localhost:5001/`
4. You see the **login page** — sign in to access the dashboard

> 💡 The first launch after installation may take 2–3 extra minutes as the AI finishes loading. Subsequent launches are instant.

---

## Part 4: Using the Dashboard

### Logging In
1. Open your browser and go to: **`http://localhost:5001/`**
2. On the **first run**, you will be asked to **create a password** — choose one you will remember
3. On all future visits, enter your password and click **Sign In**

### The Dashboard at a Glance

| Section | What It Shows |
|---|---|
| **System Live** (green dot) | Confirms the AI pipeline is running |
| **Persons Stored** | Number of people detected and saved |
| **Images Stored** | Number of captured photos in the database |
| **Camera Management** | Add, test, or remove cameras |
| **Storage Usage** | Disk space used by Magic Click |
| **AI Latency** | How fast the AI is processing (should say "Optimal") |

### The Retrieval Tab
Click **Retrieval** at the bottom to search through all captured images.

---

## Part 5: Where Logs Are Stored

Logs are text files that record what Magic Click is doing. You may need these if something goes wrong.

| Log File | Location | What It Records |
|---|---|---|
| **Install log** | `magic_click_install.log` in your Magic Click folder | Everything that happened during setup |
| **macOS install log** | `/tmp/magic_click_postinstall.log` | macOS-specific installation steps |
| **Windows install log** | `%TEMP%\magic_click_python_install.log` | Python installation on Windows |
| **Pipeline log** | Your Magic Click folder → `magic_click_setup.log` | Live camera and AI activity |

**How to view a log file:**
- **macOS/Linux**: Double-click the log file. It will open in TextEdit or similar.
- **Windows**: Right-click the log file → **Open with** → **Notepad**

---

## Part 6: Verifying Everything Is Working

Use this checklist after launch to confirm all parts of the pipeline are running correctly.

### ✅ Step-by-Step Verification

**Check 1 — Is the system online?**
- Open your browser and go to: `http://localhost:5001/api/health`
- You should see: `{"status": "ok"}`
- ✅ If you see this, the API is running. ❌ If the page doesn't load, see Troubleshooting.

**Check 2 — Is the login page working?**
- Go to: `http://localhost:5001/`
- You should see the Magic Click login page.
- ✅ If you do, the dashboard is serving correctly.

**Check 3 — Is the camera active?**
- After logging in, look for the **green "SYSTEM LIVE"** dot in the top of the dashboard.
- Under **Camera Management**, your camera should appear with a status indicator.
- ✅ Green = running. ❌ Red or missing = camera issue (see Troubleshooting).

**Check 4 — Is the AI processing working?**
- On the dashboard, check the **AI Latency** panel.
- It should say **"OPTIMAL"** or show a number like **"42 ms"**.
- ✅ Any number under 200 ms is good.

**Check 5 — Is data being saved?**
- Walk in front of your camera.
- After a few seconds, the **Persons Stored** and **Images Stored** counters should increase.
- ✅ Increasing numbers = the full pipeline is working end-to-end.

---

## Part 7: Shutting Down Safely

**Never just close your browser tab** — this leaves background processes running.

### To shut down properly:
1. Sign in to the dashboard if not already logged in
2. Click the **⏻ Shutdown** button in the top-right corner
3. Confirm the shutdown when prompted
4. The system will cleanly stop all AI processes, camera feeds, and the API server
5. Your browser will show a "System offline" message

> 💡 **To Sign Out without shutting down:** Click **Sign Out** in the top-right. This closes your browser session but keeps the system running.

---

## Troubleshooting Common Issues

### ❌ "Cannot be opened because the developer cannot be verified" (macOS)
- Right-click the file → **Open** → **Open** again in the dialog
- Or: **System Settings → Privacy & Security → Open Anyway**

### ❌ "Windows protected your PC" (Windows)
- Click **More info** → **Run anyway**

### ❌ The setup window never appears after installing
- Check the install log: `magic_click_install.log` in your Magic Click folder
- Make sure Python 3.10+ is installed: [python.org/downloads](https://www.python.org/downloads/)

### ❌ The browser shows "This site can't be reached" on localhost:5001
- The system may still be starting. Wait 30 seconds and refresh.
- Make sure you opened Magic Click first — the server only runs while Magic Click is open.

### ❌ The camera doesn't appear in the dashboard
- Make sure you granted camera access to the application
- Try adding the camera manually: Source = `0` (for built-in) or `1` (for USB)

### ❌ "AI Latency" shows "HIGH" or the numbers stop updating
- Your computer may be under heavy load. Close other applications.
- The first run processes more slowly while warming up AI models.

---

*Magic Click — Version 2.0 | For support, see the README in your Magic Click folder.*
