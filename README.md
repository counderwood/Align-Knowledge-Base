# Align Builders Knowledge Base — Deployment Guide

## What this does
- Connects to your Google Drive "Align KB" folder automatically
- Downloads and indexes all PDFs when it starts
- Re-checks Drive every hour for new PDFs
- Your team accesses it through a web link — no setup needed on their end

---

## Your folder structure should look like this
```
align_kb_cloud/
  ├── server.py
  ├── requirements.txt
  └── static/
      └── index.html
```

---

## Step 1 — Push to GitHub

1. Go to github.com and create a free account
2. Click the green "New" button to create a new repository
3. Name it "align-kb" and click "Create repository"
4. Download GitHub Desktop from desktop.github.com
5. Open GitHub Desktop → File → Add Local Repository → point to this folder
6. Click "Publish repository"

---

## Step 2 — Deploy to Railway

1. Go to railway.app
2. Click "Start a New Project"
3. Click "Deploy from GitHub repo"
4. Select your "align-kb" repo
5. Railway will start setting it up automatically

---

## Step 3 — Add your environment variables

In Railway, click on your project → go to the "Variables" tab → add these one at a time:

### Variable 1
Name:  ANTHROPIC_API_KEY
Value: your key starting with sk-ant-...

### Variable 2
Name:  DRIVE_FOLDER_ID
Value: 1BfpoW2xFO1byMc1AH5JIiplozYZr7LmU

### Variable 3
Name:  GOOGLE_CREDENTIALS
Value: open your JSON key file in Notepad, select ALL the text, copy it, paste it here

---

## Step 4 — Get your public link

Railway gives you a URL like:
  https://align-kb.up.railway.app

Send that link to your team. Done.

---

## Adding new PDFs

Just drag them into the "Align KB" folder in Google Drive.
The app re-checks Drive every hour automatically.
Or click the "Sync Drive" button in the top right of the chat UI to trigger it immediately.

---

## Rotating your Google key (important security step)

Since the JSON key was shared, generate a new one:
1. Go to console.cloud.google.com
2. IAM & Admin → Service Accounts → click your service account
3. Keys tab → Add Key → Create New Key → JSON
4. Download the new file
5. Update the GOOGLE_CREDENTIALS variable in Railway with the new file contents
6. Delete the old key from the Keys tab
