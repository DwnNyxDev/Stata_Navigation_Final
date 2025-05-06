#imports
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from tqdm import tqdm
from PIL import Image
import os
import io
import concurrent.futures
import time



def find_all_images_in_folder(drive_service, folder_id, folder_name = "Root"): 
    all_images = []

    # Get all subfolders in the current folder
    sub_folders = drive_service.files().list(
    q=f"'{folder_id}' in parents and mimeType = 'application/vnd.google-apps.folder'",
    fields="files(id, name)").execute().get('files', [])
    
    # Recursively get images from each subfolder
    for folder in sub_folders:
        # Recursively get images from the subfolder
        sub_images = find_all_images_in_folder(drive_service, folder["id"], folder["name"])
        all_images.extend(sub_images)

    # Get images in the current folder
    images_in_current_folder = drive_service.files().list(
        q=f"'{folder_id}' in parents and mimeType contains 'image/'",
        fields="files(id, name, mimeType)").execute().get('files', [])
    
    all_images.extend({"id": img['id'], "label": folder_name} for img in images_in_current_folder)

    return all_images

# Process each image and save it to the output folder
def download_and_save(image, max_retries=3):
    image_id = image["id"]
    label = image["label"]

    for attempt in range(1, max_retries + 1):
        try:
            request = drive_service.files().get_media(fileId=image_id)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()

            fh.seek(0)
            img = Image.open(fh).convert('RGB')
            file_path = os.path.join(output_folder, label, image_id) + ".jpg"
            img.save(file_path)
            return  # success, exit the function

        except Exception as e:
            print(f"⚠ Attempt {attempt} failed for {image_id}: {e}")
            if attempt < max_retries:
                wait_time = 2 ** attempt  # exponential backoff
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"❌ Failed to download {image_id} after {max_retries} attempts.")


if __name__ == "__main__":
    # Path to your service account key
    SERVICE_ACCOUNT_FILE = 'service_account.json'
    SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

    # Authenticate
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)

    # Build the Drive API client
    drive_service = build('drive', 'v3', credentials=creds)

    #Load images and labels from Google Drive
    stata_folder_id = "1he-Vpjb3tGm3FQuCU3MT3un7MZr0-F05"
    images = find_all_images_in_folder(drive_service, stata_folder_id)
    class_names = [folder["name"] for folder in drive_service.files().list(
        q=f"'{stata_folder_id}' in parents and mimeType = 'application/vnd.google-apps.folder'",
        fields="files(id, name)").execute().get('files', [])]
    # Output folder for resized images
    output_folder = 'processed_images'
    os.makedirs(output_folder, exist_ok=True)
    for class_name in class_names:
        os.makedirs(os.path.join(output_folder, class_name), exist_ok=True)

    # Run parallel downloads in notebook (adjust max_workers as needed)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        for _ in tqdm(executor.map(download_and_save, images), total=len(images), dynamic_ncols=True):
            pass
        
    print("All images downloaded and saved successfully.")