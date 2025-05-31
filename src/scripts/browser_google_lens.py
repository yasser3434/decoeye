import os
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from urllib.parse import urlparse
import json

# Initialize Chrome browser
options = webdriver.ChromeOptions()
options.add_argument("--start-maximized")
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# Folder with cropped images
cropped_folder = "data/generated_images/vide/flux_kontext_max/2025-05-31/cropped_objects"
output_data = []

for filename in os.listdir(cropped_folder):
    if not filename.lower().endswith((".jpg", ".png")):
        continue

    image_path = os.path.abspath(os.path.join(cropped_folder, filename))
    print(f"Searching for: {filename}")

    # Open Google Lens upload page
    driver.get("https://lens.google.com/upload")

    # Upload the image
    time.sleep(2)
    upload_input = driver.find_element(By.XPATH, '//input[@type="file"]')
    upload_input.send_keys(image_path)

    # Wait for search results to load
    time.sleep(7)

    # Get result cards
    try:
        cards = driver.find_elements(By.XPATH, '//a[contains(@class, "Vd9M6")]')
        matches = []
        for card in cards[:5]:  # Limit to first 5
            try:
                title = card.get_attribute("aria-label") or "No Title"
                link = card.get_attribute("href")
                img = card.find_element(By.TAG_NAME, "img").get_attribute("src")
                matches.append({
                    "title": title,
                    "link": link,
                    "image": img
                })
            except Exception as e:
                print("Skipping one card:", e)
        
        output_data.append({
            "source_image": filename,
            "matches": matches
        })
    except Exception as e:
        print("Error scraping Google Lens:", e)

# Save output as JSON
with open("matched_products.json", "w") as f:
    json.dump(output_data, f, indent=2)

driver.quit()
print("Finished searching and saving matched data.")

