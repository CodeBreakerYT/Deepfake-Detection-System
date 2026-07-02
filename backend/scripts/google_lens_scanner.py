import requests
import base64

class GoogleLensScanner:
    """
    Utility class to interact with Google Lens upload API.
    Provides a free method to perform reverse image searches.
    """
    def __init__(self):
        self.upload_url = "https://lens.google.com/v3/upload"

    def get_lens_url_for_image(self, file_path: str = None, base64_str: str = None) -> str:
        """
        Uploads an image (either from file path or base64 string) to Google Lens
        and returns the resulting Google Search URL.
        """
        if file_path:
            with open(file_path, "rb") as f:
                image_data = f.read()
            filename = file_path.split("/")[-1].split("\\")[-1]
        elif base64_str:
            # Handle potential data URI scheme
            if "," in base64_str:
                base64_str = base64_str.split(",")[1]
            image_data = base64.b64decode(base64_str)
            filename = "upload.jpg"
        else:
            raise ValueError("Must provide either file_path or base64_str")

        files = {
            'encoded_image': (filename, image_data, 'image/jpeg')
        }

        try:
            # We allow redirects to capture the final Google Search URL in response.url
            response = requests.post(self.upload_url, files=files, timeout=10, allow_redirects=True)
            if response.status_code == 200:
                return response.url
            else:
                raise Exception(f"Google Lens returned status code {response.status_code}")
        except Exception as e:
            print(f"Error during Google Lens upload: {e}")
            return None
