# Import the core Kotaemon App class that handles the RAG functionality
from ktem.main import App
import os

# Create an instance of the Kotaemon App
# This handles document management, embeddings, and chat functionality
app = App()

# Build the Gradio web interface
# This creates all the UI components like chat, file upload, settings etc.
demo = app.make()

# Get the temporary directory path from environment variables
# This is where Gradio stores uploaded files and other temporary data
GRADIO_TEMP_DIR = os.getenv("GRADIO_TEMP_DIR", None)

# If no temp directory is specified, create one in the app data directory
if GRADIO_TEMP_DIR is None:
    GRADIO_TEMP_DIR = os.path.join(KH_APP_DATA_DIR, "gradio_tmp")
    os.environ["GRADIO_TEMP_DIR"] = GRADIO_TEMP_DIR

# Launch the Gradio web interface with specific configurations
demo.queue().launch(
    favicon_path=app._favicon,  # Set the browser tab icon
    inbrowser=True,            # Automatically open in browser
    allowed_paths=[            # Directories that Gradio can access
        "libs/ktem/ktem/assets",  # For UI assets
        GRADIO_TEMP_DIR,         # For temporary files
    ],
    share=KH_GRADIO_SHARE,    # Whether to create a public URL
) 