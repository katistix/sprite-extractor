# @katistix/sprite-extractor 🎮✂️

Got some cool hand-drawn game assets on paper? Want to digitize them effortlessly? This tool lets you upload a photo, and it will automatically crop out and remove backgrounds from your objects — turning your pencil art into sprites for your game! 🎨✨

---

## Features 🌟
- **Upload an image** of your hand-drawn game assets 📸.
- **Adjust the threshold** to fine-tune object detection using an easy slider 🎚️.
- **Real-time preview** with object bounding boxes 🟩.
- **Extract and export** your assets as PNG files 🖼️.

---

## Installation 🚀

1. **Clone the repo**:

   ```bash
   git clone https://github.com/katistix/sprite-extractor.git
   cd sprite-extractor
   ```

2. **Set up a virtual environment** (because we love isolation 😎):

   ```bash
   python3 -m venv venv
   ```

3. **Activate your venv**:

   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
   - On Windows:
     ```bash
     .\venv\Scripts\activate
     ```

4. **Install the dependencies** from `requirements.txt`:

   ```bash
   pip3 install -r requirements.txt
   ```

---

## How to Use 🔥

1. **Run the app**:

   ```bash
   python3 main.py
   ```

2. **Upload your image** 🖼️:
   - Click **"Upload Image"** and choose a photo of your hand-drawn game assets.

3. **Adjust the threshold** slider 🔲:
   - Fine-tune the detection of objects in your image.

4. **Preview the magic** 👀:
   - See the app draw bounding boxes around detected objects and show you a count of them.

5. **Export your assets** 💾:
   - Click **"Extract/Export"** to save your objects as PNGs to a folder of your choice.

---

## Notes 📝
- Designed for macOS users, may not work on other platforms (though you're welcome to try 😅).
- Best results when the photo is clear and has good contrast. 📸👌

---

## License 📄

MIT License. Check out the full details in the [LICENSE](LICENSE) file.