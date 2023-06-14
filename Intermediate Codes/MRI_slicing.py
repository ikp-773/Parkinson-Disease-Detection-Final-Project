from skimage.transform import resize
import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import nibabel as nib
from tensorflow.keras.models import load_model
from PIL import Image


model_path = 'slice_mri_model.h5'
model = load_model(model_path)

num_slices=10


def load_and_preprocess_mri(path):
    # Load the MRI image
    mri = nib.load(path).get_fdata()

    slices = []



    mid_slice = mri.shape[2] // 2
    for i in range(-(num_slices//2), (num_slices//2)if (num_slices % 2 == 0) else (num_slices//2)+1):
        slice_data = np.squeeze(mri[:, :, mid_slice+i])
        slice_data = (slice_data - np.min(slice_data)) / \
            (np.max(slice_data) - np.min(slice_data)) * 255  # normalize
        slice_data = slice_data.astype(np.uint8)
        slice_img = Image.fromarray(slice_data)

        # Resize the slice to (256, 256)
        slice_img = slice_img.resize( (256, 256))

        # Convert the image to grayscale format
        slice_data = np.array(slice_img.convert('L'))
        # Normalize pixel values to the range [0, 1]
        slices.append(slice_data)

        # Convert slice_data to 3D tensor
        # slice_3d = np.expand_dims(slice_img, axis=-1)
    volume = np.stack(slices, axis=0)[..., np.newaxis]
    volume = volume[np.newaxis, ...]


    return volume


def load_mri():
    path = filedialog.askopenfilename()
    slices = load_and_preprocess_mri(path)
    # slices = np.expand_dims(slices, axis=0)
    # prediction = model.predict(slices)
   
    predictions = model.predict(slices)
    conf_avg = np.mean(np.max(predictions, axis=1) * 100)
    result = "Parkinson's" if conf_avg > 0.7 else "Control"
    result_label.config(text=f"Prediction: {result}")



root = tk.Tk()

load_button = tk.Button(root, text="Load MRI", command=load_mri)
load_button.pack()

result_label = tk.Label(root, text="")
result_label.pack()

root.mainloop()
