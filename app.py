import os
import streamlit as st
import subprocess
import os.path as osp
import glob
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch
from PIL import Image

device = torch.device('cpu')
def process_image(image_path, model):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)

    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round()
    return output.astype(np.uint8)

def main():
    st.title("Upload image here for better resolution")

    # Create a directory to store uploaded images
    if not os.path.exists("LR"):
        os.makedirs("LR")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image_path = os.path.join("LR", uploaded_file.name)
        with open(image_path, "wb") as f:
            f.write(uploaded_file.read())
        
        st.success("Image uploaded and saved successfully!")
        st.image(image_path, caption="Uploaded Image", use_column_width=True)
        
        # Load the model
        model_path = 'models/RRDB_ESRGAN_x4.pth'
        device = torch.device('cpu')
        model = arch.RRDBNet(3, 3, 64, 23, gc=32)
        model.load_state_dict(torch.load(model_path), strict=True)
        model.eval()
        model = model.to(device)
        
        # Process the image
        result_image = process_image(image_path, model)
        
        # Save and display the result image
        result_image_path = f'results/{os.path.splitext(uploaded_file.name)[0]}_rlt.png'
        cv2.imwrite(result_image_path, result_image)
        st.image(result_image_path, caption="Enhanced Image", use_column_width=True)
        st.success(f"Enhanced image saved to result: {result_image_path}")

if __name__ == "__main__":
    main()

