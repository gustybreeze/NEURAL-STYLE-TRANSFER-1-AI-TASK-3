# NEURAL-STYLE-TRANSFER-1-AI-TASK-3

*COMPANY*: CODTECH IT SOLUTIONS PVT.LTD

*NAME*: SAMEER KUMAR MISHRA

*INTERN ID*: CT04DZ379

*DOMAIN*: PYTHON PROGRAMMING

*DURATION*: 4 WEEKS

*MENTOR*: NEELA SANTHOSH KUMAR



**Overview**
Neural Style Transfer (NST) is an exciting deep learning technique that combines the content of one image with the style of another to generate a new, visually appealing image. In this task, we implemented a system that uses a pre-trained Convolutional Neural Network (CNN) (e.g., VGG19) to perform this artistic transformation.

This technique demonstrates how neural networks can be used creatively in art generation, image manipulation, and design automation.


**Objective**
- Understand and apply the concept of style transfer using deep learning.

- Use content and style images to create a fused, stylized image.

- Leverage a pre-trained model to extract high-level image features.

- Implement the optimization loop to minimize style and content loss.


**Tools and Technologies**
- Python 3.x

- TensorFlow / PyTorch (depending on implementation)

- NumPy

- Matplotlib

- PIL (Python Imaging Library) or OpenCV

- Pre-trained VGG19 model


**Folder Structure**
bash
Copy
Edit
task_3_neural_style_transfer/
│
├── style_transfer.py          # Main code for neural style transfer
├── content.jpg                # Content image
├── style.jpg                  # Style reference image
├── output.jpg                 # Generated output image
├── README.md                  # Project description and instructions
└── requirements.txt           # Dependencies


**How It Works**

*Input Images:*

- A content image (e.g., a landscape or a building).

- A style image (e.g., a painting by Van Gogh or Picasso).

*Feature Extraction:*

The model uses a pre-trained VGG19 CNN to extract feature representations of content and style layers.

*Loss Calculation:*

- Content Loss: Difference between content features of the content image and the generated image.

- Style Loss: Difference between Gram matrices (correlation of features) of the style image and the generated image.

*Optimization:*

An iterative optimization (using gradient descent) is performed to adjust the pixels of the generated image to minimize the total loss.

*Output:*

A new image is generated that preserves the structure of the content image and mimics the style of the reference artwork.


**Example Output**
- Content Image

- Style Image

- Generated Image


**Installation**
Install the required dependencies using pip:

bash
Copy
Edit
pip install -r requirements.txt
Or install individually:

bash
Copy
Edit
pip install tensorflow numpy matplotlib pillow

Run the Program
bash
Copy
Edit
python style_transfer.py


**Customization**
To use your own images:

- Replace content.jpg and style.jpg with your own.

- Make sure they are in the same folder as the code.

- Optionally, adjust hyperparameters like:

- Style vs content weight

- Image resolution

- Number of iterations


**Learning Outcomes**
Learned about convolutional layers, feature maps, and Gram matrices.

Understood the concept of transfer learning using pre-trained CNNs.

Explored the use of optimization in image generation.

Gained experience working with TensorFlow/Keras/PyTorch for real-world image tasks.


**Conclusion**

Neural Style Transfer is a fascinating application of deep learning where art meets AI. This task helped reinforce concepts of CNNs, transfer learning, and creative machine learning, while also providing practical experience with code implementation and tuning model parameters for artistic effect.


**Output**
![Image](https://github.com/user-attachments/assets/00084de0-8209-4f75-af44-f0a176b8af2d)
content_image
![Image](https://github.com/user-attachments/assets/dc1de024-4c56-425e-9d24-b676bde33c92)
style_image
![Image](https://github.com/user-attachments/assets/3374aa6c-5479-4b00-b5cc-0439396d2635)
stylized_output

