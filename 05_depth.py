from ultralytics import YOLO

# Load a model
model = YOLO("yolo26n-depth.pt")  # load an official model
# model = YOLO("path/to/best.pt")  # load a custom model

# Predict with the model
results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image

# Access the results
for result in results:
    depth_map = result.depth.data.cpu().numpy()  # torch.Tensor -> NumPy float32, shape (H, W), meters
    result.show()  # display to screen
    result.save(filename="result.jpg")  # save to disk