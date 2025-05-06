#imports
import torch
from torchvision import models, transforms
from PIL import Image
import urllib

# Initialize the model
model = models.resnet50(num_classes=5)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model weights
model_root = "models"
model_name = "Ywl"

model_path = f"{model_root}/{model_name}.pth"
state = torch.load(model_path, map_location='cpu')

classes = state['class_names']
model.fc = torch.nn.Sequential(
    torch.nn.Dropout(p=0.3),  # Add dropout for regularization
    torch.nn.Linear(model.fc.in_features, len(classes))  # Replace the final layer to match the number of classes
)

model.load_state_dict(state['model_state_dict'])
model = model.to(device)
model.eval()  # Set the model to evaluation mode

# Define an image preprocessing function
# The model expects images of size 224x224 and normalized with ImageNet statistics
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet means
        std=[0.229, 0.224, 0.225]    # ImageNet stds
    )
])

# Load an image from a URL image address

# r&d pub
image_url = "https://www.dropbox.com/scl/fi/0dn4c40ppt1l270mq9lsb/20250426_160700.jpg?rlkey=n4mgs5vdjjib14itb0ihwumvq&st=7nnrkmzn&raw=1"

# patio
# image_url = "https://www.dropbox.com/scl/fi/pktsbgdvh42a9wvoidtsp/20250426_161320-0.jpg?rlkey=12oos5fbn4qkvmcnk29uhgjgu&st=1u6glbd4&raw=1"

#entrance
# image_url = "https://www.dropbox.com/scl/fi/kfczevc4s95b5zqdr5lgl/20250426_160130.jpg?rlkey=gl6wz14xc60v9jvcvzfjqfhhl&st=kbny6gld&raw=1"

#elevator
# image_url = "https://www.dropbox.com/scl/fi/aw8c1uob1910ogso79dhj/20250426_160254.jpg?rlkey=u069ul42m6ndw6odlz1do2pw8&st=my1vzae9&raw=1"

#stata
# image_url = "https://www.dropbox.com/scl/fi/nl0r88rws0idit3bejrg3/20250426_155636-1.jpg?rlkey=09c84u1gcgealygw1kg45z0th&st=p6on4q2q&raw=1"

#cafeteria
# image_url = "https://www.dropbox.com/scl/fi/9nfak80ztiu7t55pwp6zo/20250426_162809.jpg?rlkey=w4lwpou74qkd61bi6vstj0d13&st=sesmao71&raw=1"
img = Image.open(urllib.request.urlopen(image_url)).convert('RGB')

input_img = transform(img).unsqueeze(0)

#Run the model on the input image
# Forward pass
with torch.no_grad():
    input_img = input_img.to('cuda') if torch.cuda.is_available() else input_img.to('cpu')
    logits = model(input_img)
    probs = torch.nn.functional.softmax(logits, dim=1)

# Get the top-5 predictions
top5_probs, top5_idxs = probs.topk(5)

print("Top 5 predicted places:")
for prob, idx in zip(top5_probs[0], top5_idxs[0]):
    print(f"{classes[idx]} ({prob.item():.4f})")