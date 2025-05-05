import torch
from torchvision import models, transforms
from torch.utils.mobile_optimizer import optimize_for_mobile

model = models.resnet50(num_classes=5)
model.fc = torch.nn.Sequential(
    torch.nn.Dropout(p=0.3),  # Add dropout for regularization
    torch.nn.Linear(model.fc.in_features, 5)  # Replace the final layer to match the number of classes
)


#If GPU is available, load the model to GPU
if torch.cuda.is_available():
    model = model.to('cuda')
else:
    model = model.to('cpu')


model.eval()  # Set the model to evaluation mode

# Load the model weights
model_root = "models"
model_name = "yw0"
model_epoch = 10

model_path = f"{model_root}/{model_name}/epoch_{model_epoch}.pth"
state = torch.load(model_path, map_location='cpu')
model.load_state_dict(state['model_state_dict'])

example = torch.rand(1, 3, 224, 224)
traced_script_module = torch.jit.trace(model, example)
traced_script_module_optimized = optimize_for_mobile(traced_script_module)
traced_script_module_optimized._save_for_lite_interpreter(f"{model_root}/{model_name}/epoch_{model_epoch}_optimized.ptl")
