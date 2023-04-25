
import torch
from rfa_toolbox import create_graph_from_pytorch_model, visualize_architecture

# Model
model_name = "YoloV5s"
# model = torch.hub.load('ultralytics/yolov5', f'{model_name.lower()}')  # or yolov5m, yolov5l, yolov5x, custom
model = torch.hub.load('ultralytics/yolov5', 'custom', 'yolov5s.onnx')  # or yolov5m, yolov5l, yolov5x, custom
model.train()
graph = create_graph_from_pytorch_model(model.cpu(), (4, 3, 640, 360))
visualize_architecture(graph, model_name=f"{model_name}", input_res=10000).render(f"{model_name}")