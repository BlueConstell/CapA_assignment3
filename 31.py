
import jetson.inference
import jetson.utils
from PIL import Image, ImageDraw



image_path="/home/nvidia/jetson-inference/data/images/dog_2.jpg"
img = jetson.utils.loadImage(image_path)

net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
detections = net.Detect(img)

image = Image.open(image_path)
outcome = ImageDraw.Draw(image)


display = jetson.utils.videoOutput("display://0")

if img is not None and detections is not None:

    display.Render(img)
    display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))
    for detection in detections:
        print(f"ClassID:{detection.ClassID}")
        print(f"Confidence:{detection.Confidence}")
        print(f"Left:{detection.Left}")
        print(f"Top:{detection.Top}")
        print(f"Right:{detection.Right}")
        print(f"Bottom:{detection.Right}")
        print(f"Width:{detection.Width}")
        print(f"Height:{detection.Height}")
        print(f"Area:{detection.Area}")
        print(f"Center:{detection.Center}")
    
outcome.rectangle((detection.Left, detection.Top, detection.Right, detection.Bottom), outline=(255,0,0), width=2)
    
    
label = f"{net.GetClassDesc(detection.ClassID)}: {detection.Confidence:.2f}"
outcome.text((detection.Left, detection.Top - 10), label, fill=(255,0,0))


output_path = "/home/nvidia/jetson-inference/data/outcome33.jpg"
image.save(output_path)

print(f"Annotated image saved to {output_path}")
    