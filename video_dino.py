from grounding_dino import load_model, run_dino
import cv2
import os

def run_dino_video(model, operation_id, phrase):
    vidcap = cv2.VideoCapture('gopro.mp4')
    out = cv2.VideoWriter(f'static/video/out{operation_id}.avi', cv2.VideoWriter_fourcc(*'XVID'), 52.39, (848,480))
    success,image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite("1.jpg", image)     # save frame as JPEG file
        success,image = vidcap.read()
        print('Read a new frame: ', count)
        count += 1
        result = run_dino(model, "1.jpg", phrase, semb=True)
        if result:
            frame = cv2.imread(os.path.join("static", "grounding_dino_output.jpg"))
            out.write(frame)
            print(result)
        else:
            out.write(image)

if __name__ == "__main__":
    config_file = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"  # change the path of the model config file
    grounded_checkpoint = "groundingdino_swint_ogc.pth" # change the path of the model
    device = "cuda"
    # load model
    model = load_model(config_file, grounded_checkpoint, device=device)
    run_dino_video(model, "0", "paper cup . paper . cardboard . plastic bottle . plastic cup . food waste . wood . can . foil . glass bottle . white plastic bottle . green glass bottle")