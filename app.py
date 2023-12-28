import logging
import os
import urllib

from PIL import Image

import config
from flask import Flask, request
from flask_cors import CORS, cross_origin
from flask_restful import Api, Resource
from grounding_dino import load_model, run_dino
from video_dino import run_dino_video
from automatic_label_ram import ram_json
from Tag2Text.models import tag2text
from segment_anything import (
    build_sam,
    SamPredictor
) 

project_dir = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG,
                    format='[%(asctime)s]: {} %(levelname)s %(message)s'.format(os.getpid()),
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[logging.StreamHandler()])
cors = CORS(app, resources={r"/*": {"origins": "*"}})
app.config.update(
    CORS_HEADERS='Content-Type'
)

logger = logging.getLogger()

api = Api(prefix=config.API_PREFIX)
config_file = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"  # change the path of the model config file
grounded_checkpoint = "groundingdino_swint_ogc.pth" # change the path of the model
device = "cuda"
model = load_model(config_file, grounded_checkpoint, device=device)

# load model
ram_checkpoint = "./Tag2Text/ram_swin_large_14m.pth"
ram_model = tag2text.ram(pretrained=ram_checkpoint,
                                    image_size=384,
                                    vit='swin_l')
ram_model.eval()
ram_model = ram_model.to(device)
sam_checkpoint = "sam_vit_h_4b8939.pth"
predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))

class DinoDetectAPIView(Resource):
    """POST API class"""
    @cross_origin()
    def post(self):
        """
        (POST)

        upload: <urlstr/image>
        phrase: <str>

        """
        res = {
            "results": {},
            "errors": {},
            "success": False
        }
        data = request.form
        app.logger.info('new detect')
        try:
            upload = data["upload"]
            app.logger.info(upload)
            filename = urllib.parse.urlparse(upload).path.split("/")[-1]
            if not os.path.splitext(filename)[1]:
                filename += ".png"
            path = os.path.join(config.MEDIA_ROOT, filename)
            try:
                urllib.request.urlretrieve(upload, path)
            except:
                res["errors"]["sync"] = f"001: Could not load image from url: {upload}"
                return res
        except:
            upload = request.files["upload"]
            filename = upload.filename
            path = os.path.join(config.MEDIA_ROOT, filename)
            upload.save(path)

        try:
            Image.open(path)
        except:
            res["errors"]["sync"] = "002: Invalid img file"
            return res

        res["results"] = run_dino(model, path, data["phrase"])

        res["success"] = True

        os.remove(path)

        app.logger.info(res)
        return res

class RAMAPIView(Resource):
    """POST API class"""
    @cross_origin()
    def post(self):
        """
        (POST)

        upload: <urlstr/image>

        """
        res = {
            "results": {},
            "errors": {},
            "success": False
        }
        data = request.form
        app.logger.info('new RAM')
        try:
            upload = data["upload"]
            app.logger.info(upload)
            filename = urllib.parse.urlparse(upload).path.split("/")[-1]
            if not os.path.splitext(filename)[1]:
                filename += ".png"
            path = os.path.join(config.MEDIA_ROOT, filename)
            try:
                urllib.request.urlretrieve(upload, path)
            except:
                res["errors"]["sync"] = f"001: Could not load image from url: {upload}"
                return res
        except:
            upload = request.files["upload"]
            filename = upload.filename
            path = os.path.join(config.MEDIA_ROOT, filename)
            upload.save(path)

        try:
            Image.open(path)
        except:
            res["errors"]["sync"] = "002: Invalid img file"
            return res
        res["results"] = ram_json(model, path, ram_model, predictor)

        res["success"] = True

        os.remove(path)

        app.logger.info(res)
        return res

class SembAPIView(Resource):
    """POST API class"""
    @cross_origin()
    def post(self):
        """
        (POST)

        phrase: <str>

        """
        data = request.form
        app.logger.info('new semb')
        operation_id = len(list(os.listdir("static/video")))
        run_dino_video(model, operation_id, data["phrase"])
        return f"Sucessful, id: {operation_id}"


api.add_resource(DinoDetectAPIView, '/detect')
api.add_resource(RAMAPIView, '/ram')
api.add_resource(SembAPIView, '/semb')
api.init_app(app)

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=8005, debug=True)